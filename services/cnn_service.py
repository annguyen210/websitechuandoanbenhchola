# ============================================================
# File: services/cnn_service.py
# Vai trò: BƯỚC 2 trong pipeline — Phân loại nhóm bệnh lá bằng CNN
# Xử lý CNN, phân loại 5 nhóm bệnh, tính phần trăm, cảnh báo độ tin cậy.


# Luồng xử lý chính:
#   1. Nạp model EfficientNetB3 từ file model_0.h5
#   2. Đọc ảnh lá (từ YOLO crop hoặc toàn ảnh gốc)
#   3. Chạy TTA (Test-Time Augmentation) — 3 biến thể: gốc, lật ngang, lật dọc
#   4. Tính xác suất trung bình từ 3 lần dự đoán (model_probs)
#   5. Phân tích màu sắc RGB của ảnh để tạo visual_probs (visual symptom scores)
#   6. Calibrate: kết hợp model_probs + visual_probs theo trọng số phụ thuộc entropy
#   7. Sharpen phân phối cuối để nhóm chiếm cao nhất rõ ràng hơn
#   8. Trả về nhóm bệnh chiếm tỉ lệ % cao nhất và danh sách top 5
#
# 5 nhóm bệnh được phân loại:
#   - healthy: Khỏe mạnh
#   - cassava_bacterial_blight (CBB): Bệnh đốm lá do vi khuẩn
#   - cassava_brown_streak_disease (CBSD): Bệnh vằn/rỉ sắt
#   - cassava_green_mottle (CGM): Bệnh đốm xanh nhạt
#   - cassava_mosaic_disease (CMD): Bệnh khảm vàng-xanh
# ============================================================

from __future__ import annotations

import json
import math
from pathlib import Path
from threading import Lock

from PIL import Image, ImageOps

from services.config import Settings
from services.exceptions import InferenceError


class CnnClassificationService:
    # Model được lưu ở cấp class để dùng chung giữa các request (singleton)
    _model = None
    _model_lock = Lock()

    def __init__(self, settings: Settings):
        self.settings = settings

    def _load_tensorflow(self):
        """
        Import numpy và TensorFlow theo kiểu lazy (chỉ khi cần).
        Giới hạn số luồng CPU để tránh deadlock trên server ít vCPU.
        Trả về (None, None) nếu TensorFlow chưa được cài.
        """
        try:
            import numpy as np
            import tensorflow as tf
            # Giới hạn thread pool TF để tránh tranh chấp tài nguyên
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
        except Exception:
            return None, None
        return np, tf

    def _load_model(self):
        """
        Nạp model CNN từ file model_0.h5 với 2 chiến lược:
          1. Thử tf.keras.models.load_model() trực tiếp (ưu tiên)
          2. Nếu lỗi → rebuild kiến trúc EfficientNetB3 thủ công
             rồi load_weights() theo tên layer (skip_mismatch=True)
        Dùng double-checked locking để thread-safe trong môi trường WSGI.
        """
        _, tf = self._load_tensorflow()
        if tf is None or not self.settings.cnn_model_path.exists():
            return None

        if self.__class__._model is None:
            with self.__class__._model_lock:
                if self.__class__._model is None:
                    try:
                        # Cách 1: Load toàn bộ model (kiến trúc + trọng số)
                        self.__class__._model = tf.keras.models.load_model(
                            str(self.settings.cnn_model_path),
                            compile=False,
                        )
                    except Exception:
                        try:
                            # Cách 2: Rebuild kiến trúc EfficientNetB3 thủ công
                            # rồi load chỉ trọng số theo tên layer
                            base = tf.keras.applications.EfficientNetB3(
                                include_top=False,
                                weights=None,
                                input_shape=(300, 300, 3),
                                pooling="avg",
                                name="efficientnet-b3",
                            )
                            x = tf.keras.layers.Dropout(0.5, name="dropout")(base.output)
                            out = tf.keras.layers.Dense(5, activation="softmax", name="output")(x)
                            model = tf.keras.Model(inputs=base.input, outputs=out)
                            model.load_weights(
                                str(self.settings.cnn_model_path),
                                by_name=True,
                                skip_mismatch=True,  # Bỏ qua layer không khớp tên
                            )
                            self.__class__._model = model
                        except Exception:
                            return None  # Cả 2 cách đều thất bại → chạy fallback

        return self.__class__._model

    def _load_labels(self, expected_count: int) -> list[str]:
        if not self.settings.cnn_labels_path.exists():
            return [f"class_{i}" for i in range(expected_count)]

        with open(self.settings.cnn_labels_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict):
            labels = raw.get("labels", [])
        elif isinstance(raw, list):
            labels = raw
        else:
            labels = []

        normalized = [str(lb).strip() for lb in labels if str(lb).strip()]
        if len(normalized) == expected_count:
            return normalized

        return [
            normalized[i] if i < len(normalized) else f"class_{i}"
            for i in range(expected_count)
        ]

    def _get_tta_variants(self, image: Image.Image, target_size: tuple) -> list[Image.Image]:
        """3 biến thể TTA: gốc, lật ngang, lật dọc."""
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS  # type: ignore[attr-defined]

        base = image.resize(target_size, resample)
        return [
            base,
            base.transpose(Image.FLIP_LEFT_RIGHT),
            base.transpose(Image.FLIP_TOP_BOTTOM),
        ]

    def _preprocess(self, image_array, tf):
        """EfficientNet preprocessing — chuẩn khớp với quá trình training EfficientNetB3."""
        arr = image_array.copy()
        return tf.keras.applications.efficientnet.preprocess_input(arr)

    def _entropy(self, probs: list[float]) -> float:
        """Shannon entropy (nats). Cao → model phân vân, ảnh có thể không rõ hoặc không liên quan."""
        eps = 1e-9
        return -sum(p * math.log(p + eps) for p in probs)

    def _normalize_probs(self, probs: list[float]) -> list[float]:
        total = float(sum(max(0.0, p) for p in probs))
        if total <= 0:
            return [1.0 / len(probs) for _ in probs]
        return [max(0.0, float(p)) / total for p in probs]

    def _visual_symptom_scores(self, image: Image.Image, labels: list[str], np) -> tuple[list[float], dict]:
        """Phân tích triệu chứng bệnh lá bằng pure numpy RGB — không dùng PIL HSV để tránh lỗi conversion."""
        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR  # type: ignore[attr-defined]

        sample = image.resize((224, 224), resample).convert("RGB")
        rgb = np.asarray(sample, dtype="float32") / 255.0

        r_ch = rgb[:, :, 0]
        g_ch = rgb[:, :, 1]
        b_ch = rgb[:, :, 2]

        # Tính saturation và value từ RGB thủ công (đáng tin hơn PIL HSV)
        max_c = np.maximum(np.maximum(r_ch, g_ch), b_ch)
        min_c = np.minimum(np.minimum(r_ch, g_ch), b_ch)
        sat = np.where(max_c > 0.05, (max_c - min_c) / (max_c + 1e-8), 0.0)
        val = max_c

        # Mask vùng lá: loại nền trắng/đen, giữ pixel có màu
        leaf_mask = (val > 0.08) & (val < 0.96) & ((g_ch > 0.08) | (sat > 0.10))
        leaf_pixels = int(np.count_nonzero(leaf_mask))
        if leaf_pixels < 200:
            uniform = [1.0 / len(labels) for _ in labels]
            return uniform, {"quality": "low", "message": "Vùng lá quá nhỏ hoặc ảnh quá tối/sáng."}

        def pct(mask) -> float:
            return float(np.count_nonzero(mask & leaf_mask)) / leaf_pixels

        # Healthy: xanh lá thuần, bão hòa, không có vết bệnh
        healthy_px = (g_ch > r_ch + 0.04) & (g_ch > b_ch + 0.04) & (sat > 0.18) & (val > 0.18) & (val < 0.88)
        green_ratio = pct(healthy_px)

        # CMD (Mosaic — Bệnh khảm): vàng-xanh đốm — R≈G cao, B thấp
        yellow_px = (r_ch > 0.38) & (g_ch > 0.38) & (b_ch < 0.30) & (np.abs(r_ch - g_ch) < 0.18)
        yellow_ratio = pct(yellow_px)
        # Độ biến thiên màu cao = pattern khảm
        g_leaf = g_ch[leaf_mask]
        r_leaf = r_ch[leaf_mask]
        color_var = float((np.std(g_leaf) + np.std(r_leaf)) / 2.0) if len(g_leaf) > 0 else 0.0

        # CGM (Green Mottle — Đốm xanh): xanh nhạt, bão hòa thấp
        pale_px = (g_ch > r_ch * 1.01) & (g_ch > b_ch * 1.01) & (sat < 0.28) & (sat > 0.04) & (val > 0.36) & (val < 0.82)
        pale_ratio = pct(pale_px)

        # CBB (Bacterial Blight — Đốm lá vi khuẩn): nâu tối, góc lá
        # Thêm điều kiện g_ch < b_ch * 1.45 để loại trừ pixel cam/rỉ sắt của CBSD (vốn có g >> b)
        brown_px = (r_ch > g_ch + 0.04) & (r_ch > b_ch + 0.08) & (g_ch < b_ch * 1.45) & (val > 0.10) & (val < 0.72) & (sat > 0.14)
        brown_ratio = pct(brown_px)
        dark_px = (val < 0.22) & (sat > 0.07)
        necrosis_ratio = pct(dark_px)

        # CBSD (Brown Streak — Rỉ sắt/sọc nâu): cam-rỉ đậm dọc gân
        rust_px = (r_ch > 0.44) & (r_ch > g_ch * 1.20) & (g_ch > b_ch * 1.10) & (sat > 0.26) & (val > 0.26)
        rust_ratio = pct(rust_px)
        # CBSD nhạt hơn: vùng vàng-cam dọc gân (CBSD sớm hoặc chuyển màu), g >> b là đặc điểm phân biệt với CBB
        cbsd_light_px = (r_ch > 0.38) & (g_ch > 0.25) & (b_ch < 0.22) & (r_ch > g_ch * 1.12) & (g_ch > b_ch * 1.40) & (val > 0.25)
        cbsd_light_ratio = pct(cbsd_light_px)

        lesion_total = min(1.0, brown_ratio + rust_ratio + cbsd_light_ratio * 0.5 + necrosis_ratio)
        mottling_total = min(1.0, yellow_ratio + pale_ratio)

        # Tính điểm cơ bản từ đặc trưng màu sắc pixel
        scores = {
            "healthy": max(0.01, green_ratio * 1.50 - lesion_total * 2.20 - yellow_ratio * 1.10),
            "cassava_bacterial_blight": max(0.01, brown_ratio * 2.70 + necrosis_ratio * 2.20 + yellow_ratio * 0.15),
            "cassava_brown_streak_disease": max(0.01, rust_ratio * 3.10 + cbsd_light_ratio * 1.40 + brown_ratio * 0.20),
            "cassava_green_mottle": max(0.01, pale_ratio * 2.40 + color_var * 0.50),
            "cassava_mosaic_disease": max(0.01, yellow_ratio * 2.20 + color_var * 1.60 + pale_ratio * 0.25),
        }

        # Boost rõ ràng cho từng pattern đặc trưng
        if green_ratio > 0.62 and lesion_total < 0.04 and yellow_ratio < 0.07:
            scores["healthy"] += 0.55
        if yellow_ratio > 0.10 and lesion_total < 0.10:
            scores["cassava_mosaic_disease"] += 0.50
            scores["cassava_green_mottle"] += 0.10
        if pale_ratio > 0.14 and yellow_ratio < 0.12 and lesion_total < 0.08:
            scores["cassava_green_mottle"] += 0.45
        if brown_ratio + necrosis_ratio > 0.05:
            scores["cassava_bacterial_blight"] += 0.55
        if rust_ratio > 0.03 or cbsd_light_ratio > 0.06:
            scores["cassava_brown_streak_disease"] += 0.65
        # Phân biệt CBSD vs CBB khi cả hai đều xuất hiện: cam/rỉ nhiều hơn nâu → CBSD
        if (rust_ratio + cbsd_light_ratio) > brown_ratio and (rust_ratio > 0.02 or cbsd_light_ratio > 0.04):
            scores["cassava_brown_streak_disease"] += 0.30
            scores["cassava_bacterial_blight"] = max(0.01, scores["cassava_bacterial_blight"] - 0.15)
        if color_var > 0.08 and yellow_ratio > 0.07:
            scores["cassava_mosaic_disease"] += 0.32

        ordered = [scores.get(label, 0.01) for label in labels]
        visual_probs = self._normalize_probs(ordered)
        top_idx = max(range(len(visual_probs)), key=lambda i: visual_probs[i])
        top_score = visual_probs[top_idx]
        second_score = sorted(visual_probs, reverse=True)[1] if len(visual_probs) > 1 else 0.0
        evidence_strength = min(1.0, max(0.0,
            (top_score - second_score) * 2.5
            + max(lesion_total, mottling_total) * 1.5
            + color_var * 0.5
        ))

        return visual_probs, {
            "quality": "ok",
            "top_label": labels[top_idx],
            "evidence_strength": round(float(evidence_strength), 4),
            "ratios": {
                "green": round(green_ratio, 4),
                "pale_mottle": round(pale_ratio, 4),
                "yellow_mosaic": round(yellow_ratio, 4),
                "brown_spot": round(brown_ratio, 4),
                "rust_orange": round(rust_ratio, 4),
                "cbsd_light": round(cbsd_light_ratio, 4),
                "dark_necrosis": round(necrosis_ratio, 4),
                "color_variation": round(color_var, 4),
            },
        }

    def _sharpen_probs(self, probs: list[float], temperature: float = 0.45) -> list[float]:
        """Làm sắc nét phân phối xác suất: winner chiếm % cao hơn, loser thấp hơn."""
        eps = 1e-9
        log_p = [math.log(max(p, eps)) / temperature for p in probs]
        max_lp = max(log_p)
        exp_p = [math.exp(lp - max_lp) for lp in log_p]
        return self._normalize_probs(exp_p)

    def _calibrate_with_visual_symptoms(
        self,
        model_probs: list[float],
        visual_probs: list[float],
        visual_evidence: dict,
        model_entropy_ratio: float = 0.0,
        model_top_conf: float = 0.0,
    ) -> list[float]:
        strength = float(visual_evidence.get("evidence_strength", 0.0) or 0.0)

        if model_entropy_ratio > 0.80:
            # CNN gần như ngẫu nhiên — dựa chủ yếu vào visual
            cnn_w, vis_w = 0.15, 0.85
        elif model_entropy_ratio > 0.60:
            # CNN kém tin cậy — ưu tiên visual
            cnn_w, vis_w = 0.30, 0.70
        elif model_top_conf > 0.72:
            # CNN rất tự tin → giảm ảnh hưởng visual, chỉ dùng visual như hiệu chỉnh nhỏ
            vis_w = 0.12 + min(0.18, strength * 0.25)
            cnn_w = 1.0 - vis_w
        else:
            # CNN ổn định — blend theo strength của visual evidence
            vis_w = 0.22 + min(0.35, strength * 0.50)
            cnn_w = 1.0 - vis_w

        calibrated = [cnn_w * mp + vis_w * vp for mp, vp in zip(model_probs, visual_probs)]
        blended = self._normalize_probs(calibrated)

        # Sharpen nhẹ kết quả cuối để phân nhóm cao nhất rõ ràng hơn
        temperature = 0.60 if model_entropy_ratio > 0.60 else 0.75
        return self._sharpen_probs(blended, temperature=temperature)

    def classify(self, image_path: Path) -> dict:
        np, tf = self._load_tensorflow()
        model = self._load_model()

        if np is None or tf is None or model is None:
            return self._fallback_classification(
                "CNN đang ở chế độ dự phòng vì thiếu TensorFlow hoặc chưa tải được model_0.h5."
            )

        input_shape = getattr(model, "input_shape", None) or (None, 300, 300, 3)
        target_size = (
            int(input_shape[1] or 300),
            int(input_shape[2] or 300),
        )

        original_image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        tta_variants = self._get_tta_variants(original_image, target_size)

        all_predictions: list[list[float]] = []
        for variant in tta_variants:
            arr = np.asarray(variant, dtype="float32")
            arr = self._preprocess(arr, tf)
            batch = np.expand_dims(arr, axis=0)
            try:
                pred = model.predict(batch, verbose=0)[0].tolist()
                all_predictions.append(pred)
            except Exception:
                continue

        if not all_predictions:
            raise InferenceError("CNN không thể chạy phân loại trên ảnh này.")

        model_probs = self._normalize_probs(np.mean(all_predictions, axis=0).tolist())
        n_classes = len(model_probs)

        # Tính entropy của riêng CNN TRƯỚC khi calibrate để đánh giá mức độ tin cậy
        max_entropy = math.log(n_classes)
        model_entropy_ratio = self._entropy(model_probs) / max_entropy

        labels = self._load_labels(n_classes)
        model_top_conf = max(model_probs) if model_probs else 0.0
        visual_probs, visual_evidence = self._visual_symptom_scores(original_image, labels, np)
        averaged = self._calibrate_with_visual_symptoms(
            model_probs, visual_probs, visual_evidence, model_entropy_ratio, model_top_conf
        )

        entropy_val = self._entropy(averaged)
        entropy_ratio = entropy_val / max_entropy
        ranked = sorted(
            (
                {
                    "label": labels[i],
                    "display_label": self._humanize_label(labels[i]),
                    "confidence": round(float(score), 4),
                    "raw_cnn_confidence": round(float(model_probs[i]), 4),
                    "visual_confidence": round(float(visual_probs[i]), 4),
                }
                for i, score in enumerate(averaged)
            ),
            key=lambda item: item["confidence"],
            reverse=True,
        )

        top = ranked[0]
        max_conf = top["confidence"]
        second_conf = ranked[1]["confidence"] if len(ranked) > 1 else 0.0
        conf_gap = max_conf - second_conf

        if entropy_ratio > 0.78:
            warning = (
                f"Phân phối xác suất quá đều (entropy {entropy_ratio * 100:.0f}% mức tối đa). "
                "Ảnh có thể không rõ hoặc chất lượng kém. "
                "Kết quả KHÔNG đáng tin cậy — hãy dùng ảnh lá cây rõ nét, đủ ánh sáng."
            )
        elif max_conf < 0.40:
            warning = (
                f"Độ tin cậy rất thấp ({max_conf * 100:.1f}%). "
                "Ảnh có thể bị che khuất, mờ hoặc chụp từ xa. Nên kiểm tra thực tế."
            )
        elif max_conf < 0.60:
            warning = (
                f"Độ tin cậy trung bình ({max_conf * 100:.1f}%). "
                "Nên chụp lại ảnh rõ hơn, đủ ánh sáng, tập trung vào bề mặt lá."
            )
        elif conf_gap < 0.10:
            warning = (
                f"Hai nhóm bệnh có xác suất gần nhau "
                f"({max_conf * 100:.1f}% vs {second_conf * 100:.1f}%). "
                "Nên kiểm tra thêm thực tế."
            )
        else:
            warning = ""

        return {
            "label": top["label"],
            "display_label": top["display_label"],
            "confidence": top["confidence"],
            "top_predictions": ranked[:5],
            "tta_runs": len(all_predictions),
            "preprocess_mode": "efficientnet",
            "entropy": round(entropy_val, 4),
            "entropy_ratio": round(entropy_ratio, 4),
            "input_size": {"width": target_size[0], "height": target_size[1]},
            "visual_evidence": visual_evidence,
            "calibration_applied": True,
            "fallback": False,
            "warning": warning,
        }

    def _humanize_label(self, label: str) -> str:
        _label_map = {
            "cassava_bacterial_blight": "Cassava Bacterial Blight (Bệnh bạc, cháy lá do vi khuẩn)",
            "cassava_brown_streak_disease": "Cassava Brown Streak Disease (Bệnh vằn, sọc nâu lá)",
            "cassava_green_mottle": "Cassava Green Mottle (Bệnh đốm xanh lá)",
            "cassava_mosaic_disease": "Cassava Mosaic Disease (Bệnh khảm lá)",
            "healthy": "Healthy (Khỏe mạnh)",
        }
        return _label_map.get(label, label)

    def _fallback_classification(self, message: str) -> dict:
        labels = self._load_labels(5)
        ranked = [
            {
                "label": labels[i],
                "display_label": self._humanize_label(labels[i]),
                "confidence": score,
            }
            for i, score in enumerate([0.34, 0.24, 0.18, 0.14, 0.10])
            if i < len(labels)
        ]
        top = ranked[0]
        return {
            "label": top["label"],
            "display_label": top["display_label"],
            "confidence": top["confidence"],
            "top_predictions": ranked,
            "tta_runs": 0,
            "preprocess_mode": "fallback",
            "entropy": 0.0,
            "entropy_ratio": 0.0,
            "input_size": {"width": 300, "height": 300},
            "fallback": True,
            "warning": message,
        }
