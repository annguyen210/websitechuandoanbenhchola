from __future__ import annotations

import json
import math
from pathlib import Path
from threading import Lock

from PIL import Image, ImageOps

from services.config import Settings
from services.exceptions import InferenceError


class CnnClassificationService:
    _model = None
    _model_lock = Lock()

    def __init__(self, settings: Settings):
        self.settings = settings

    def _load_tensorflow(self):
        try:
            import numpy as np
            import tensorflow as tf
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
        except Exception:
            return None, None
        return np, tf

    def _load_model(self):
        _, tf = self._load_tensorflow()
        if tf is None or not self.settings.cnn_model_path.exists():
            return None

        if self.__class__._model is None:
            with self.__class__._model_lock:
                if self.__class__._model is None:
                    try:
                        self.__class__._model = tf.keras.models.load_model(
                            str(self.settings.cnn_model_path),
                            compile=False,
                        )
                    except Exception:
                        try:
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
                                skip_mismatch=True,
                            )
                            self.__class__._model = model
                        except Exception:
                            return None

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
        """3 biến thể TTA hợp lệ cho lá sắn: gốc, lật ngang, lật dọc."""
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
        """Shannon entropy (nats). Cao → model phân vân, ảnh có thể không phải lá sắn."""
        eps = 1e-9
        return -sum(p * math.log(p + eps) for p in probs)

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

        # Đọc ảnh gốc nguyên bản — không tăng cường trước (model train trên ảnh tự nhiên)
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

        # Trung bình TTA — ổn định hơn single-pass
        averaged = np.mean(all_predictions, axis=0).tolist()

        # Entropy để phát hiện ảnh không liên quan đến lá sắn
        entropy_val = self._entropy(averaged)
        n_classes = len(averaged)
        max_entropy = math.log(n_classes)       # ln(5) ≈ 1.609
        entropy_ratio = entropy_val / max_entropy  # 0 = chắc chắn, 1 = hoàn toàn phân vân

        labels = self._load_labels(n_classes)
        ranked = sorted(
            (
                {
                    "label": labels[i],
                    "display_label": self._humanize_label(labels[i]),
                    "confidence": round(float(score), 4),
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
            # Phân phối gần đồng đều → model không nhận ra lá sắn
            warning = (
                f"Phân phối xác suất quá đều (entropy {entropy_ratio * 100:.0f}% mức tối đa). "
                "Ảnh có thể không phải lá sắn hoặc chất lượng quá kém. "
                "Kết quả KHÔNG đáng tin cậy — hãy dùng ảnh lá sắn rõ nét."
            )
        elif max_conf < 0.40:
            warning = (
                f"Độ tin cậy rất thấp ({max_conf * 100:.1f}%). "
                "Ảnh có thể không phải lá sắn hoặc bị che khuất, mờ. Nên kiểm tra thực tế."
            )
        elif max_conf < 0.60:
            warning = (
                f"Độ tin cậy trung bình ({max_conf * 100:.1f}%). "
                "Nên chụp lại ảnh rõ hơn, đủ ánh sáng, tập trung vào lá sắn."
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
            "fallback": False,
            "warning": warning,
        }

    def _humanize_label(self, label: str) -> str:
        return label.replace("-", " ").replace("_", " ").strip().title()

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
