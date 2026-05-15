from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from PIL import Image, ImageEnhance, ImageOps

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
                            import tensorflow as tf
                            base_model = tf.keras.applications.EfficientNetB3(
                                include_top=False,
                                weights=None,
                                input_shape=(300, 300, 3),
                                pooling='avg',
                                name='efficientnet-b3'
                            )
                            x = tf.keras.layers.Dropout(0.5, name='dropout')(base_model.output)
                            outputs = tf.keras.layers.Dense(5, activation='softmax', name='output')(x)
                            model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
                            model.load_weights(str(self.settings.cnn_model_path), by_name=True, skip_mismatch=True)
                            self.__class__._model = model
                        except Exception:
                            return None

        return self.__class__._model

    def _load_labels(self, expected_count: int) -> list[str]:
        if not self.settings.cnn_labels_path.exists():
            return [f"class_{index}" for index in range(expected_count)]

        with open(self.settings.cnn_labels_path, "r", encoding="utf-8") as file:
            raw = json.load(file)

        if isinstance(raw, dict):
            labels = raw.get("labels", [])
        elif isinstance(raw, list):
            labels = raw
        else:
            labels = []

        normalized = [str(label).strip() for label in labels if str(label).strip()]
        if len(normalized) == expected_count:
            return normalized

        return [
            normalized[index] if index < len(normalized) else f"class_{index}"
            for index in range(expected_count)
        ]

    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Tăng nhẹ độ tương phản và độ sắc nét để làm nổi bật đặc trưng bệnh lá."""
        image = ImageEnhance.Contrast(image).enhance(1.2)
        image = ImageEnhance.Sharpness(image).enhance(1.1)
        return image

    def _get_tta_images(self, image: Image.Image, target_size: tuple) -> list[Image.Image]:
        """Tạo các phiên bản augment để chạy TTA (Test Time Augmentation)."""
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS  # type: ignore[attr-defined]

        variants = [
            image,
            image.transpose(Image.FLIP_LEFT_RIGHT),
            image.transpose(Image.FLIP_TOP_BOTTOM),
            image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM),
            image.rotate(90, expand=True),
            image.rotate(270, expand=True),
        ]
        return [img.resize(target_size, resample) for img in variants]

    def _preprocess_with_mode(self, image_array, tf, mode: str):
        """Chuẩn hóa mảng ảnh float32 với chế độ preprocessing chỉ định."""
        if mode == "scale_01":
            return image_array / 255.0
        if mode == "efficientnet":
            # preprocess_input có thể sửa in-place nên phải copy trước
            arr = image_array.copy()
            return tf.keras.applications.efficientnet.preprocess_input(arr)
        return image_array / 255.0

    def _run_predictions_with_mode(
        self, model, tta_images: list, np, tf, mode: str
    ) -> list[list[float]]:
        """Chạy toàn bộ TTA forward-pass với một chế độ preprocessing."""
        predictions = []
        for aug_img in tta_images:
            arr = np.asarray(aug_img, dtype="float32")
            arr = self._preprocess_with_mode(arr, tf, mode)
            batch = np.expand_dims(arr, axis=0)
            try:
                pred = model.predict(batch, verbose=0)[0].tolist()
                predictions.append(pred)
            except Exception:
                continue
        return predictions

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
        # Tăng cường ảnh nhẹ để làm nổi bật đặc trưng bệnh trước khi phân loại
        enhanced_image = self._enhance_image(original_image)
        tta_images = self._get_tta_images(enhanced_image, target_size)

        # Thử cả hai chế độ preprocessing — chế độ đúng sẽ cho kết quả tự tin hơn (max conf cao hơn)
        preds_scale01 = self._run_predictions_with_mode(model, tta_images, np, tf, "scale_01")
        preds_efficientnet = self._run_predictions_with_mode(model, tta_images, np, tf, "efficientnet")

        def top_conf(preds: list) -> float:
            return float(np.max(np.mean(preds, axis=0))) if preds else 0.0

        conf_01 = top_conf(preds_scale01)
        conf_eff = top_conf(preds_efficientnet)

        # Chọn chế độ có độ tự tin cao nhất — chế độ preprocessing đúng luôn cho phân phối rõ nét hơn
        if conf_eff >= conf_01:
            all_predictions = preds_efficientnet
            best_mode = "efficientnet"
        else:
            all_predictions = preds_scale01
            best_mode = "scale_01"

        if not all_predictions:
            raise InferenceError("CNN không thể chạy phân loại trên ảnh này.")

        averaged = np.mean(all_predictions, axis=0).tolist()

        labels = self._load_labels(len(averaged))
        ranked = sorted(
            (
                {
                    "label": labels[index],
                    "display_label": self._humanize_label(labels[index]),
                    "confidence": round(float(score), 4),
                }
                for index, score in enumerate(averaged)
            ),
            key=lambda item: item["confidence"],
            reverse=True,
        )

        top = ranked[0]
        max_conf_val = top["confidence"]
        second_conf = ranked[1]["confidence"] if len(ranked) > 1 else 0.0
        conf_gap = max_conf_val - second_conf

        if max_conf_val < 0.40:
            warning = (
                f"Độ tin cậy rất thấp ({max_conf_val * 100:.1f}%). "
                "Ảnh có thể không phải lá cây hoặc chất lượng ảnh kém. Cần kiểm tra thực tế."
            )
        elif max_conf_val < 0.60:
            warning = (
                f"Độ tin cậy thấp ({max_conf_val * 100:.1f}%). "
                "Nên chụp lại ảnh rõ hơn để có kết quả chính xác hơn."
            )
        elif conf_gap < 0.10:
            warning = (
                f"Hai nhóm bệnh có xác suất gần nhau "
                f"({max_conf_val * 100:.1f}% vs {second_conf * 100:.1f}%). "
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
            "preprocess_mode": best_mode,
            "input_size": {"width": target_size[0], "height": target_size[1]},
            "fallback": False,
            "warning": warning,
        }

    def _preprocess(self, image_array, tf):
        mode = self.settings.cnn_preprocess_mode
        if mode == "scale_01":
            return image_array / 255.0
        if mode == "efficientnet":
            return tf.keras.applications.efficientnet.preprocess_input(image_array)
        return image_array

    def _humanize_label(self, label: str) -> str:
        return label.replace("-", " ").replace("_", " ").strip().title()

    def _fallback_classification(self, message: str) -> dict:
        labels = self._load_labels(5)
        ranked = []
        base_scores = [0.34, 0.24, 0.18, 0.14, 0.10]

        for index, label in enumerate(labels[:5]):
            ranked.append(
                {
                    "label": label,
                    "display_label": self._humanize_label(label),
                    "confidence": base_scores[index],
                }
            )

        top_prediction = ranked[0]
        return {
            "label": top_prediction["label"],
            "display_label": top_prediction["display_label"],
            "confidence": top_prediction["confidence"],
            "top_predictions": ranked,
            "tta_runs": 0,
            "preprocess_mode": "fallback",
            "input_size": {"width": 300, "height": 300},
            "fallback": True,
            "warning": message,
        }
