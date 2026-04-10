from __future__ import annotations

import json
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

        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        image = image.resize(target_size)
        image_array = np.asarray(image, dtype="float32")
        image_array = self._preprocess(image_array, tf)
        batch = np.expand_dims(image_array, axis=0)

        try:
            prediction = model.predict(batch, verbose=0)[0].tolist()
        except Exception as exc:
            raise InferenceError("Không thể chạy CNN để phân loại tình trạng lá.") from exc

        if not prediction:
            raise InferenceError("CNN không trả về xác suất phân loại hợp lệ.")

        labels = self._load_labels(len(prediction))
        ranked = sorted(
            (
                {
                    "label": labels[index],
                    "display_label": self._humanize_label(labels[index]),
                    "confidence": round(float(score), 4),
                }
                for index, score in enumerate(prediction)
            ),
            key=lambda item: item["confidence"],
            reverse=True,
        )

        top_prediction = ranked[0]
        return {
            "label": top_prediction["label"],
            "display_label": top_prediction["display_label"],
            "confidence": top_prediction["confidence"],
            "top_predictions": ranked[:5],
            "input_size": {"width": target_size[0], "height": target_size[1]},
            "fallback": False,
            "warning": "",
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
            "input_size": {"width": 300, "height": 300},
            "fallback": True,
            "warning": message,
        }
