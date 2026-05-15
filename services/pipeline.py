from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from services.cnn_service import CnnClassificationService
from services.config import Settings
from services.exceptions import BadRequestError
from services.llm_service import LlmAdviceService
from services.yolo_service import YoloLeafService


class AnalysisPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.ensure_runtime_directories()
        self.yolo = YoloLeafService(settings)
        self.cnn = CnnClassificationService(settings)
        self.llm = LlmAdviceService(settings)

    def analyze_upload(self, upload: FileStorage) -> dict:
        original_path = self._save_upload(upload)
        processed_dir = self.settings.upload_dir / "processed"

        started_at = time.perf_counter()
        yolo_started = time.perf_counter()
        detection = self.yolo.detect(original_path, processed_dir)
        yolo_ms = round((time.perf_counter() - yolo_started) * 1000, 2)

        cnn_started = time.perf_counter()
        classification = self.cnn.classify(detection["crop_path"])
        cnn_ms = round((time.perf_counter() - cnn_started) * 1000, 2)

        llm_started = time.perf_counter()
        llm_report = self.llm.generate(detection, classification, image_path=detection["crop_path"])
        llm_ms = round((time.perf_counter() - llm_started) * 1000, 2)

        total_ms = round((time.perf_counter() - started_at) * 1000, 2)

        return {
            "pipeline": [
                {
                    "step": "YOLO",
                    "status": "fallback" if detection.get("fallback") else "done",
                    "detail": detection["message"],
                    "duration_ms": yolo_ms,
                },
                {
                    "step": "CNN",
                    "status": "fallback" if classification.get("fallback") else "done",
                    "detail": (
                        classification.get("warning")
                        or f"Phân loại lớp {classification['display_label']} với độ tin cậy {classification['confidence'] * 100:.2f}%."
                    ),
                    "duration_ms": cnn_ms,
                },
                {
                    "step": "Gemini",
                    "status": "done",
                    "detail": f"Sinh giải thích bằng nguồn {llm_report['source']}.",
                    "duration_ms": llm_ms,
                },
            ],
            "images": {
                "original": self._relative_asset(original_path),
                "annotated": self._relative_asset(detection["annotated_path"]),
                "cropped_leaf": self._relative_asset(detection["crop_path"]),
            },
            "detection": {
                "found": detection["found"],
                "confidence": detection["confidence"],
                "label": detection["label"],
                "bbox": detection["bbox"],
            },
            "classification": classification,
            "llm": llm_report,
            "meta": {"total_duration_ms": total_ms},
        }

    def _save_upload(self, upload: FileStorage) -> Path:
        filename = secure_filename(upload.filename or "")
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if extension not in self.settings.allowed_extensions:
            supported = ", ".join(self.settings.allowed_extensions)
            raise BadRequestError(f"Định dạng ảnh chưa hỗ trợ. Hãy dùng: {supported}.")

        token = uuid4().hex
        target = self.settings.upload_dir / "originals" / f"{token}.{extension}"
        upload.save(target)
        return target

    def _relative_asset(self, path: Path) -> str:
        return path.relative_to(self.settings.upload_dir).as_posix()
