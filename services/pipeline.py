# ============================================================
# File: services/pipeline.py
# Vai trò: Điều phối toàn bộ luồng phân tích ảnh lá cây
# Kết nối 3 bước xử lý theo thứ tự:
#   Bước 1: YOLO phát hiện và crop vùng lá
#   Bước 2: CNN phân loại nhóm bệnh từ ảnh crop
#   Bước 3: Gemini tổng hợp kết quả và sinh nội dung tư vấn
# Điều phối luồng: upload ảnh → YOLO → CNN → Gemini → trả kết quả.
# Kết quả cuối được trả về cho Flask endpoint /api/analyze.
# ============================================================

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
    """
    Pipeline phân tích ảnh lá cây 3 bước: YOLO → CNN → Gemini.
    Được khởi tạo một lần khi app khởi động và tái sử dụng cho mọi request.
    """

    def __init__(self, settings: Settings):
        # Khởi tạo cấu hình và 3 service con
        self.settings = settings
        self.settings.ensure_runtime_directories()  # Tạo thư mục upload nếu chưa có

        # Bước 1: YOLO — phát hiện vùng lá trong ảnh
        self.yolo = YoloLeafService(settings)

        # Bước 2: CNN — phân loại bệnh từ ảnh vùng lá
        self.cnn = CnnClassificationService(settings)

        # Bước 3: Gemini — tổng hợp và sinh nội dung tư vấn cho người dùng
        self.llm = LlmAdviceService(settings)

    def analyze_upload(self, upload: FileStorage) -> dict:
        """
        Xử lý ảnh upload từ người dùng qua toàn bộ pipeline.
        Trả về dict chứa kết quả từ cả 3 bước + metadata thời gian xử lý.
        """

        # Lưu ảnh gốc vào disk, lấy đường dẫn
        original_path = self._save_upload(upload)
        processed_dir = self.settings.upload_dir / "processed"

        # --- Đo thời gian toàn bộ pipeline ---
        started_at = time.perf_counter()

        # === BƯỚC 1: YOLO phát hiện vùng lá ===
        yolo_started = time.perf_counter()
        detection = self.yolo.detect(original_path, processed_dir)
        yolo_ms = round((time.perf_counter() - yolo_started) * 1000, 2)

        # === BƯỚC 2: CNN phân loại bệnh từ ảnh crop ===
        cnn_started = time.perf_counter()
        classification = self.cnn.classify(detection["crop_path"])
        cnn_ms = round((time.perf_counter() - cnn_started) * 1000, 2)

        # === BƯỚC 3: Gemini sinh nội dung tư vấn ===
        # Truyền cả ảnh crop để Gemini có thể phân tích trực quan
        llm_started = time.perf_counter()
        llm_report = self.llm.generate(detection, classification, image_path=detection["crop_path"])
        llm_ms = round((time.perf_counter() - llm_started) * 1000, 2)

        total_ms = round((time.perf_counter() - started_at) * 1000, 2)

        # Đóng gói kết quả đầy đủ trả về cho client
        return {
            # Thông tin trạng thái và thời gian từng bước
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

            # Đường dẫn tương đối đến các file ảnh đã xử lý
            "images": {
                "original": self._relative_asset(original_path),
                "annotated": self._relative_asset(detection["annotated_path"]),
                "cropped_leaf": self._relative_asset(detection["crop_path"]),
            },

            # Kết quả từ YOLO: có tìm thấy lá không, confidence, bounding box
            "detection": {
                "found": detection["found"],
                "confidence": detection["confidence"],
                "label": detection["label"],
                "bbox": detection["bbox"],
            },

            # Kết quả từ CNN: nhóm bệnh, xác suất 5 nhóm, visual calibration
            "classification": classification,

            # Kết quả từ Gemini: chẩn đoán tổng hợp, hướng dẫn xử lý, cảnh báo
            "llm": llm_report,

            # Metadata: tổng thời gian xử lý
            "meta": {"total_duration_ms": total_ms},
        }

    def _save_upload(self, upload: FileStorage) -> Path:
        """
        Lưu file ảnh upload vào thư mục originals với tên UUID ngẫu nhiên.
        Kiểm tra định dạng file trước khi lưu.
        Ném BadRequestError nếu định dạng không hỗ trợ.
        """
        filename = secure_filename(upload.filename or "")
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        # Kiểm tra đuôi file có nằm trong danh sách cho phép không
        if extension not in self.settings.allowed_extensions:
            supported = ", ".join(self.settings.allowed_extensions)
            raise BadRequestError(f"Định dạng ảnh chưa hỗ trợ. Hãy dùng: {supported}.")

        # Tạo tên file ngẫu nhiên để tránh trùng lặp
        token = uuid4().hex
        target = self.settings.upload_dir / "originals" / f"{token}.{extension}"
        upload.save(target)
        return target

    def _relative_asset(self, path: Path) -> str:
        """
        Chuyển đường dẫn tuyệt đối của ảnh thành đường dẫn tương đối
        để client có thể dùng trong URL /uploads/<path>.
        """
        return path.relative_to(self.settings.upload_dir).as_posix()
