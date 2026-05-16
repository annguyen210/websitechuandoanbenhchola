# ============================================================
# File: services/config.py
# Vai trò: Quản lý cấu hình toàn bộ ứng dụng
# Đọc các biến môi trường từ file .env và tạo object Settings
# dùng chung cho tất cả các service trong pipeline phân tích.       
# ============================================================

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# Xác định thư mục gốc của dự án và nạp biến môi trường từ .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """
    Tập hợp toàn bộ cấu hình của ứng dụng dưới dạng dataclass bất biến.
    Được khởi tạo một lần duy nhất qua hàm get_settings() và dùng lại ở mọi nơi.
    """

    # Tên hiển thị của ứng dụng
    app_name: str

    # Thư mục gốc và thư mục lưu ảnh upload
    base_dir: Path
    upload_dir: Path

    # Đường dẫn đến file model YOLO (.pt) để phát hiện vùng lá
    yolo_model_path: Path

    # Đường dẫn đến file model CNN (.h5) để phân loại bệnh
    cnn_model_path: Path

    # Đường dẫn đến file nhãn JSON cho CNN (5 nhóm bệnh)
    cnn_labels_path: Path

    # API key để gọi Gemini (Google AI) — có thể None nếu chưa cấu hình
    gemini_api_key: str | None

    # Tên model Gemini chính và danh sách fallback khi model chính lỗi
    gemini_model: str
    gemini_model_fallbacks: list[str] | None

    # Ngưỡng confidence tối thiểu để YOLO chấp nhận một detection (0.0–1.0)
    yolo_conf_threshold: float

    # Tỉ lệ mở rộng bounding box khi crop lá (ví dụ 0.08 = mở rộng 8% mỗi chiều)
    crop_padding_ratio: float

    # Giới hạn kích thước file upload tính bằng MB
    max_upload_size_mb: int

    # Các định dạng ảnh được phép upload (jpg, jpeg, png, webp)
    allowed_extensions: tuple[str, ...]

    # Chế độ preprocessing cho CNN — mặc định "efficientnet"
    cnn_preprocess_mode: str

    def ensure_runtime_directories(self) -> None:
        """Tạo các thư mục cần thiết khi chạy app nếu chưa tồn tại."""
        (self.upload_dir / "originals").mkdir(parents=True, exist_ok=True)
        (self.upload_dir / "processed").mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """
    Đọc toàn bộ cấu hình từ biến môi trường và trả về object Settings.
    Được gọi một lần duy nhất khi khởi động app.py.
    """

    # Parse danh sách đuôi file cho phép, tách bằng dấu phẩy
    allowed_extensions = tuple(
        ext.strip().lower()
        for ext in os.getenv("ALLOWED_EXTENSIONS", "jpg,jpeg,png,webp").split(",")
        if ext.strip()
    )

    return Settings(
        app_name=os.getenv("APP_NAME", "LeafCare AI"),
        base_dir=BASE_DIR,
        upload_dir=BASE_DIR / "uploads",
        yolo_model_path=BASE_DIR / os.getenv("YOLO_MODEL_PATH", "moduleyolola/best.pt"),
        cnn_model_path=BASE_DIR / os.getenv("CNN_MODEL_PATH", "model_0.h5"),
        cnn_labels_path=BASE_DIR / os.getenv("CNN_LABELS_PATH", "config/cnn_labels.json"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_model_fallbacks=["gemini-flash-latest"],
        yolo_conf_threshold=float(os.getenv("YOLO_CONF_THRESHOLD", "0.25")),
        crop_padding_ratio=float(os.getenv("CROP_PADDING_RATIO", "0.08")),
        max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", "10")),
        allowed_extensions=allowed_extensions,
        cnn_preprocess_mode=os.getenv("CNN_PREPROCESS_MODE", "efficientnet").strip().lower(),
    )
