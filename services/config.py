
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    app_name: str
    base_dir: Path
    upload_dir: Path
    yolo_model_path: Path
    cnn_model_path: Path
    cnn_labels_path: Path
    gemini_api_key: str | None
    gemini_model: str
    gemini_model_fallbacks: list[str] | None
    yolo_conf_threshold: float
    crop_padding_ratio: float
    max_upload_size_mb: int
    allowed_extensions: tuple[str, ...]
    cnn_preprocess_mode: str

    def ensure_runtime_directories(self) -> None:
        (self.upload_dir / "originals").mkdir(parents=True, exist_ok=True)
        (self.upload_dir / "processed").mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
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
