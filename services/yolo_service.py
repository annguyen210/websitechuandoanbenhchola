# ============================================================
# File: services/yolo_service.py
# Vai trò: BƯỚC 1 trong pipeline — Phát hiện và cắt vùng lá
# Dùng model YOLO (best.pt) để tìm bounding box của lá cây
# trong ảnh upload, sau đó crop và lưu vùng lá đó để CNN xử lý.
# Nếu YOLO không tìm thấy lá, dùng toàn bộ ảnh gốc (fallback).
# ============================================================

from __future__ import annotations

from pathlib import Path
from threading import Lock
from uuid import uuid4

from PIL import Image, ImageDraw, ImageOps

from services.config import Settings
from services.exceptions import ConfigurationError, DependencyError, InferenceError


class YoloLeafService:
    """
    Service phát hiện vùng lá bằng YOLO.
    Model được load một lần duy nhất và dùng chung (singleton pattern)
    để tránh tốn bộ nhớ khi có nhiều request đồng thời.
    """

    # Biến class — lưu model sau khi load lần đầu, dùng chung cho mọi instance
    _model = None
    _model_lock = Lock()  # Khóa thread để tránh load model đồng thời nhiều lần

    def __init__(self, settings: Settings):
        # Nhận cấu hình từ bên ngoài (dependency injection)
        self.settings = settings

    def _load_model(self):
        """
        Nạp model YOLO từ file .pt nếu chưa có trong bộ nhớ.
        Sử dụng double-checked locking để thread-safe.
        Trả về None nếu thiếu file model hoặc thiếu thư viện ultralytics.
        """
        # Kiểm tra file model tồn tại không
        if not self.settings.yolo_model_path.exists():
            return None

        if self.__class__._model is None:
            try:
                from ultralytics import YOLO
            except ModuleNotFoundError:
                # ultralytics chưa được cài — chạy fallback mode
                return None

            # Double-checked locking: tránh load model 2 lần khi có nhiều thread
            with self.__class__._model_lock:
                if self.__class__._model is None:
                    self.__class__._model = YOLO(str(self.settings.yolo_model_path))

        return self.__class__._model

    def detect(self, image_path: Path, output_dir: Path) -> dict:
        """
        Phát hiện vùng lá trong ảnh và cắt ra.
        - Nếu YOLO tìm thấy lá → crop vùng lá, vẽ bounding box lên ảnh annotated
        - Nếu không tìm thấy → dùng toàn bộ ảnh gốc (fallback)
        Trả về dict chứa đường dẫn ảnh crop, annotated và metadata.
        """
        model = self._load_model()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Đọc ảnh và tự động xoay theo EXIF metadata (ảnh chụp từ điện thoại)
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")

        # Nếu không có model → fallback dùng toàn ảnh
        if model is None:
            return self._build_fallback_assets(
                image,
                output_dir,
                "YOLO đang ở chế độ dự phòng vì thiếu ultralytics hoặc chưa tìm thấy trọng số.",
            )

        # Chạy YOLO inference trên ảnh
        try:
            result = model.predict(
                source=str(image_path),
                conf=self.settings.yolo_conf_threshold,  # Ngưỡng confidence tối thiểu
                verbose=False,
            )[0]
        except Exception as exc:
            # YOLO lỗi runtime → fallback
            return self._build_fallback_assets(
                image,
                output_dir,
                f"YOLO lỗi khi suy luận, hệ thống dùng toàn ảnh cho bước CNN. Chi tiết: {exc}",
            )

        # Kiểm tra có phát hiện được object nào không
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return self._build_fallback_assets(image, output_dir)

        # Lấy thông tin bounding boxes: confidence, class id, tọa độ
        confidences = boxes.conf.tolist()
        classes = boxes.cls.tolist() if boxes.cls is not None else [0] * len(confidences)
        coordinates = boxes.xyxy.tolist()

        # Chọn box có confidence cao nhất làm vùng lá chính
        best_index = max(range(len(confidences)), key=confidences.__getitem__)
        x1, y1, x2, y2 = coordinates[best_index]

        # Mở rộng box thêm padding để tránh cắt sát lá
        x1, y1, x2, y2 = self._expand_box(x1, y1, x2, y2, image.width, image.height)

        # Crop vùng lá từ ảnh gốc
        crop = image.crop((x1, y1, x2, y2))

        # Vẽ bounding box lên bản copy để hiển thị cho người dùng
        annotated = image.copy()
        self._draw_bbox(
            annotated,
            (x1, y1, x2, y2),
            confidences[best_index],
            self._resolve_label(result.names, int(classes[best_index])),
        )

        # Lưu ảnh crop và annotated ra disk với tên UUID ngẫu nhiên
        token = uuid4().hex
        crop_path = output_dir / f"{token}_crop.jpg"
        annotated_path = output_dir / f"{token}_annotated.jpg"
        crop.save(crop_path, quality=95)
        annotated.save(annotated_path, quality=95)

        return {
            "found": True,
            "confidence": round(float(confidences[best_index]), 4),
            "label": self._resolve_label(result.names, int(classes[best_index])),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "crop_path": crop_path,
            "annotated_path": annotated_path,
            "message": "YOLO đã tìm thấy vùng lá rõ nhất để chuyển sang bước CNN.",
            "fallback": False,
        }

    def _build_fallback_assets(self, image: Image.Image, output_dir: Path, message: str = "") -> dict:
        """
        Tạo fallback khi YOLO không phát hiện được lá hoặc gặp lỗi.
        Lưu toàn bộ ảnh gốc làm cả crop lẫn annotated để CNN vẫn có đầu vào.
        """
        token = uuid4().hex
        crop_path = output_dir / f"{token}_crop.jpg"
        annotated_path = output_dir / f"{token}_annotated.jpg"
        image.save(crop_path, quality=95)
        image.save(annotated_path, quality=95)

        return {
            "found": False,
            "confidence": 0.0,
            "label": "unknown",
            "bbox": None,
            "crop_path": crop_path,
            "annotated_path": annotated_path,
            "message": message or "YOLO không phát hiện được vùng lá — dùng toàn bộ ảnh gốc.",
            "fallback": True,
        }

    def _expand_box(
        self,
        x1: float, y1: float, x2: float, y2: float,
        image_width: int, image_height: int,
    ) -> tuple[int, int, int, int]:
        """
        Mở rộng bounding box thêm padding_ratio về mỗi phía.
        Đảm bảo tọa độ không vượt ra ngoài biên ảnh.
        """
        width = x2 - x1
        height = y2 - y1
        pad_x = width * self.settings.crop_padding_ratio
        pad_y = height * self.settings.crop_padding_ratio

        return (
            max(0, int(x1 - pad_x)),
            max(0, int(y1 - pad_y)),
            min(image_width, int(x2 + pad_x)),
            min(image_height, int(y2 + pad_y)),
        )

    def _draw_bbox(
        self,
        image: Image.Image,
        box: tuple[int, int, int, int],
        confidence: float,
        label: str,
    ) -> None:
        """
        Vẽ bounding box và nhãn lên ảnh để hiển thị cho người dùng.
        Sử dụng màu xanh lá (#0b8f46) đặc trưng của ứng dụng.
        """
        draw = ImageDraw.Draw(image)
        x1, y1, x2, y2 = box
        title = f"{label} | {confidence * 100:.1f}%"

        # Vẽ khung hình chữ nhật bo góc
        draw.rounded_rectangle((x1, y1, x2, y2), outline="#0b8f46", width=5, radius=12)

        # Vẽ nền label phía trên box
        text_bbox = draw.textbbox((x1, y1), title)
        padding = 8
        background = (
            text_bbox[0] - padding,
            max(0, text_bbox[1] - padding),
            text_bbox[2] + padding,
            text_bbox[3] + padding,
        )
        draw.rounded_rectangle(background, fill="#0b8f46", radius=10)
        draw.text((x1, max(4, y1 - 2)), title, fill="white")

    def _resolve_label(self, names, class_id: int) -> str:
        """
        Lấy tên nhãn của class từ dict hoặc list names của YOLO.
        Trả về "leaf" nếu không tìm thấy.
        """
        if isinstance(names, dict):
            return str(names.get(class_id, "leaf"))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return "leaf"
