# ============================================================
# File: services/exceptions.py
# Vai trò: Định nghĩa các loại lỗi tùy chỉnh của ứng dụng
# Mỗi exception class gắn với một HTTP status code cụ thể
# để Flask có thể trả về response lỗi đúng chuẩn cho client.
# ============================================================


class AppError(Exception):
    """
    Lớp lỗi gốc của ứng dụng.
    Tất cả lỗi tùy chỉnh đều kế thừa từ đây.
    Mặc định trả về HTTP 500 (Internal Server Error).
    """
    status_code = 500


class BadRequestError(AppError):
    """
    Lỗi do phía client gửi dữ liệu không hợp lệ.
    Ví dụ: không gửi ảnh, định dạng file sai, ảnh vượt kích thước.
    Trả về HTTP 400 (Bad Request).
    """
    status_code = 400


class ConfigurationError(AppError):
    """
    Lỗi cấu hình hệ thống — thiếu API key, đường dẫn model không hợp lệ.
    Trả về HTTP 500.
    """
    status_code = 500


class DependencyError(AppError):
    """
    Lỗi phụ thuộc bên ngoài — thiếu thư viện, model chưa tải được.
    Ví dụ: ultralytics chưa cài, TensorFlow lỗi.
    Trả về HTTP 500.
    """
    status_code = 500


class InferenceError(AppError):
    """
    Lỗi xảy ra trong quá trình suy luận (inference) của model AI.
    Ví dụ: CNN không thể chạy predict trên ảnh đầu vào.
    Trả về HTTP 500.
    """
    status_code = 500
