import os
from pathlib import Path
import importlib.util

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from services.config import get_settings
from services.exceptions import AppError
from services.pipeline import AnalysisPipeline


settings = get_settings()
app = Flask(
    __name__,
    template_folder=str(settings.base_dir / "templates"),
    static_folder=str(settings.base_dir / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_size_mb * 1024 * 1024
pipeline = AnalysisPipeline(settings)


def asset_url(relative_path: str | None) -> str | None:
    if not relative_path:
        return None
    return url_for("uploaded_file", filename=relative_path)


def attach_urls(payload: dict) -> dict:
    images = payload.get("images", {})
    payload["images"] = {
        "original": asset_url(images.get("original")),
        "annotated": asset_url(images.get("annotated")),
        "cropped_leaf": asset_url(images.get("cropped_leaf")),
    }
    return payload


@app.get("/")
def index() -> str:
    return render_template("index.html", app_name=settings.app_name)


@app.get("/api/health")
def health() -> tuple[dict, int]:
    ultralytics_ready = importlib.util.find_spec("ultralytics") is not None
    tensorflow_ready = importlib.util.find_spec("tensorflow") is not None
    return (
        jsonify(
            {
                "status": "ok",
                "app_name": settings.app_name,
                "recommended_python": "Python 3.11 hoặc 3.12",
                "dependencies": {
                    "yolo_model_found": Path(settings.yolo_model_path).exists(),
                    "cnn_model_found": Path(settings.cnn_model_path).exists(),
                    "cnn_labels_found": Path(settings.cnn_labels_path).exists(),
                    "openai_key_configured": bool(settings.openai_api_key),
                    "ultralytics_ready": ultralytics_ready,
                    "tensorflow_ready": tensorflow_ready,
                },
            }
        ),
        200,
    )


@app.post("/api/analyze")
def analyze() -> tuple[dict, int]:
    image = request.files.get("image")
    if image is None or not image.filename:
        return jsonify({"success": False, "error": "Vui lòng tải lên một ảnh lá cây."}), 400

    try:
        result = pipeline.analyze_upload(image)
    except AppError as exc:
        return jsonify({"success": False, "error": str(exc)}), exc.status_code
    except Exception:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Máy chủ gặp lỗi ngoài dự kiến trong quá trình phân tích ảnh.",
                }
            ),
            500,
        )

    return jsonify({"success": True, "result": attach_urls(result)}), 200


@app.get("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(settings.upload_dir, filename)


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(_: RequestEntityTooLarge) -> tuple[dict, int]:
    return (
        jsonify(
            {
                "success": False,
                "error": f"Kích thước ảnh vượt quá {settings.max_upload_size_mb}MB.",
            }
        ),
        413,
    )


@app.errorhandler(HTTPException)
def handle_http_exception(exc: HTTPException) -> tuple[dict, int]:
    if request.path.startswith("/api/"):
        return (
            jsonify(
                {
                    "success": False,
                    "error": exc.description or "Yêu cầu không hợp lệ.",
                }
            ),
            exc.code or 500,
        )
    return exc


@app.errorhandler(Exception)
def handle_internal_error(exc: Exception) -> tuple[dict, int]:
    if request.path.startswith("/api/"):
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Máy chủ gặp lỗi ngoài dự kiến. Vui lòng thử lại sau.",
                }
            ),
            500,
        )
    raise exc


if __name__ == "__main__":
    settings.ensure_runtime_directories()
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
