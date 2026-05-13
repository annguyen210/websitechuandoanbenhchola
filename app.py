import os

# Limit CPU threads to prevent deadlock / timeout on low-vCPU instances
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import base64
import io
from datetime import datetime
from pathlib import Path 
import importlib.util

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for, Response
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
                    "gemini_key_configured": bool(settings.gemini_api_key),
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


@app.post("/api/report")
def generate_report() -> tuple[Response, int]:
    data = request.get_json(force=True, silent=True) or {}
    result = data.get("result", data)

    llm = result.get("llm", {})
    classification = result.get("classification", {})
    images = result.get("images", {})

    cnn_label = classification.get("display_label", "Không xác định")
    cnn_conf = round(classification.get("confidence", 0) * 100, 1)
    health_score = int(llm.get("health_score", 50))
    headline = llm.get("headline", cnn_label)
    summary = llm.get("summary", "")
    plant_type = llm.get("plant_type", "-")
    spread_level = llm.get("spread_level", "-")
    economic_impact = llm.get("economic_impact", "-")
    care_steps = llm.get("care_steps", [])
    next_steps = llm.get("next_steps", [])
    warning = llm.get("warning", "")
    treatment_schedule = llm.get("treatment_schedule", [])
    disease_progression = llm.get("disease_progression", {})
    source = llm.get("source", "CNN")
    model = llm.get("model", "")
    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    # Embed original image as base64
    original_b64 = ""
    original_url = images.get("original", "")
    if original_url:
        rel = original_url.lstrip("/")
        parts = rel.split("/", 1)
        if len(parts) == 2:
            img_path = settings.upload_dir / parts[1]
            if img_path.exists():
                with open(img_path, "rb") as f:
                    raw = f.read()
                ext = img_path.suffix.lstrip(".").lower()
                mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
                original_b64 = f"data:{mime};base64,{base64.b64encode(raw).decode()}"

    def li_items(items):
        return "".join(f"<li>{item}</li>" for item in items) if items else "<li>Không có</li>"

    prog_rows = "".join(
        f'<tr><td style="padding:6px 12px;border:1px solid #d4eddb;font-weight:600;color:#1a7f46;">{k}</td>'
        f'<td style="padding:6px 12px;border:1px solid #d4eddb;">{v}</td></tr>'
        for k, v in disease_progression.items()
    )
    treat_rows = "".join(
        f"<li>Ngày {t.get('days_later', 0)}: {t.get('action', '')}</li>"
        for t in treatment_schedule
    ) if treatment_schedule else "<li>Không có</li>"

    score_color = "#22c55e" if health_score >= 60 else ("#f59e0b" if health_score >= 35 else "#ef4444")
    img_tag = f'<img src="{original_b64}" style="max-width:340px;max-height:260px;object-fit:contain;border-radius:12px;border:2px solid #d4eddb;" />' if original_b64 else '<div style="width:340px;height:200px;background:#eef7ec;border-radius:12px;display:flex;align-items:center;justify-content:center;color:#5d7567;">Không có ảnh</div>'

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>LeafAI – Báo cáo chẩn đoán</title>
<style>
  body{{margin:0;padding:32px;font-family:'Segoe UI',Arial,sans-serif;background:#f3fbf2;color:#183224;}}
  .wrap{{max-width:860px;margin:0 auto;background:#fff;border-radius:20px;box-shadow:0 8px 40px rgba(27,69,41,.13);overflow:hidden;}}
  .header{{background:linear-gradient(135deg,#156f3e,#0d5e32);color:#fff;padding:28px 36px;}}
  .header h1{{margin:0 0 6px;font-size:1.9rem;letter-spacing:-.03em;}}
  .header p{{margin:0;opacity:.8;font-size:.95rem;}}
  .body{{padding:28px 36px;}}
  .top-row{{display:flex;gap:28px;align-items:flex-start;margin-bottom:28px;flex-wrap:wrap;}}
  .diagnosis-box{{flex:1;min-width:220px;background:#f0fbf2;border:1.5px solid #b7e4c7;border-radius:14px;padding:20px;}}
  .diagnosis-box .label{{font-size:.8rem;font-weight:700;color:#1a7f46;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;}}
  .diagnosis-box .value{{font-size:1.6rem;font-weight:800;color:#183224;margin-bottom:4px;}}
  .diagnosis-box .sub{{font-size:.88rem;color:#5d7567;}}
  .score-wrap{{display:flex;align-items:center;gap:10px;margin-top:12px;}}
  .score-bar{{flex:1;height:14px;background:#d4eddb;border-radius:7px;overflow:hidden;}}
  .score-fill{{height:100%;border-radius:7px;background:{score_color};width:{health_score}%;}}
  .score-num{{font-weight:800;color:{score_color};font-size:1.05rem;}}
  .info-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:22px;}}
  .info-cell{{background:#f6fbf5;border-radius:10px;padding:14px 16px;}}
  .info-cell .k{{font-size:.8rem;font-weight:700;color:#1a7f46;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;}}
  .info-cell .v{{font-size:.95rem;color:#183224;}}
  h3{{color:#1a7f46;font-size:1rem;margin:20px 0 10px;border-left:3px solid #1a7f46;padding-left:10px;}}
  ul{{margin:0;padding-left:20px;color:#374151;line-height:1.75;}}
  table{{width:100%;border-collapse:collapse;font-size:.9rem;}}
  .warning-box{{background:#fef7de;border:1px solid #fde68a;border-radius:10px;padding:12px 16px;color:#78560a;font-size:.88rem;margin-top:18px;line-height:1.6;}}
  .footer{{background:#f0fbf2;padding:16px 36px;font-size:.82rem;color:#5d7567;display:flex;justify-content:space-between;}}
  .source-badge{{display:inline-block;padding:3px 10px;border-radius:20px;background:rgba(26,127,70,.12);color:#1a7f46;font-size:.78rem;font-weight:700;}}
  @media print{{body{{padding:0;background:#fff;}} .wrap{{box-shadow:none;border-radius:0;}}}}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <h1>LeafAI – Báo cáo chẩn đoán lá cây</h1>
    <p>Ngày xuất báo cáo: {now} &nbsp;|&nbsp; Nguồn phân tích: <span style="background:rgba(255,255,255,.2);padding:2px 8px;border-radius:10px;">{source} ({model})</span></p>
  </div>
  <div class="body">
    <div class="top-row">
      <div>{img_tag}</div>
      <div class="diagnosis-box" style="flex:1;">
        <div class="label">Kết quả CNN</div>
        <div class="value">{headline}</div>
        <div class="sub">Lớp phân loại: {cnn_label} &nbsp;|&nbsp; Độ tin cậy: {cnn_conf}%</div>
        <div class="sub" style="margin-top:4px;">Loại cây: {plant_type}</div>
        <div class="label" style="margin-top:14px;">Điểm sức khỏe cây</div>
        <div class="score-wrap">
          <div class="score-bar"><div class="score-fill"></div></div>
          <div class="score-num">{health_score}/100</div>
        </div>
      </div>
    </div>

    <h3>Mô tả tình trạng</h3>
    <p style="margin:0 0 16px;line-height:1.75;color:#374151;">{summary}</p>

    <div class="info-grid">
      <div class="info-cell"><div class="k">Mức độ lây lan</div><div class="v">{spread_level}</div></div>
      <div class="info-cell"><div class="k">Ảnh hưởng kinh tế</div><div class="v">{economic_impact}</div></div>
    </div>

    <h3>Các bước xử lý đề xuất</h3>
    <ul>{li_items(care_steps)}</ul>

    <h3>Theo dõi tiếp theo</h3>
    <ul>{li_items(next_steps)}</ul>

    <h3>Lịch điều trị</h3>
    <ul>{treat_rows}</ul>

    {"<h3>Dự báo diễn biến (nếu không điều trị)</h3><table><thead><tr><th style='padding:6px 12px;border:1px solid #d4eddb;background:#e8f5e9;text-align:left;'>Thời điểm</th><th style='padding:6px 12px;border:1px solid #d4eddb;background:#e8f5e9;text-align:left;'>Tình trạng dự báo</th></tr></thead><tbody>" + prog_rows + "</tbody></table>" if disease_progression else ""}

    <div class="warning-box">{warning}</div>
  </div>
  <div class="footer">
    <span>LeafAI – Hệ thống chẩn đoán bệnh lá cây (YOLO + CNN + Gemini)</span>
    <span class="source-badge">AI – Chỉ mang tính tham khảo</span>
  </div>
</div>
</body>
</html>"""

    filename = f"leafai-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.html"
    return (
        Response(
            html,
            mimetype="text/html; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        ),
        200,
    )


@app.post("/api/qr")
def generate_qr() -> tuple[dict, int]:
    data = request.get_json(force=True, silent=True) or {}

    cnn_label = data.get("cnn_label", "Không xác định")
    cnn_conf = data.get("cnn_conf", 0)
    health_score = data.get("health_score", 50)
    plant_type = data.get("plant_type", "-")
    summary = data.get("summary", "")
    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    qr_text = (
        f"LeafAI - Chẩn đoán lá cây\n"
        f"Ngày: {now}\n"
        f"Bệnh/Tình trạng: {cnn_label}\n"
        f"Độ tin cậy CNN: {cnn_conf:.0f}%\n"
        f"Điểm sức khỏe: {health_score}/100\n"
        f"Loại cây: {plant_type}\n"
        f"Tóm tắt: {summary[:120] if summary else 'Không có'}"
    )

    try:
        import qrcode as qrlib
        qr = qrlib.QRCode(
            version=None,
            error_correction=qrlib.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_text)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#1a7f46", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        return jsonify({
            "success": True,
            "qr_data_url": f"data:image/png;base64,{img_b64}",
            "text": qr_text,
        }), 200
    except ImportError:
        return jsonify({"success": False, "error": "Thư viện qrcode chưa được cài. Chạy: pip install qrcode"}), 500
    except Exception as exc:
        return jsonify({"success": False, "error": f"Không thể tạo QR Code: {exc}"}), 500


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
