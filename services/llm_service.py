from __future__ import annotations

import json
import mimetypes
import re
from pathlib import Path

from google import genai
from google.genai import types

from services.config import Settings


class LlmAdviceService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    def generate(self, detection: dict, classification: dict, image_path: Path | None = None) -> dict:
        if not self.settings.gemini_api_key:
            return self._fallback_report(
                classification,
                "Chưa cấu hình API key Gemini. Hệ thống dùng gợi ý mặc định từ CNN.",
            )

        prompt = self._build_prompt(detection, classification)
        contents = self._build_contents(prompt, image_path)
        models_to_try = [self.settings.gemini_model] + (self.settings.gemini_model_fallbacks or [])

        last_error: Exception | None = None
        for model_name in models_to_try:
            try:
                client = self._get_client()
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                )
                data = self._parse_json(response.text)
                return {
                    "source": "gemini",
                    "model": model_name,
                    "image_analyzed": image_path is not None and (image_path.exists() if image_path else False),
                    **data,
                }
            except Exception as exc:
                last_error = exc
                if image_path is not None:
                    try:
                        client = self._get_client()
                        response = client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                        )
                        data = self._parse_json(response.text)
                        return {
                            "source": "gemini",
                            "model": model_name,
                            "image_analyzed": False,
                            **data,
                        }
                    except Exception as exc2:
                        last_error = exc2
                continue

        return self._fallback_report(
            classification,
            f"Gemini không phản hồi ({last_error}). Hệ thống dùng gợi ý mặc định từ CNN.",
        )

    def _build_contents(self, prompt: str, image_path: Path | None) -> list | str:
        if image_path is None or not image_path.exists():
            return prompt

        try:
            mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            return [
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                types.Part.from_text(text=prompt),
            ]
        except Exception:
            return prompt

    def _build_prompt(self, detection: dict, classification: dict) -> str:
        top_predictions = "\n".join(
            f"  {i+1}. {item['display_label']}: {item['confidence'] * 100:.2f}%"
            for i, item in enumerate(classification["top_predictions"])
        )
        cnn_label = classification["display_label"]
        cnn_conf = classification["confidence"] * 100
        tta_runs = classification.get("tta_runs", 1)
        preprocess_mode = classification.get("preprocess_mode", "efficientnet")
        entropy_ratio = classification.get("entropy_ratio", 0.0)

        # Chú thích tin cậy cho Gemini
        if entropy_ratio > 0.78:
            conf_note = (
                f"⚠️ CẢNH BÁO: Entropy cao ({entropy_ratio * 100:.0f}% mức tối đa) — "
                "CNN không nhận dạng được lá sắn rõ ràng, ảnh có thể không phải lá sắn."
            )
        elif cnn_conf < 40:
            conf_note = f"⚠️ Độ tin cậy RẤT THẤP ({cnn_conf:.1f}%)"
        elif cnn_conf < 60:
            conf_note = f"⚠️ Độ tin cậy thấp ({cnn_conf:.1f}%)"
        else:
            conf_note = f"Độ tin cậy tốt ({cnn_conf:.1f}%)"

        low_conf_note = " ⚠️ ĐỘ TIN CẬY THẤP" if cnn_conf < 60 else ""
        yolo_found = "Có" if detection["found"] else "Không – phân tích toàn bộ ảnh gốc"

        return f"""
Bạn là chuyên gia bệnh học thực vật cấp cao, chuyên sâu về cây sắn (cassava/mì). Nhiệm vụ: phân tích hình ảnh lá thực tế để chẩn đoán bệnh và đưa ra tư vấn xử lý thiết thực.

═══════════════════════════════════════════════
BƯỚC 1 — PHÂN TÍCH HÌNH ẢNH THỰC TẾ (LÀM TRƯỚC TIÊN)
═══════════════════════════════════════════════
Quan sát KỸ hình ảnh lá cây được gửi kèm. Ghi nhận những gì bạn THỰC SỰ THẤY:

• Màu sắc lá: xanh đồng đều | vàng nhạt | đốm vàng | khảm vàng-xanh | nâu/đen?
• Bề mặt: phẳng mịn | nhăn | cong vênh | phồng rộp?
• Vết thương / tổn thương: có/không — mô tả vị trí, màu sắc, hình dạng, mức độ lan?
• Gân lá: bình thường | đổi màu | vết nâu/vàng dọc gân?
• Mép lá: nguyên vẹn | hoại tử khô | mềm thối?
• Nhận định sơ bộ từ ảnh: Lá KHỎE hay có TRIỆU CHỨNG BỆNH RÕ RÀNG?

BẢNG NHẬN DIỆN 5 LỚP BỆNH — ĐỌC KỸ ĐỂ ĐỐI CHIẾU:
1. Cassava Bacterial Blight (CBB)      → đốm góc lá hình đa giác viền vàng→nâu, tiết dịch đen, héo từ ngọn xuống
2. Cassava Brown Streak Disease (CBSD) → vệt vàng dọc gân lá, đốm hoại tử nâu-vàng xen kẽ, thối vỏ củ
3. Cassava Green Mottle (CGM)          → lốm đốm xanh nhạt không đều, lá hơi biến dạng nhẹ
4. Cassava Mosaic Disease (CMD)        → khảm vàng-xanh rõ rệt, lá nhăn cuộn mạnh, biến dạng đáng kể
5. Healthy                             → xanh đồng đều, không đốm, không biến dạng, không tổn thương

═══════════════════════════════════════════════
BƯỚC 2 — KẾT QUẢ CNN (THÔNG TIN THAM KHẢO)
═══════════════════════════════════════════════
Mô hình CNN (EfficientNetB3, {tta_runs} lần TTA, preprocessing={preprocess_mode}):
• Chẩn đoán CNN: {cnn_label}{low_conf_note}
• YOLO phát hiện lá: {yolo_found}
• Đánh giá độ tin cậy: {conf_note}
• Xác suất 5 lớp (cao → thấp):
{top_predictions}

═══════════════════════════════════════════════
BƯỚC 3 — ĐỐI CHIẾU & KẾT LUẬN TỔNG HỢP
═══════════════════════════════════════════════
QUY TẮC — CNN là nguồn chính, quan sát ảnh là nguồn bổ sung/kiểm chứng:

• CNN conf ≥ 55% VÀ entropy bình thường (<78%): DÙNG KẾT QUẢ CNN → final_diagnosis = kết quả CNN, cnn_agreement = "agree".

• CNN conf 40–54% hoặc entropy cao (>60% nhưng <78%): THAM KHẢO CNN → nếu ảnh không có bằng chứng HOÀN TOÀN trái ngược, giữ final_diagnosis = CNN, cnn_agreement = "uncertain".

• CNN conf < 40% HOẶC entropy > 78%: PHÂN TÍCH ẢNH THỰC TẾ LÀ CHÍNH → nếu ảnh có triệu chứng bệnh rõ, đặt final_diagnosis theo quan sát ảnh, cnn_agreement = "disagree". Nếu không rõ hoặc không phải lá sắn, ghi nhận trong warning.

═══════════════════════════════════════════════
YÊU CẦU ĐẦU RA — JSON TIẾNG VIỆT HỢP LỆ
═══════════════════════════════════════════════
{{
  "headline": "Tiêu đề ≤12 từ phản ánh chẩn đoán tổng hợp từ ảnh + CNN",
  "final_diagnosis": "một trong: cassava_bacterial_blight | cassava_brown_streak_disease | cassava_green_mottle | cassava_mosaic_disease | healthy",
  "cnn_agreement": "agree | disagree | uncertain",
  "visual_confidence": "high | medium | low",
  "summary": "3-4 câu mô tả CỤ THỂ triệu chứng bạn THỰC SỰ THẤY trong ảnh + mức độ khớp với chẩn đoán.",
  "visual_observations": [
    "Quan sát 1: chi tiết cụ thể từ ảnh (màu sắc, đốm, kết cấu...)",
    "Quan sát 2: chi tiết cụ thể khác",
    "Quan sát 3: chi tiết cụ thể khác"
  ],
  "disease_evidence": "Giải thích TẠI SAO hình ảnh khớp hoặc KHÔNG khớp với {cnn_label}. Nêu bằng chứng hình ảnh cụ thể.",
  "care_steps": [
    "Bước 1 xử lý cụ thể theo bệnh đã chẩn đoán",
    "Bước 2",
    "Bước 3",
    "Bước 4"
  ],
  "next_steps": ["Theo dõi tiếp 1", "Theo dõi tiếp 2"],
  "recommendations": ["Khuyến nghị thêm 1 liên quan đến phòng bệnh/canh tác", "Khuyến nghị thêm 2"],
  "warning": "Nhận xét về mức độ khớp CNN ({cnn_conf:.0f}%) vs quan sát ảnh. Ghi rõ nếu entropy cao hoặc ảnh không phải lá sắn.",
  "health_score": 50,
  "economic_impact": "Ước tính thiệt hại kinh tế nếu không xử lý kịp thời",
  "spread_level": "thấp | trung bình | cao",
  "treatment_schedule": [{{"action": "Hành động cụ thể", "days_later": 0}}],
  "plant_type": "Loại cây xác định từ ảnh",
  "disease_progression": {{"3_days": "...", "7_days": "...", "14_days": "..."}}
}}
""".strip()

    def _parse_json(self, content: str) -> dict:
        cleaned = re.sub(r"^```json|^```|```$", "", content.strip(), flags=re.MULTILINE).strip()
        data = json.loads(cleaned)
        return {
            "headline": str(data.get("headline", "")).strip(),
            "final_diagnosis": str(data.get("final_diagnosis", "")).strip(),
            "cnn_agreement": str(data.get("cnn_agreement", "uncertain")).strip(),
            "visual_confidence": str(data.get("visual_confidence", "medium")).strip(),
            "summary": str(data.get("summary", "")).strip(),
            "care_steps": [str(i).strip() for i in data.get("care_steps", []) if str(i).strip()],
            "next_steps": [str(i).strip() for i in data.get("next_steps", []) if str(i).strip()],
            "warning": str(data.get("warning", "")).strip(),
            "health_score": int(data.get("health_score", 50)),
            "economic_impact": str(data.get("economic_impact", "")).strip(),
            "spread_level": str(data.get("spread_level", "unknown")).strip(),
            "treatment_schedule": data.get("treatment_schedule", []),
            "plant_type": str(data.get("plant_type", "unknown")).strip(),
            "disease_progression": data.get("disease_progression", {}),
            "visual_observations": [
                str(i).strip() for i in data.get("visual_observations", []) if str(i).strip()
            ],
            "disease_evidence": str(data.get("disease_evidence", "")).strip(),
            "recommendations": [str(i).strip() for i in data.get("recommendations", []) if str(i).strip()],
        }

    def _fallback_report(self, classification: dict, reason: str) -> dict:
        label = classification["display_label"]
        confidence = classification["confidence"] * 100
        return {
            "source": "fallback",
            "model": "local-template",
            "image_analyzed": False,
            "headline": f"Kết quả CNN: {label}",
            "final_diagnosis": classification.get("label", ""),
            "cnn_agreement": "uncertain",
            "visual_confidence": "low",
            "summary": (
                f"CNN đang nghiêng về lớp '{label}' với độ tin cậy khoảng {confidence:.1f}%. "
                "Bạn nên xem đây là gợi ý ban đầu để kiểm tra lá và điều kiện chăm sóc thực tế."
            ),
            "care_steps": [
                "Tách riêng cây có dấu hiệu bất thường để hạn chế lây lan.",
                "Kiểm tra lại mặt trên, mặt dưới lá và chụp thêm ảnh sáng rõ nếu cần.",
                "Điều chỉnh tưới nước, ánh sáng và độ thông thoáng quanh cây.",
                "Loại bỏ phần lá hư nặng nếu cây đã bị tổn thương rõ rệt.",
            ],
            "next_steps": [
                "Theo dõi sự thay đổi của đốm lá trong 3-5 ngày tiếp theo.",
                "So sánh thêm với ảnh chuẩn hoặc hỏi cán bộ nông nghiệp khi cần.",
            ],
            "recommendations": [],
            "warning": reason,
            "health_score": 50,
            "economic_impact": "",
            "spread_level": "unknown",
            "treatment_schedule": [],
            "plant_type": "unknown",
            "disease_progression": {},
            "visual_observations": [],
            "disease_evidence": "",
        }
