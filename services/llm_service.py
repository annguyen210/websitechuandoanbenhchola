from __future__ import annotations

import json
import re

from google import genai

from services.config import Settings


class LlmAdviceService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    def generate(self, detection: dict, classification: dict) -> dict:
        if not self.settings.gemini_api_key:
            return self._fallback_report(
                classification,
                "Chưa cấu hình API key Gemini. Hệ thống dùng gợi ý mặc định từ CNN.",
            )

        prompt = self._build_prompt(detection, classification)
        models_to_try = [self.settings.gemini_model] + (self.settings.gemini_model_fallbacks or [])

        last_error: Exception | None = None
        for model_name in models_to_try:
            try:
                client = self._get_client()
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                data = self._parse_json(response.text)
                return {"source": "gemini", "model": model_name, **data}
            except Exception as exc:
                last_error = exc
                continue

        return self._fallback_report(
            classification,
            f"Gemini không phản hồi ({last_error}). Hệ thống dùng gợi ý mặc định từ CNN.",
        )

    def _build_prompt(self, detection: dict, classification: dict) -> str:
        top_predictions = "\n".join(
            f"- {item['display_label']}: {item['confidence'] * 100:.2f}%"
            for item in classification["top_predictions"]
        )
        cnn_label = classification["display_label"]
        cnn_conf = classification["confidence"] * 100
        low_conf_note = " (ĐỘ TIN CẬY THẤP - cần kiểm tra thực tế thêm)" if cnn_conf < 60 else ""

        return f"""
Bạn là chuyên gia thực vật học cấp cao. Nhiệm vụ của bạn là XÁC NHẬN và BỔ SUNG thông tin dựa trên kết quả đã được mô hình CNN phân loại.

=== KẾT QUẢ TỪ HỆ THỐNG CNN (NGUỒN CHÍNH XÁC - KHÔNG ĐƯỢC THAY ĐỔI) ===
- Chẩn đoán CNN: {cnn_label}{low_conf_note}
- Độ tin cậy CNN: {cnn_conf:.1f}%
- YOLO phát hiện lá: {"Có" if detection["found"] else "Không (dùng toàn ảnh)"}
- Độ tin cậy YOLO: {detection["confidence"] * 100:.1f}%
- Toàn bộ dự đoán CNN:
{top_predictions}

=== QUY TẮC BẮT BUỘC ===
1. KHÔNG được thay đổi hoặc phủ nhận kết quả CNN "{cnn_label}" - đây là chẩn đoán chính thức.
2. Tất cả thông tin bạn cung cấp PHẢI phù hợp với bệnh/tình trạng "{cnn_label}".
3. Nếu CNN có độ tin cậy < 60%, ghi rõ cảnh báo trong trường warning.
4. Bổ sung thông tin phụ trợ thực tế, sát với đặc điểm bệnh "{cnn_label}".

Trả về JSON hợp lệ với đúng các khóa (tiếng Việt, ngắn gọn, dễ hiểu):
- headline: tiêu đề ngắn gọn xác nhận chẩn đoán CNN "{cnn_label}"
- summary: 2-3 câu mô tả triệu chứng và đặc điểm thực tế của "{cnn_label}"
- care_steps: mảng 3-4 bước xử lý cụ thể, thực tế cho bệnh "{cnn_label}"
- next_steps: mảng 2-3 bước theo dõi sau khi xử lý
- warning: 1 câu về độ tin cậy CNN ({cnn_conf:.0f}%) và lưu ý cần thiết
- health_score: số nguyên 1-100 phản ánh mức độ nghiêm trọng của "{cnn_label}" (1=rất nặng, 100=khỏe)
- economic_impact: thiệt hại kinh tế điển hình của "{cnn_label}"
- spread_level: mức lây lan thực tế của "{cnn_label}" ("thấp"/"trung bình"/"cao")
- treatment_schedule: mảng bước điều trị theo thời gian [{{"action": "...", "days_later": 0}}]
- plant_type: loại cây thường gặp bệnh "{cnn_label}"
- disease_progression: object gồm 3_days, 7_days, 14_days mô tả diễn biến nếu không điều trị
""".strip()

    def _parse_json(self, content: str) -> dict:
        cleaned = re.sub(r"^```json|```$", "", content.strip(), flags=re.MULTILINE).strip()
        data = json.loads(cleaned)
        return {
            "headline": str(data.get("headline", "")).strip(),
            "summary": str(data.get("summary", "")).strip(),
            "care_steps": [str(item).strip() for item in data.get("care_steps", []) if str(item).strip()],
            "next_steps": [str(item).strip() for item in data.get("next_steps", []) if str(item).strip()],
            "warning": str(data.get("warning", "")).strip(),
            "health_score": int(data.get("health_score", 50)),
            "economic_impact": str(data.get("economic_impact", "")).strip(),
            "spread_level": str(data.get("spread_level", "unknown")).strip(),
            "treatment_schedule": data.get("treatment_schedule", []),
            "plant_type": str(data.get("plant_type", "unknown")).strip(),
            "disease_progression": data.get("disease_progression", {}),
        }

    def _fallback_report(self, classification: dict, reason: str) -> dict:
        label = classification["display_label"]
        confidence = classification["confidence"] * 100
        return {
            "source": "fallback",
            "model": "local-template",
            "headline": f"Kết quả gần nhất: {label}",
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
            "warning": reason,
            "health_score": 50,
            "economic_impact": "",
            "spread_level": "unknown",
            "treatment_schedule": [],
            "plant_type": "unknown",
            "disease_progression": {},
        }
