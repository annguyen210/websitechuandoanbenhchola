# ============================================================
# File: services/llm_service.py
# Vai trò: BƯỚC 3 trong pipeline — Sinh nội dung tư vấn bằng Gemini AI
#
# Luồng xử lý chính:
#   1. Xây dựng prompt từ kết quả CNN + detection (YOLO)
#   2. Gửi ảnh lá + prompt đến Gemini API (Google AI)
#   3. Gemini phân tích ảnh trực quan và kết hợp với dữ liệu CNN
#   4. Parse JSON từ response của Gemini
#   5. Sanitize: loại bỏ các từ không mong muốn (sắn, cassava...)
#   6. Trả về dict chứa đầy đủ thông tin tư vấn cho người dùng
#
# Nguyên tắc ưu tiên:
#   - Gemini phân tích ảnh THỰC TẾ là nguồn chính
#   - CNN là tham khảo phụ, có thể bị override nếu ảnh rõ triệu chứng
#   - Kết quả cuối luôn được lọc bỏ "sắn/cassava" trước khi trả về
# ============================================================

from __future__ import annotations

import json
import mimetypes
import re
import unicodedata
from pathlib import Path

from google import genai
from google.genai import types

from services.config import Settings


class LlmAdviceService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        """Khởi tạo Gemini client lazy — chỉ tạo một lần, tái sử dụng cho mọi request."""
        if self._client is None:
            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    def generate(self, detection: dict, classification: dict, image_path: Path | None = None) -> dict:
        """
        Hàm chính: sinh báo cáo tư vấn bằng Gemini AI.

        Luồng xử lý:
          1. Kiểm tra API key — nếu thiếu → trả về fallback ngay
          2. Build prompt từ kết quả CNN + detection
          3. Build contents: gửi kèm ảnh lá (base64) nếu có
          4. Thử từng model trong danh sách (chính → fallback)
          5. Với mỗi model: thử gửi ảnh + prompt, nếu lỗi thì thử text-only
          6. Parse JSON từ response → sanitize → trả về
          7. Nếu tất cả model lỗi → fallback_report từ CNN
        """
        if not self.settings.gemini_api_key:
            return self._fallback_report(
                classification,
                "Chưa cấu hình API key Gemini. Hệ thống dùng gợi ý mặc định từ CNN.",
            )

        # Xây dựng prompt chứa kết quả CNN và hướng dẫn phân tích
        prompt = self._build_prompt(detection, classification)

        # Đóng gói ảnh + prompt hoặc chỉ prompt nếu không có ảnh
        contents = self._build_contents(prompt, image_path)

        # Danh sách model để thử theo thứ tự ưu tiên
        models_to_try = [self.settings.gemini_model] + (self.settings.gemini_model_fallbacks or [])

        # Cấu hình generation: tăng max_output_tokens để Gemini trả về nội dung dài đủ chi tiết
        gen_config = types.GenerateContentConfig(
            max_output_tokens=8192,   # Cho phép response dài tối đa ~8000 token
            temperature=0.4,          # Thấp = nhất quán, ít "sáng tạo" lạc đề
        )

        last_error: Exception | None = None
        for model_name in models_to_try:
            try:
                client = self._get_client()
                # Lần 1: gửi ảnh + prompt (multimodal) với config đầy đủ
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=gen_config,
                )
                # Parse JSON → sanitize từ "sắn/cassava" → trả về
                data = self._sanitize_report(self._parse_json(response.text))
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
                        # Lần 2: thử lại text-only nếu multimodal lỗi
                        response = client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                            config=gen_config,
                        )
                        data = self._sanitize_report(self._parse_json(response.text))
                        return {
                            "source": "gemini",
                            "model": model_name,
                            "image_analyzed": False,
                            **data,
                        }
                    except Exception as exc2:
                        last_error = exc2
                continue  # Thử model tiếp theo trong danh sách

        # Tất cả model đều thất bại → dùng fallback từ CNN
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

        # Thông tin hiệu chỉnh màu sắc từ visual_symptom_scores (nếu có)
        visual_evidence = classification.get("visual_evidence", {})
        ratios = visual_evidence.get("ratios", {})
        visual_hint = ""
        if ratios:
            visual_hint = (
                f"\n• Phân tích màu sắc ảnh (pixel-level): "
                f"xanh lá={ratios.get('green', 0):.0%}, "
                f"vàng/khảm(CMD)={ratios.get('yellow_mosaic', 0):.0%}, "
                f"xanh nhạt(CGM)={ratios.get('pale_mottle', 0):.0%}, "
                f"nâu/đốm(CBB)={ratios.get('brown_spot', 0):.0%}, "
                f"rỉ cam đậm(CBSD)={ratios.get('rust_orange', 0):.0%}, "
                f"vàng cam nhạt(CBSD)={ratios.get('cbsd_light', 0):.0%}, "
                f"hoại tử tối={ratios.get('dark_necrosis', 0):.0%}, "
                f"biến thiên màu={ratios.get('color_variation', 0):.0%}"
            )

        if entropy_ratio > 0.78:
            conf_note = f"⚠️ CẢNH BÁO NGHIÊM TRỌNG: CNN gần như ngẫu nhiên (entropy {entropy_ratio * 100:.0f}% mức tối đa) — BỎ QUA hoàn toàn kết quả CNN, dùng quan sát ảnh làm cơ sở duy nhất."
        elif cnn_conf < 40:
            conf_note = f"⚠️ CNN không tin cậy ({cnn_conf:.1f}%) — ưu tiên quan sát ảnh."
        elif cnn_conf < 60:
            conf_note = f"⚠️ CNN độ tin cậy thấp ({cnn_conf:.1f}%) — đối chiếu kỹ với ảnh."
        else:
            conf_note = f"CNN tin cậy ({cnn_conf:.1f}%)."

        low_conf_note = " ⚠️ KHÔNG TIN CẬY" if cnn_conf < 40 else (" ⚠️ THẤP" if cnn_conf < 60 else "")
        yolo_found = "Có" if detection["found"] else "Không – phân tích toàn bộ ảnh gốc"

        return f"""
Bạn là chuyên gia bệnh học thực vật cấp cao với 20 năm kinh nghiệm. Nhiệm vụ: phân tích hình ảnh lá cây thực tế, chẩn đoán bệnh chính xác và đưa ra nội dung tư vấn CHI TIẾT, ĐẦY ĐỦ, DÀI để người dùng có thể dùng làm tài liệu tham khảo đáng tin cậy.

YÊU CẦU ĐỘ DÀI — BẮT BUỘC TUÂN THỦ:
• summary: TỐI THIỂU 150 từ — mô tả đầy đủ triệu chứng, mức độ, nhóm bệnh và lý do kết luận
• disease_evidence: TỐI THIỂU 100 từ — giải thích bằng chứng hình ảnh cụ thể và đối chiếu với các nhóm bệnh khác
• causes: TỐI THIỂU 130 từ — tên khoa học tác nhân, cơ chế, đường lây, thời gian ủ bệnh
• favorable_conditions: TỐI THIỂU 130 từ — điều kiện thời tiết, môi trường, canh tác chi tiết
• care_steps: TỐI THIỂU 8 bước, mỗi bước TỐI THIỂU 25 từ
• prevention: TỐI THIỂU 6 biện pháp, mỗi biện pháp TỐI THIỂU 25 từ
• recommendations: TỐI THIỂU 5 khuyến nghị, mỗi mục TỐI THIỂU 25 từ
• recommended_products: TỐI THIỂU 3 sản phẩm/thuốc cụ thể với tên thương mại + liều dùng
• Nếu viết quá ngắn — kết quả sẽ bị từ chối và yêu cầu viết lại

QUY ƯỚC NGÔN NGỮ BẮT BUỘC:
• TUYỆT ĐỐI không dùng: "sắn", "lá sắn", "cây sắn", "bệnh sắn", "cassava", "mì"
• Chỉ gọi nhóm bệnh bằng: CBB, CBSD, CGM, CMD, Healthy kèm mô tả triệu chứng

═══════════════════════════════════════════════
BƯỚC 1 — PHÂN TÍCH HÌNH ẢNH THỰC TẾ (ƯU TIÊN CAO NHẤT)
═══════════════════════════════════════════════
Quan sát KỸ hình ảnh lá cây được gửi kèm. Ghi nhận CỤ THỂ những gì bạn THỰC SỰ THẤY:

• Màu sắc lá: xanh đồng đều | vàng nhạt | đốm vàng | khảm vàng-xanh | nâu/đen/hoại tử?
• Bề mặt lá: phẳng mịn | nhăn | cong vênh | phồng rộp | biến dạng?
• Vết tổn thương: vị trí, màu sắc, hình dạng (đốm góc/vòng/vệt), mức độ lan?
• Gân lá: bình thường | đổi màu | vết nâu/vàng dọc gân?
• Mép lá: nguyên vẹn | hoại tử khô | mềm thối?
• Mật độ tổn thương: nhẹ (<20% diện tích) | vừa (20-60%) | nặng (>60%)?

BẢNG NHẬN DIỆN 5 NHÓM:
1. CBB     → đốm lá góc/đa giác, viền vàng chuyển nâu, mô cháy khô, có thể héo phần ngọn
2. CBSD    → vệt vàng/nâu hoặc rỉ sắt dọc gân lá, hoại tử nâu-vàng xen kẽ, mảng sọc không đều
3. CGM     → lốm đốm xanh nhạt không đều, lá biến dạng nhẹ, hơi nhăn, mảng xanh-vàng mịn
4. CMD     → khảm vàng-xanh rõ rệt, lá nhăn/cuộn mạnh, biến dạng đáng kể, mảng khảm tương phản
5. Healthy → xanh đồng đều, không đốm, không biến dạng, không tổn thương

═══════════════════════════════════════════════
BƯỚC 2 — KẾT QUẢ CNN (CHỈ LÀ THAM KHẢO — CÓ THỂ SAI)
═══════════════════════════════════════════════
Mô hình CNN (EfficientNetB3, {tta_runs} lần TTA, mode={preprocess_mode}):
• Chẩn đoán CNN: {cnn_label}{low_conf_note}
• YOLO phát hiện lá: {yolo_found}
• Đánh giá CNN: {conf_note}{visual_hint}
• Xác suất 5 nhóm (cao → thấp):
{top_predictions}

═══════════════════════════════════════════════
BƯỚC 3 — KẾT LUẬN (QUAN SÁT ẢNH LÀ NGUỒN CHÍNH — KHÔNG ĐƯỢC SAO CHÉP CNN)
═══════════════════════════════════════════════
⚠️ CẢNH BÁO QUAN TRỌNG: CNN CÓ THỂ SAI. Bạn PHẢI tự quan sát ảnh và đưa ra kết luận độc lập.

TRƯỚC KHI điền final_diagnosis, hãy xác nhận 3 bằng chứng hình ảnh cụ thể bạn THỰC SỰ THẤY trong ảnh. Chỉ điền final_diagnosis sau khi đã có đủ 3 bằng chứng đó.

QUY TẮC BẮT BUỘC:
• final_diagnosis LUÔN LUÔN dựa vào những gì bạn thấy trong ảnh — KHÔNG bao giờ chỉ copy CNN.
• Nếu quan sát ảnh và CNN TRÙNG KHỚP → cnn_agreement = "agree".
• Nếu quan sát ảnh KHÁC CNN → final_diagnosis = theo quan sát ảnh, cnn_agreement = "disagree", giải thích rõ trong disease_evidence tại sao ảnh cho thấy kết luận khác CNN.
• Nếu ảnh không đủ rõ → cnn_agreement = "uncertain", dùng CNN làm tham khảo và ghi rõ trong warning.
• Ví dụ: CNN bảo CMD nhưng ảnh thấy rõ vệt cam-rỉ dọc gân → final_diagnosis = "cassava_brown_streak_disease", cnn_agreement = "disagree".

═══════════════════════════════════════════════
YÊU CẦU ĐẦU RA — JSON TIẾNG VIỆT HỢP LỆ
═══════════════════════════════════════════════
{{
  "headline": "Tiêu đề ≤12 từ phản ánh chẩn đoán từ quan sát ảnh thực tế",
  "final_diagnosis": "một trong: cassava_bacterial_blight | cassava_brown_streak_disease | cassava_green_mottle | cassava_mosaic_disease | healthy",
  "cnn_agreement": "agree | disagree | uncertain",
  "visual_confidence": "high | medium | low",
  "summary": "KHÔNG dùng 'sắn'/'cassava'. TỐI THIỂU 120 TỪ. Mô tả toàn diện: màu sắc tổng thể lá (xanh/vàng/nâu/đốm), hình dạng và vị trí vết tổn thương, tình trạng gân lá và mép lá, mức độ lan rộng (ước tính % diện tích), mức độ nghiêm trọng, nhóm bệnh phù hợp nhất và giải thích TẠI SAO triệu chứng thực tế dẫn đến kết luận này.",
  "visual_observations": [
    "Quan sát 1 (tối thiểu 20 từ): màu sắc nền lá và sắc độ tổng thể — ghi rõ tông màu cụ thể",
    "Quan sát 2 (tối thiểu 20 từ): hình dạng, kích thước, viền và màu trung tâm của vết tổn thương",
    "Quan sát 3 (tối thiểu 20 từ): tình trạng gân lá — đổi màu, vệt nâu/vàng, hoặc bình thường",
    "Quan sát 4 (tối thiểu 20 từ): mật độ và phân bố tổn thương (ước tính % diện tích bị ảnh hưởng)",
    "Quan sát 5 (tối thiểu 20 từ): kết cấu bề mặt — mịn/nhăn/phồng/hoại tử khô/tiết dịch",
    "Quan sát 6 (tối thiểu 20 từ): đặc điểm nổi bật giúp phân biệt với nhóm bệnh dễ nhầm"
  ],
  "disease_evidence": "KHÔNG dùng 'sắn'/'cassava'. TỐI THIỂU 80 TỪ. Giải thích chi tiết: (1) bằng chứng hình ảnh CỤ THỂ nào trong ảnh là cơ sở cho chẩn đoán, (2) tại sao đặc điểm đó đặc trưng cho nhóm bệnh được chọn chứ không phải nhóm khác, (3) phân tích so sánh với ít nhất 2 nhóm bệnh dễ nhầm, (4) đánh giá mức độ chắc chắn của kết luận.",
  "causes": "KHÔNG dùng 'sắn'/'cassava'. TỐI THIỂU 100 TỪ. Trình bày đầy đủ: tên khoa học của tác nhân gây bệnh (vi khuẩn/virus/nấm), cơ chế xâm nhiễm vào tế bào và mô lá, các con đường lây lan chính (gió/mưa/côn trùng/dụng cụ cắt tỉa/tiếp xúc cơ học/đất), thời gian ủ bệnh từ khi nhiễm đến khi xuất hiện triệu chứng đầu tiên, và tốc độ lan rộng điển hình trong điều kiện thuận lợi.",
  "favorable_conditions": "KHÔNG dùng 'sắn'/'cassava'. TỐI THIỂU 100 TỪ. Mô tả chi tiết: khoảng nhiệt độ tối ưu (°C) cho tác nhân phát triển mạnh nhất, ngưỡng độ ẩm không khí và đất kích hoạt bệnh, mùa vụ và thời điểm trong năm có nguy cơ cao nhất, điều kiện ánh sáng và thông gió ảnh hưởng thế nào, các yếu tố canh tác làm tăng nguy cơ (mật độ trồng dày, bón phân đạm dư, tưới phun làm ướt lá, đất nghèo dinh dưỡng).",
  "care_steps": [
    "Bước 1 — NGAY LẬP TỨC (trong vòng 2 giờ): Cách ly cây bị bệnh khỏi các cây lân cận, dừng mọi hoạt động cắt tỉa tại khu vực đó để tránh lây lan cơ học.",
    "Bước 2 — NGÀY ĐẦU TIÊN: Dùng kéo/dao khử trùng cẩn thận cắt bỏ toàn bộ lá và cành bị tổn thương nặng (>50% diện tích). Đốt hoặc chôn sâu, không để tại vườn.",
    "Bước 3 — NGÀY 1-3: Vệ sinh toàn bộ dụng cụ cắt tỉa bằng cồn 70% hoặc dung dịch thuốc tím 0.1%. Rửa tay kỹ trước và sau khi tiếp xúc với cây bệnh.",
    "Bước 4 — NGÀY 2-5: Phun thuốc đặc trị phù hợp với nhóm bệnh đã chẩn đoán. Pha đúng nồng độ theo hướng dẫn, phun đều cả mặt trên và dưới lá.",
    "Bước 5 — TUẦN 1-2: Điều chỉnh chế độ tưới nước — chuyển sang tưới gốc thay vì tưới phun, tưới vào buổi sáng để lá khô trước tối.",
    "Bước 6 — TUẦN 2-3: Tỉa thưa tán lá để tăng thông thoáng, giảm độ ẩm vi khí hậu. Loại bỏ cỏ dại quanh gốc để giảm độ ẩm đất.",
    "Bước 7 — SAU 2-4 TUẦN: Bón bổ sung phân kali và vi lượng (Ca, Mg, B) để tăng sức đề kháng. Tránh bón đạm quá nhiều trong giai đoạn hồi phục."
  ],
  "next_steps": [
    "Ngày 3: Kiểm tra xem vết bệnh có lan thêm không — đo diện tích và chụp ảnh ghi lại để so sánh.",
    "Ngày 7: Đánh giá hiệu quả thuốc điều trị — nếu vết bệnh không thu nhỏ, cân nhắc đổi loại thuốc hoặc tăng nồng độ.",
    "Ngày 14: Khảo sát toàn bộ vườn/khu trồng để phát hiện cây khác có dấu hiệu tương tự.",
    "Ngày 21-30: Nếu triệu chứng không cải thiện hoặc tiếp tục lan rộng — liên hệ cán bộ nông nghiệp địa phương hoặc gửi mẫu lá đến phòng xét nghiệm.",
    "Dài hạn: Lưu nhật ký ảnh và ghi chép diễn biến từng tuần để có tài liệu theo dõi và phòng ngừa cho vụ sau."
  ],
  "prevention": [
    "Khử trùng dụng cụ: Ngâm kéo/dao trong cồn 70% hoặc dung dịch javel 1% ít nhất 30 giây trước và sau mỗi lần cắt tỉa, đặc biệt khi di chuyển giữa các cây khác nhau.",
    "Luân canh cây trồng: Không trồng cùng loại cây trên cùng một vị trí trong ít nhất 2-3 vụ liên tiếp để phá vỡ vòng đời mầm bệnh tồn tại trong đất.",
    "Quản lý mật độ trồng: Đảm bảo khoảng cách tối thiểu theo khuyến cáo, tỉa bớt cành để ánh sáng và gió có thể lưu thông tốt giữa các cây.",
    "Giám sát định kỳ: Kiểm tra kỹ cả mặt trên và dưới lá mỗi 7-10 ngày để phát hiện triệu chứng từ giai đoạn đầu khi còn dễ kiểm soát.",
    "Quản lý nguồn giống: Chỉ sử dụng giống/hom/vật liệu trồng từ nguồn đã được kiểm dịch hoặc vùng không có bệnh. Không lấy vật liệu giống từ cây bệnh.",
    "Điều chỉnh tưới nước: Ưu tiên tưới nhỏ giọt hoặc tưới gốc thay vì tưới phun. Nếu phải tưới phun, tưới vào sáng sớm để lá có đủ thời gian khô trước chiều tối."
  ],
  "recommendations": [
    "Chụp ảnh chẩn đoán chuẩn: Chụp cận cảnh lá bệnh dưới ánh sáng tự nhiên đủ sáng, thấy rõ cả mặt trên, mặt dưới lá và gân lá — ảnh mờ hoặc thiếu sáng làm giảm độ chính xác phân tích AI đáng kể.",
    "Không tự điều trị bừa bãi: Tránh phun thuốc hóa học khi chưa xác định chắc chắn nhóm bệnh — phun sai thuốc không những không hiệu quả mà còn tạo kháng thuốc và gây hại cho môi trường.",
    "Cân bằng dinh dưỡng: Bổ sung phân bón cân đối NPK theo giai đoạn sinh trưởng — cây đủ dinh dưỡng có sức đề kháng cao hơn đáng kể so với cây suy dinh dưỡng.",
    "Lưu hồ sơ bệnh: Ghi chép ngày phát hiện, triệu chứng ban đầu, phương án xử lý và kết quả — tài liệu này rất có giá trị để đưa ra quyết định chính xác trong các vụ trồng tiếp theo.",
    "Tham vấn chuyên gia: Với bệnh lan rộng trên diện tích lớn (>10% cây trồng bị ảnh hưởng) hoặc triệu chứng không rõ ràng, nên gửi mẫu lá đến trạm bảo vệ thực vật hoặc trung tâm nông nghiệp địa phương để được xét nghiệm chính xác."
  ],
  "warning": "KHÔNG dùng 'sắn'/'cassava'. Ghi rõ mức độ tin cậy của chẩn đoán dựa trên chất lượng ảnh và sự khớp giữa CNN và quan sát trực quan. Nêu rõ những điểm còn không chắc chắn và lý do cần kiểm tra thêm thực địa trước khi áp dụng biện pháp xử lý diện rộng.",
  "health_score": 50,
  "economic_impact": "KHÔNG dùng 'sắn'/'cassava'. 3-4 câu ước tính: % năng suất có thể mất nếu không xử lý kịp thời, chi phí thuốc và công xử lý ước tính, nguy cơ lây lan sang các cây/khu vực lân cận và thiệt hại tiềm năng, so sánh chi phí phòng ngừa vs chi phí xử lý khi bệnh đã nặng.",
  "spread_level": "thấp | trung bình | cao",
  "treatment_schedule": [
    {{"action": "Cách ly cây bệnh + cắt bỏ lá tổn thương nặng + khử trùng dụng cụ", "days_later": 0}},
    {{"action": "Phun thuốc đặc trị lần 1, kiểm tra mức độ lan rộng", "days_later": 3}},
    {{"action": "Đánh giá hiệu quả điều trị, phun thuốc lần 2 nếu cần, điều chỉnh tưới nước", "days_later": 7}},
    {{"action": "Kiểm tra toàn bộ vườn, phát hiện cây lây lan, bón phân hỗ trợ phục hồi", "days_later": 14}},
    {{"action": "Đánh giá tổng thể hiệu quả điều trị, quyết định tiếp tục hay dừng xử lý", "days_later": 21}}
  ],
  "disease_progression": {{
    "3_ngày": "Mô tả cụ thể tình trạng vết bệnh và mức độ lan rộng dự kiến sau 3 ngày không can thiệp",
    "7_ngày": "Mô tả cụ thể mức độ nghiêm trọng và ảnh hưởng đến toàn cây sau 7 ngày không can thiệp",
    "14_ngày": "Mô tả khả năng lây lan sang cây khác và thiệt hại tổng thể sau 14 ngày không can thiệp",
    "30_ngày": "Mô tả hậu quả nghiêm trọng và thiệt hại kinh tế nếu để bệnh phát triển tự nhiên trong 1 tháng"
  }},
  "disease_stage": "Giai đoạn bệnh hiện tại dựa trên diện tích và mức độ tổn thương quan sát được trong ảnh. Chỉ rõ: 'Giai đoạn đầu — triệu chứng mới xuất hiện, dưới 20% diện tích lá bị ảnh hưởng' hoặc 'Giai đoạn giữa — triệu chứng rõ ràng, 20-60% diện tích lá bị ảnh hưởng' hoặc 'Giai đoạn nặng — triệu chứng nghiêm trọng, trên 60% diện tích lá bị ảnh hưởng, nguy cơ lan sang cây khác cao'. Giải thích ngắn gọn tại sao xếp vào giai đoạn đó.",
  "affected_parts": [
    "Liệt kê cụ thể từng bộ phận lá bị ảnh hưởng quan sát thấy trong ảnh: lá non / lá trưởng thành / lá già, gân chính / gân phụ, mép lá, cuống lá, bề mặt lá (trên/dưới), v.v. Mỗi mục mô tả rõ loại tổn thương trên bộ phận đó (đổi màu / hoại tử / biến dạng...)."
  ],
  "recommended_products": [
    "Sản phẩm 1: Tên thương mại + hoạt chất chính + loại (thuốc trừ khuẩn/kháng virus/kháng nấm/phân bón vi lượng) + liều lượng pha + cách dùng (phun lá/tưới gốc) + số lần xử lý khuyến nghị.",
    "Sản phẩm 2: Tương tự cấu trúc trên, ưu tiên sản phẩm khác nhóm hoạt chất để tránh kháng thuốc.",
    "Sản phẩm 3: Sản phẩm bổ sung sức đề kháng / phân bón hỗ trợ phục hồi (kali, canxi, vi lượng) với liều dùng cụ thể."
  ]
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
            "disease_progression": data.get("disease_progression", {}),
            "visual_observations": [
                str(i).strip() for i in data.get("visual_observations", []) if str(i).strip()
            ],
            "disease_evidence": str(data.get("disease_evidence", "")).strip(),
            "recommendations": [str(i).strip() for i in data.get("recommendations", []) if str(i).strip()],
            "causes": str(data.get("causes", "")).strip(),
            "favorable_conditions": str(data.get("favorable_conditions", "")).strip(),
            "prevention": [str(i).strip() for i in data.get("prevention", []) if str(i).strip()],
            "disease_stage": str(data.get("disease_stage", "")).strip(),
            "affected_parts": [str(i).strip() for i in data.get("affected_parts", []) if str(i).strip()],
            "recommended_products": [str(i).strip() for i in data.get("recommended_products", []) if str(i).strip()],
        }

    def _sanitize_text(self, value: str) -> str:
        # NFC normalize để đảm bảo Unicode nhất quán (Gemini có thể trả về NFD)
        cleaned = unicodedata.normalize("NFC", str(value))

        # Bước 1: Thay tên tiếng Anh đầy đủ (regex, case-insensitive)
        cleaned = re.sub(r"Cassava\s+Bacterial\s+Blight", "CBB", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"Cassava\s+Brown\s+Streak\s+Disease", "CBSD", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"Cassava\s+Green\s+Mottle", "CGM", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"Cassava\s+Mosaic\s+Disease", "CMD", cleaned, flags=re.IGNORECASE)

        # Bước 2: Cụm có "sắn" — lowercase toàn bộ chuỗi tạm để bắt mọi case
        # Lưu bản lowercase để phát hiện, thay trên bản gốc
        lower = cleaned.lower()
        compounds = [
            ("lá sắn", "lá"), ("cây sắn", "cây"), ("bệnh sắn", "bệnh lá"),
            ("củ sắn", "củ"), ("giống sắn", "giống"), ("trồng sắn", "trồng"),
            ("vườn sắn", "vườn"), ("ruộng sắn", "ruộng"), ("cánh đồng sắn", "cánh đồng"),
            ("hom sắn", "hom"), ("rễ sắn", "rễ"),
        ]
        for needle, replacement in compounds:
            idx = lower.find(needle)
            while idx != -1:
                cleaned = cleaned[:idx] + replacement + cleaned[idx + len(needle):]
                lower = cleaned.lower()
                idx = lower.find(needle)

        # Bước 3: standalone "sắn" còn sót — tìm/thay trực tiếp trên lowercase
        # split/join đảm bảo bắt được mọi vị trí kể cả cuối dòng/đầu câu
        cleaned = cleaned.replace("sắn", "cây").replace("Sắn", "Cây").replace("SẮN", "CÂY")

        # Bước 4: "cassava" còn sót
        cleaned = re.sub(r"cassava", "nhóm bệnh", cleaned, flags=re.IGNORECASE)

        return re.sub(r"\s{2,}", " ", cleaned).strip()

    def _sanitize_report(self, value):
        if isinstance(value, dict):
            return {key: self._sanitize_report(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._sanitize_report(item) for item in value]
        if isinstance(value, str):
            return self._sanitize_text(value)
        return value

    def _fallback_report(self, classification: dict, reason: str) -> dict:
        label = classification["display_label"]
        confidence = classification["confidence"] * 100
        return self._sanitize_report({
            "source": "fallback",
            "model": "local-template",
            "image_analyzed": False,
            "headline": f"Kết quả CNN: {label}",
            "final_diagnosis": classification.get("label", ""),
            "cnn_agreement": "uncertain",
            "visual_confidence": "low",
            "summary": (
                f"CNN đang nghiêng về nhóm '{label}' với độ tin cậy khoảng {confidence:.1f}%. "
                "Đây là gợi ý ban đầu dựa trên xác suất model và hiệu chỉnh triệu chứng ảnh. "
                "Hãy đối chiếu thêm màu sắc, dạng đốm, mức độ lan và tình trạng gân lá trước khi xử lý."
            ),
            "care_steps": [
                "Tách riêng cây có dấu hiệu bất thường để hạn chế lây lan.",
                "Kiểm tra kỹ mặt trên, mặt dưới lá và chụp thêm ảnh rõ, đủ sáng nếu cần.",
                "Điều chỉnh tưới nước, ánh sáng và độ thông thoáng quanh cây.",
                "Loại bỏ phần lá hư nặng nếu cây đã bị tổn thương rõ rệt.",
                "Ghi lại ảnh sau 3-5 ngày để so sánh tốc độ lan rộng và mức độ hồi phục.",
            ],
            "next_steps": [
                "Theo dõi sự thay đổi của đốm lá trong 3-5 ngày tiếp theo.",
                "Kiểm tra xem vết bệnh lan theo gân lá, theo mép lá hay thành mảng rải rác.",
                "Tham khảo cán bộ nông nghiệp hoặc chuyên gia nếu triệu chứng lan rộng.",
            ],
            "prevention": [
                "Giữ tán lá thông thoáng, tránh để lá ẩm kéo dài.",
                "Vệ sinh dụng cụ cắt tỉa và loại bỏ phần lá bệnh nặng khỏi khu vực trồng.",
                "Theo dõi định kỳ để phát hiện sớm đốm, khảm màu hoặc vệt nâu mới xuất hiện.",
            ],
            "recommendations": [
                "Ưu tiên chụp ảnh cận lá, đủ sáng, thấy rõ cả gân và mép lá để lần phân tích sau ổn định hơn.",
                "Không xử lý hóa chất đại trà khi chưa xác nhận mức độ lan và nhóm triệu chứng chính.",
            ],
            "warning": reason,
            "health_score": 50,
            "economic_impact": "Nếu triệu chứng tiếp tục lan, cây có thể giảm quang hợp, sinh trưởng chậm và mất năng suất.",
            "spread_level": "unknown",
            "treatment_schedule": [],
            "disease_progression": {},
            "visual_observations": [],
            "disease_evidence": "",
            "causes": "Có thể liên quan đến tác nhân gây bệnh trên lá hoặc điều kiện môi trường làm mô lá suy yếu. Cần đối chiếu thêm dấu hiệu thực địa để xác nhận.",
            "favorable_conditions": "Độ ẩm cao, lá ướt lâu, vườn kém thông thoáng và cây suy dinh dưỡng thường làm triệu chứng phát triển nhanh hơn.",
            "disease_stage": "",
            "affected_parts": [],
            "recommended_products": [],
        })
