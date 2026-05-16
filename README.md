# LeafAI

Repo deploy: `websitechuandoanbenhchola`

Web app chan doan benh la cay theo pipeline:

1. YOLO (`moduleyolola/best.pt`) tim vung la.
2. CNN (`model_0.h5`) phan loai benh.
3. ChatGPT API sinh mo ta va goi y cham soc.

## Cong nghe

- Frontend: HTML, CSS, JavaScript thuan
- Backend: Flask
- Inference: Ultralytics YOLO + TensorFlow/Keras

## Luu y moi truong

- Nen dung Python `3.11.x`
- `model_0.h5` co 5 lop dau ra
- `config/cnn_labels.json` dang map theo bo cassava 5 lop pho bien:
  - `cassava_bacterial_blight`
  - `cassava_brown_streak_disease`
  - `cassava_green_mottle`
  - `cassava_mosaic_disease`
  - `healthy`

## Chay local

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python app.py
```

Mo trinh duyet tai `http://localhost:5000`

## Cau truc chinh

- `app.py`: route Flask va API upload/phan tich
- `services/yolo_service.py`: phat hien va crop vung la
- `services/cnn_service.py`: load `model_0.h5` va phan loai
- `services/llm_service.py`: goi ChatGPT API hoac fallback local
- `templates/index.html`: giao dien
- `static/css/styles.css`: style
- `static/js/app.js`: upload, goi API, render ket qua

## Deploy Render

Project da co san `render.yaml` va `.python-version`.

### Cach deploy

1. Push repo nay len GitHub.
2. Tren Render, chon `New +` -> `Blueprint`.
3. Chon repo `websitechuandoanbenhchola`.
4. Khi Render hoi secret, nhap `OPENAI_API_KEY`.
5. Sau khi deploy xong, kiem tra `/api/health`.

### Free tier

- `render.yaml` da duoc dat san `plan: free`
- Free web service se sleep sau 15 phut khong co traffic
- Free tier khong phu hop production
- File upload local co the mat sau restart/redeploy

### Bao mat

- `.env` da duoc ignore, khong push len GitHub.
- `OPENAI_API_KEY` chi dat trong Render secret env var, khong dat trong frontend.
- Nen rotate API key da dan trong chat va tao key moi cho production.

###

File xử lý trực tiếp (tt phụ)

app.py
File chạy Flask, nhận upload ảnh, gọi API phân tích, tạo report, tạo QR.

services/pipeline.py
Điều phối luồng: upload ảnh → YOLO → CNN → Gemini → trả kết quả.

services/yolo_service.py
Xử lý YOLO, tìm vùng lá, crop ảnh lá để đưa sang CNN.

services/cnn_service.py
Xử lý CNN, phân loại 5 nhóm bệnh, tính phần trăm, cảnh báo độ tin cậy.

services/llm_service.py
Xử lý Gemini, tạo chẩn đoán tổng hợp, giải thích, khuyến nghị, cảnh báo.

static/js/app.js
Xử lý phía giao diện: gửi ảnh lên backend, nhận kết quả, hiển thị CNN/Gemini/report/QR.

templates/index.html
Khung giao diện chính của phần mềm.

static/css/styles.css
Giao diện, bố cục, màu sắc, responsive.

File cấu hình và model

services/config.py
Đọc .env, đường dẫn model, Gemini API key, thư mục upload, giới hạn dung lượng ảnh.

services/exceptions.py
Định nghĩa lỗi riêng của app, ví dụ lỗi upload sai định dạng hoặc lỗi phân tích.

config/cnn_labels.json
Map thứ tự đầu ra CNN với 5 nhóm bệnh.

model_0.h5
Model CNN đã huấn luyện.

moduleyolola/best.pt và moduleyolola/last.pt
Model YOLO để phát hiện vùng lá.

File phục vụ chạy/deploy

requirements.txt
Danh sách thư viện cần cài.

render.yaml
Cấu hình deploy lên Render.

.env và .env.example
Chứa/cung cấp mẫu biến môi trường như Gemini API key, đường dẫn model.

uploads/originals và uploads/processed để lưu ảnh lưu ảnh người dùng upload và ảnh đã xử lý và **pycache** là cache Python tự sinh.

# CHi tiết luồng xử lý

1. app.py (“File app.py là trung tâm điều phối web. Khi người dùng upload ảnh, file này nhận request, gọi pipeline xử lý, sau đó trả kết quả JSON về frontend.”)
   Đây là file chính để chạy web Flask.

Vai trò:

Khởi tạo ứng dụng Flask.
Tạo các API chính như:
/: hiển thị giao diện web.
/api/health: kiểm tra hệ thống có model YOLO, CNN, Gemini API chưa.
/api/analyze: nhận ảnh người dùng upload và gọi pipeline phân tích.
/api/report: tạo báo cáo kết quả.
/api/qr: tạo QR chứa tóm tắt kết quả.
Trả ảnh đã xử lý về giao diện.

2. services/pipeline.py
   Đây là file điều phối toàn bộ quy trình AI.

Vai trò:

Lưu ảnh người dùng upload.
Gọi YOLO để phát hiện/tách vùng lá.
Gọi CNN để phân loại bệnh.
Gọi Gemini để tạo phần chẩn đoán tổng hợp.
Gom kết quả YOLO, CNN, Gemini thành một kết quả hoàn chỉnh trả về frontend.
Luồng chính:

Ảnh upload
→ YOLO tách vùng lá
→ CNN phân loại 5 nhóm bệnh
→ Gemini giải thích và tư vấn
→ Trả kết quả cho giao diện

“File pipeline.py giống như bộ điều phối. Nó không trực tiếp dự đoán bệnh, nhưng nó sắp xếp các bước AI chạy đúng thứ tự.”

3. services/yolo_service.py
   Đây là file xử lý phát hiện vùng lá bằng YOLO.

Vai trò:

Nhận ảnh gốc.
Dùng YOLO để tìm vùng lá rõ nhất.
Cắt ảnh vùng lá để đưa sang CNN.
Nếu YOLO không tìm thấy lá, hệ thống vẫn dùng toàn bộ ảnh gốc để tránh bị dừng.

“YOLO giúp giảm nhiễu nền ảnh. Thay vì CNN phải phân tích cả ảnh có nền, bàn tay, đất, vật thể khác, YOLO giúp tập trung vào vùng lá.”

4. services/cnn_service.py
   Đây là file quan trọng nhất cho phần CNN.

Vai trò:

Tải model model_0.h5.
Đọc file nhãn config/cnn_labels.json.
Tiền xử lý ảnh về đúng kích thước model.
Chạy CNN để lấy xác suất 5 nhóm:
Healthy
CMD
CGM
CBSD
CBB
Sắp xếp nhóm bệnh theo phần trăm từ cao xuống thấp.
Trả về nhóm cao nhất, độ tin cậy, top prediction và cảnh báo.
Phần đã tối ưu thêm:

Có TTA: chạy ảnh gốc, ảnh lật ngang, ảnh lật dọc để kết quả ổn định hơn.
Có kiểm tra entropy/xác suất: nếu model phân vân thì cảnh báo.
Có hiệu chỉnh triệu chứng ảnh: đọc thêm các dấu hiệu như đốm nâu, rỉ nâu/cam, vàng-khảm, đốm xanh nhạt để hỗ trợ CNN bám sát ảnh hơn.

“File cnn_service.py là nơi CNN phân loại bệnh. Em không chỉ lấy kết quả thô từ model, mà còn có bước kiểm tra độ tin cậy và hiệu chỉnh theo triệu chứng ảnh để hạn chế trường hợp CNN kết luận lệch nhóm.”

5. services/llm_service.py
   Đây là file xử lý Gemini.

Vai trò:

Nhận kết quả từ YOLO và CNN.
Tạo prompt gửi cho Gemini.
Yêu cầu Gemini trả về JSON có cấu trúc.
Sinh các nội dung như:
Chẩn đoán tổng hợp.
Mô tả triệu chứng.
Bằng chứng hình ảnh.
Nguyên nhân.
Điều kiện thuận lợi.
Cách xử lý.
Theo dõi tiếp theo.
Phòng ngừa.
Cảnh báo độ tin cậy.
Phần đã fix:

Không để Gemini trả lời tự do.
Bắt Gemini bám vào 5 nhóm bệnh.
Bỏ việc mặc định nhắc “sắn/lá sắn/cassava”.
Có hàm làm sạch kết quả nếu Gemini sinh ra từ không mong muốn.

“File llm_service.py giúp chuyển kết quả CNN thành lời giải thích dễ hiểu cho người dùng. Gemini không thay CNN, mà diễn giải, đối chiếu và bổ sung hướng xử lý.”

6. config/cnn_labels.json
   Đây là file cấu hình nhãn bệnh cho CNN.

Vai trò:

Xác định thứ tự 5 lớp đầu ra của model.
Đảm bảo output của CNN được map đúng nhãn bệnh.
Ví dụ:

[
"cassava_bacterial_blight",
"cassava_brown_streak_disease",
"cassava_green_mottle",
"cassava_mosaic_disease",
"healthy"
]

“File này giúp hệ thống hiểu index đầu ra của CNN tương ứng với nhóm bệnh nào.”

7. services/config.py
   Đây là file cấu hình hệ thống.

Vai trò:

Đọc biến môi trường từ .env.
Lấy đường dẫn model YOLO, CNN.
Lấy Gemini API key.
Cấu hình thư mục upload.
Cấu hình dung lượng ảnh cho phép.
Cách nói:

“File config.py giúp gom các cấu hình quan trọng vào một chỗ, để khi deploy web chỉ cần đổi biến môi trường chứ không phải sửa code chính.”

8. templates/index.html
   Đây là giao diện HTML chính.

Vai trò:

Tạo bố cục trang web.
Có khu vực upload ảnh.
Có khu vực hiển thị ảnh gốc, ảnh YOLO, ảnh crop.
Có khu vực hiển thị kết quả CNN.
Có khu vực hiển thị chẩn đoán tổng hợp từ Gemini.

“File này là phần khung giao diện mà người dùng nhìn thấy trên trình duyệt.”

9. static/js/app.js
   Đây là file JavaScript điều khiển giao diện.

Vai trò:

Bắt sự kiện người dùng chọn ảnh.
Gửi ảnh lên /api/analyze.
Nhận kết quả JSON từ backend.
Render kết quả CNN, biểu đồ phần trăm, cảnh báo, chẩn đoán Gemini.
Xử lý tải báo cáo và QR.

“File app.js là cầu nối giữa giao diện và backend. Nó gửi ảnh lên server và hiển thị kết quả phân tích cho người dùng.”

10. static/css/styles.css
    Đây là file giao diện CSS.

Vai trò:

Thiết kế màu sắc, layout, card, nút bấm.
Làm giao diện đẹp hơn, dễ đọc hơn.
Responsive cho màn hình khác nhau.

“File CSS giúp phần mềm có giao diện trực quan, dễ dùng, phù hợp để người dùng phổ thông thao tác.”
