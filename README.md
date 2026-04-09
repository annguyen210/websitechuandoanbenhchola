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
