from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import tempfile
import os
import predict  # ใช้ predict.py ของคุณ

app = FastAPI()

# ให้เว็บเข้ามาอ่านไฟล์ได้ทั้งหมด
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# ดึงหน้าเว็บ index.html
# ------------------------------
@app.get("/")
def serve_index():
    return FileResponse("index.html")


# ------------------------------
# Predict API
# ------------------------------
@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Predict
    emotion, prob = predict.predict_emotion(tmp_path)

    # Clean temp
    os.remove(tmp_path)

    return {
        "emotion": emotion,
        "probabilities": prob.tolist()
    }
