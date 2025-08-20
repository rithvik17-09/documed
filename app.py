

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import numpy as np
import cv2
import io
from PIL import Image

app = FastAPI(title="Medical AI Backend")

HF_API_KEY = "#" 
API_URL = "https://api-inference.huggingface.co/models/bert-base-uncased"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

diagnosis_pipeline = pipeline("text-classification", model="bert-base-uncased")

def preprocess_image(file: UploadFile):
    # Convert to OpenCV image
    img = Image.open(io.BytesIO(file.file.read())).convert("L")
    img = np.array(img)

    img = cv2.resize(img, (224, 224))
    img = cv2.equalizeHist(img) 

    return img

class SymptomRequest(BaseModel):
    symptoms: str

@app.post("/diagnose-text")
def diagnose_text(request: SymptomRequest):
    result = diagnosis_pipeline(request.symptoms)
    return {"input": request.symptoms, "diagnosis": result}

@app.post("/diagnose-image")
async def diagnose_image(file: UploadFile = File(...)):
    processed_img = preprocess_image(file)

    abnormality_detected = np.mean(processed_img) < 100

    return {
        "abnormality": abnormality_detected,
        "note": "This is a DSP-preprocessed result. Model can be plugged in here."
    }
@app.get("/")
def root():
    return {"status": "Medical AI Backend Running ✅"}
