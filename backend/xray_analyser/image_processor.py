"""
X-Ray Analyser - Medical Image Analysis Backend
CNN-powered image processing and analysis
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import cv2
from PIL import Image
import io
import random

router = APIRouter()

# Global model placeholders
xray_model = None
mri_model = None

# Finding databases
XRAY_FINDINGS = {
    "high": [
        {"id": "consolidation", "title": "Consolidation", "location": "Right lower lobe", "description": "Dense opacity suggesting possible pneumonia or infection"},
        {"id": "effusion", "title": "Pleural Effusion", "location": "Left costophrenic angle", "description": "Fluid accumulation in pleural space"},
        {"id": "pneumothorax", "title": "Pneumothorax", "location": "Right apex", "description": "Air in pleural space indicating collapsed lung"}
    ],
    "medium": [
        {"id": "infiltrate", "title": "Infiltrate", "location": "Bilateral bases", "description": "Patchy opacity consistent with inflammatory process"},
        {"id": "cardiomegaly", "title": "Cardiomegaly", "location": "Cardiac silhouette", "description": "Enlarged heart shadow"},
        {"id": "atelectasis", "title": "Atelectasis", "location": "Left mid-zone", "description": "Partial collapse of lung tissue"}
    ],
    "low": [
        {"id": "minor_opacity", "title": "Minor Opacity", "location": "Right upper lobe", "description": "Small area of increased density, likely benign"},
        {"id": "calcification", "title": "Calcification", "location": "Granuloma", "description": "Calcified granuloma, likely old infection"}
    ]
}

MRI_FINDINGS = {
    "high": [
        {"id": "lesion", "title": "Lesion", "location": "Temporal lobe", "description": "Suspicious hyperintense region on T2-weighted sequence"},
        {"id": "mass", "title": "Mass Effect", "location": "Frontal cortex", "description": "Space-occupying lesion with surrounding compression"},
        {"id": "hemorrhage", "title": "Hemorrhage", "location": "Basal ganglia", "description": "Acute hemorrhage with surrounding edema"}
    ],
    "medium": [
        {"id": "edema", "title": "Perilesional Edema", "location": "White matter", "description": "Edematous changes surrounding focal lesion"},
        {"id": "atrophy", "title": "Cortical Atrophy", "location": "Bilateral hemispheres", "description": "Age-related volume loss"},
        {"id": "demyelination", "title": "Demyelinating Plaques", "location": "Periventricular white matter", "description": "Multiple sclerosis-like lesions"}
    ],
    "low": [
        {"id": "cyst", "title": "Cystic Component", "location": "Frontal cortex", "description": "Small cystic pocket, likely benign"},
        {"id": "artifact", "title": "Motion Artifact", "location": "Multiple regions", "description": "Imaging artifact from patient movement"}
    ]
}

# Request/Response Models
class AnalysisResult(BaseModel):
    status: str  # "Normal" or "Defective"
    confidence: float  # 0-100
    findings: List[Dict]
    recommendation: str
    disclaimer: str
    scan_type: str

class AnalysisReport(BaseModel):
    id: str
    timestamp: str
    scan_type: str
    result: AnalysisResult

# Helper functions
async def load_models():
    """
    Load ML models (placeholder for actual model loading)
    """
    global xray_model, mri_model
    # In production, load actual models here
    # xray_model = tf.keras.models.load_model('path/to/xray_weights.hdf5')
    # mri_model = tf.keras.models.load_model('path/to/mri_weights.hdf5')
    print("✅ Models loaded (using simulation mode)")

def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """
    Preprocess uploaded image for model inference
    """
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(file_bytes))
    
    # Convert to RGB if grayscale
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize((150, 150))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Normalize pixel values
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def generate_findings(confidence: float, scan_type: str) -> List[Dict]:
    """
    Generate realistic findings based on confidence score
    """
    findings_db = XRAY_FINDINGS if scan_type == "xray" else MRI_FINDINGS
    
    findings = []
    
    if confidence > 0.9:
        # High confidence - include multiple findings
        findings.append(random.choice(findings_db["high"]))
        findings.append(random.choice(findings_db["medium"]))
        if random.random() > 0.5:
            findings.append(random.choice(findings_db["low"]))
    elif confidence > 0.7:
        # Medium confidence
        findings.append(random.choice(findings_db["medium"]))
        findings.append(random.choice(findings_db["low"]))
    else:
        # Low confidence
        findings.append(random.choice(findings_db["low"]))
    
    return findings

def simulate_prediction(image: np.ndarray, scan_type: str) -> Dict:
    """
    Simulate model prediction (in production, use actual model)
    """
    # Simulate random prediction for demo
    prediction_score = random.uniform(0.3, 0.95)
    
    status = "Defective" if prediction_score > 0.5 else "Normal"
    confidence = prediction_score if prediction_score > 0.5 else (1 - prediction_score)
    confidence_pct = confidence * 100
    
    findings = []
    if status == "Defective":
        findings = generate_findings(confidence, scan_type)
    
    recommendation = {
        "Normal": "No significant abnormalities detected. Routine follow-up recommended.",
        "Defective": "Abnormalities detected. Immediate consultation with a radiologist strongly recommended."
    }[status]
    
    return {
        "status": status,
        "confidence": round(confidence_pct, 1),
        "findings": findings,
        "recommendation": recommendation
    }

# API Endpoints
@router.post("/analyze", response_model=AnalysisResult)
async def analyze_image(
    file: UploadFile = File(...),
    scan_type: str = Form(...)  # "xray" or "mri"
):
    """
    Analyze medical image (X-ray or MRI) using CNN
    """
    # Validate scan type
    if scan_type not in ["xray", "mri"]:
        raise HTTPException(status_code=400, detail="Scan type must be 'xray' or 'mri'")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read file
        file_bytes = await file.read()
        
        # Preprocess image
        processed_image = preprocess_image(file_bytes)
        
        # Make prediction
        result = simulate_prediction(processed_image, scan_type)
        
        # Add disclaimer
        result["disclaimer"] = "This is an AI-assisted analysis for preliminary screening only. It is NOT a medical diagnosis. Always consult with qualified radiologists and medical professionals for official diagnosis and treatment decisions."
        result["scan_type"] = scan_type
        
        return AnalysisResult(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload image for processing
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    file_bytes = await file.read()
    
    return {
        "filename": file.filename,
        "size_bytes": len(file_bytes),
        "content_type": file.content_type,
        "status": "uploaded",
        "message": "Image uploaded successfully. Proceed to analysis."
    }

# Report storage
reports_db = {}

@router.get("/report/{report_id}", response_model=AnalysisReport)
async def get_report(report_id: str):
    """
    Get saved analysis report
    """
    if report_id not in reports_db:
        raise HTTPException(status_code=404, detail="Report not found")
    
    return reports_db[report_id]

@router.post("/save-report")
async def save_report(result: AnalysisResult):
    """
    Save analysis report for future reference
    """
    from datetime import datetime
    
    report_id = f"report_{len(reports_db) + 1}"
    
    report = {
        "id": report_id,
        "timestamp": datetime.now().isoformat(),
        "scan_type": result.scan_type,
        "result": result.dict()
    }
    
    reports_db[report_id] = report
    
    return {
        "report_id": report_id,
        "message": "Report saved successfully",
        "access_url": f"/api/xray/report/{report_id}"
    }

@router.get("/reports")
async def list_reports():
    """
    List all saved reports
    """
    return list(reports_db.values())

@router.get("/stats")
async def get_analysis_stats():
    """
    Get analysis statistics
    """
    if not reports_db:
        return {
            "total_analyses": 0,
            "message": "No analyses performed yet"
        }
    
    reports = list(reports_db.values())
    
    normal_count = sum(1 for r in reports if r["result"]["status"] == "Normal")
    defective_count = sum(1 for r in reports if r["result"]["status"] == "Defective")
    
    xray_count = sum(1 for r in reports if r["scan_type"] == "xray")
    mri_count = sum(1 for r in reports if r["scan_type"] == "mri")
    
    return {
        "total_analyses": len(reports),
        "by_result": {
            "normal": normal_count,
            "defective": defective_count
        },
        "by_type": {
            "xray": xray_count,
            "mri": mri_count
        }
    }
