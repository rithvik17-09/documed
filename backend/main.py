"""
Documed Backend - Main FastAPI Application
Complete AI-powered healthcare management platform
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from typing import Optional
import os
from dotenv import load_dotenv

# Import all module routers
from docmate.emergency_detector import router as docmate_router
from medimate.symptom_checker import router as medimate_router
from medimood.mood_analyzer import router as medimood_router
from pulsechain.sos_handler import router as pulsechain_router
from xray_analyser.image_processor import router as xray_router

# Load environment variables
load_dotenv()

# Application metadata
APP_METADATA = {
    "title": "Documed Backend API",
    "description": "Complete AI-powered healthcare management platform with ML models for medical image analysis, symptom checking, mental wellness tracking, and emergency response",
    "version": "1.0.0",
    "contact": {
        "name": "Documed Team",
        "url": "https://github.com/documed",
        "email": "support@documed.ai"
    },
    "license_info": {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown events
    """
    print("🚀 Starting Documed Backend Server...")
    print("📊 Loading ML models...")
    
    # Initialize ML models on startup
    try:
        from xray_analyser.cnn_inference import load_models
        await load_models()
        print("✅ ML models loaded successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not load ML models: {e}")
    
    print("✅ Server ready!")
    print("📖 API Docs: http://localhost:8000/docs")
    
    yield
    
    print("🛑 Shutting down Documed Backend Server...")

# Initialize FastAPI app
app = FastAPI(
    **APP_METADATA,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include module routers
app.include_router(docmate_router, prefix="/api/docmate", tags=["DocMate - First Aid"])
app.include_router(medimate_router, prefix="/api/medimate", tags=["MediMate - Medical Assistant"])
app.include_router(medimood_router, prefix="/api/medimood", tags=["MediMood - Mental Wellness"])
app.include_router(pulsechain_router, prefix="/api/pulsechain", tags=["PulseChain - Emergency Response"])
app.include_router(xray_router, prefix="/api/xray", tags=["X-Ray Analyser - Medical Imaging"])

# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """
    Welcome endpoint with system information
    """
    return {
        "message": "Welcome to Documed Backend API",
        "version": "1.0.0",
        "status": "operational",
        "modules": {
            "docmate": "First Aid Assistant",
            "medimate": "Medical Assistant",
            "medimood": "Mental Wellness Platform",
            "pulsechain": "Emergency Response System",
            "xray_analyser": "Medical Image Analysis"
        },
        "documentation": "/docs",
        "health_check": "/health"
    }

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint for monitoring
    """
    return {
        "status": "healthy",
        "services": {
            "api": "operational",
            "database": "operational",
            "ml_models": "operational"
        }
    }

# API information endpoint
@app.get("/api/info", tags=["System"])
async def api_info():
    """
    Get detailed API information
    """
    return {
        "name": "Documed Backend",
        "version": "1.0.0",
        "description": "AI-powered healthcare management platform",
        "endpoints": {
            "docmate": {
                "base": "/api/docmate",
                "description": "First aid assistance and emergency guidance",
                "endpoints": [
                    "POST /emergency - Analyze emergency situation",
                    "GET /guides - Get first-aid guides",
                    "POST /ask - Ask first-aid question"
                ]
            },
            "medimate": {
                "base": "/api/medimate",
                "description": "Medical assistance including symptom checker, medicine reminders, appointments",
                "endpoints": [
                    "POST /symptoms - Analyze symptoms",
                    "POST /medications - Manage medications",
                    "POST /appointments - Book appointment"
                ]
            },
            "medimood": {
                "base": "/api/medimood",
                "description": "Mental wellness tracking and mood analysis",
                "endpoints": [
                    "POST /mood - Record mood",
                    "GET /analysis - Get mood insights",
                    "POST /journal - Create journal entry"
                ]
            },
            "pulsechain": {
                "base": "/api/pulsechain",
                "description": "Emergency response and health monitoring",
                "endpoints": [
                    "POST /sos - Trigger SOS alert",
                    "POST /vitals - Submit vital signs",
                    "GET /vitals - Get vitals history"
                ]
            },
            "xray": {
                "base": "/api/xray",
                "description": "AI-powered medical image analysis",
                "endpoints": [
                    "POST /analyze - Analyze medical image",
                    "GET /report/:id - Get analysis report"
                ]
            }
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
