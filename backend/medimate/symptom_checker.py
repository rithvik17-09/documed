"""
MediMate - Medical Assistant Backend
Symptom checker, medicine reminders, and appointment booking
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import random

router = APIRouter()

# Medical knowledge base for symptom checking
SYMPTOM_DATABASE = {
    "fever": ["Common Cold", "Flu", "COVID-19", "Infection"],
    "cough": ["Common Cold", "Bronchitis", "Pneumonia", "Asthma"],
    "headache": ["Tension Headache", "Migraine", "Dehydration", "Stress"],
    "sore throat": ["Common Cold", "Strep Throat", "Tonsillitis"],
    "fatigue": ["Anemia", "Depression", "Sleep Disorders", "Chronic Fatigue"],
    "nausea": ["Gastroenteritis", "Food Poisoning", "Migraine", "Pregnancy"],
    "chest pain": ["Heart Attack", "Angina", "Acid Reflux", "Panic Attack"],
    "shortness of breath": ["Asthma", "Pneumonia", "Anxiety", "Heart Failure"],
    "dizziness": ["Vertigo", "Low Blood Pressure", "Dehydration", "Inner Ear Problems"]
}

# Request/Response Models
class SymptomRequest(BaseModel):
    symptoms: List[str]
    duration: Optional[str] = None
    severity: Optional[int] = 1  # 1-10 scale

class SymptomResponse(BaseModel):
    possible_conditions: List[Dict]
    risk_level: str
    recommendation: str
    should_see_doctor: bool

class Medication(BaseModel):
    name: str
    dosage: str
    frequency: str  # "daily", "twice daily", "every 6 hours"
    time: str  # "08:00", "14:00", "20:00"
    duration_days: int
    notes: Optional[str] = None

class MedicationReminder(BaseModel):
    medication: Medication
    next_dose: datetime
    doses_remaining: int

class Appointment(BaseModel):
    doctor_name: str
    specialty: str
    date: str
    time: str
    reason: str
    location: Optional[str] = None

# Symptom Checker
@router.post("/symptoms", response_model=SymptomResponse)
async def check_symptoms(request: SymptomRequest):
    """
    Analyze symptoms and provide possible conditions
    """
    possible_conditions = {}
    
    # Match symptoms to conditions
    for symptom in request.symptoms:
        symptom_lower = symptom.lower()
        for key, conditions in SYMPTOM_DATABASE.items():
            if key in symptom_lower or symptom_lower in key:
                for condition in conditions:
                    if condition in possible_conditions:
                        possible_conditions[condition] += 1
                    else:
                        possible_conditions[condition] = 1
    
    # Sort by frequency
    sorted_conditions = sorted(
        possible_conditions.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Create response
    conditions_list = [
        {
            "name": condition,
            "match_score": score,
            "likelihood": "high" if score >= 3 else "medium" if score >= 2 else "low"
        }
        for condition, score in sorted_conditions[:5]
    ]
    
    # Assess risk level
    critical_symptoms = ["chest pain", "shortness of breath", "severe headache"]
    risk_level = "high" if any(s in " ".join(request.symptoms).lower() for s in critical_symptoms) else "medium" if request.severity and request.severity >= 7 else "low"
    
    should_see_doctor = risk_level == "high" or (request.severity and request.severity >= 7)
    
    recommendation = {
        "high": "Seek immediate medical attention. Call 108 or visit emergency room.",
        "medium": "Schedule appointment with doctor within 24-48 hours.",
        "low": "Monitor symptoms. See doctor if they worsen or persist beyond 5 days."
    }[risk_level]
    
    return SymptomResponse(
        possible_conditions=conditions_list,
        risk_level=risk_level,
        recommendation=recommendation,
        should_see_doctor=should_see_doctor
    )

# Medicine Reminder System
medications_db = {}  # In-memory storage for demo

@router.post("/medications")
async def add_medication(medication: Medication):
    """
    Add medication to reminder system
    """
    med_id = f"med_{len(medications_db) + 1}"
    medications_db[med_id] = medication.dict()
    
    return {
        "id": med_id,
        "message": f"Medication {medication.name} added successfully",
        "next_reminder": medication.time
    }

@router.get("/medications")
async def get_medications():
    """
    Get all medications
    """
    return list(medications_db.values())

@router.get("/reminders")
async def get_reminders():
    """
    Get upcoming medication reminders
    """
    reminders = []
    now = datetime.now()
    
    for med_id, med_data in medications_db.items():
        # Calculate next dose time
        med = Medication(**med_data)
        next_dose = now + timedelta(hours=random.randint(1, 8))
        
        reminders.append({
            "medication_name": med.name,
            "dosage": med.dosage,
            "next_dose": next_dose.isoformat(),
            "frequency": med.frequency
        })
    
    return reminders

# Appointment Booking
appointments_db = {}

@router.post("/appointments")
async def book_appointment(appointment: Appointment):
    """
    Book medical appointment
    """
    appt_id = f"appt_{len(appointments_db) + 1}"
    appointments_db[appt_id] = appointment.dict()
    
    return {
        "id": appt_id,
        "message": f"Appointment booked with Dr. {appointment.doctor_name}",
        "appointment": appointment.dict(),
        "confirmation": f"Confirmed for {appointment.date} at {appointment.time}"
    }

@router.get("/appointments")
async def get_appointments():
    """
    Get all appointments
    """
    return list(appointments_db.values())

@router.get("/appointments/upcoming")
async def get_upcoming_appointments():
    """
    Get upcoming appointments (next 7 days)
    """
    # For demo, return all appointments
    return list(appointments_db.values())

# Drug Interaction Checker
@router.post("/interactions")
async def check_drug_interactions(medication_names: List[str]):
    """
    Check for potential drug interactions
    """
    # Simplified interaction checking
    common_interactions = {
        ("warfarin", "aspirin"): "Increased bleeding risk",
        ("lisinopril", "ibuprofen"): "May reduce effectiveness of blood pressure medication",
        ("metformin", "alcohol"): "Increased risk of lactic acidosis"
    }
    
    interactions_found = []
    
    for i, med1 in enumerate(medication_names):
        for med2 in medication_names[i+1:]:
            key = tuple(sorted([med1.lower(), med2.lower()]))
            if key in common_interactions:
                interactions_found.append({
                    "drugs": [med1, med2],
                    "warning": common_interactions[key],
                    "severity": "moderate"
                })
    
    return {
        "has_interactions": len(interactions_found) > 0,
        "interactions": interactions_found,
        "recommendation": "Consult with pharmacist or doctor about these medications" if interactions_found else "No known interactions detected"
    }
