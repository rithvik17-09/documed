"""
PulseChain - Emergency Response System Backend
SOS alerts, vitals monitoring, emergency contacts
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import random

router = APIRouter()

# Request/Response Models
class SOSAlert(BaseModel):
    user_id: str
    location: Dict[str, float]  # {"lat": 40.7128, "lng": -74.0060}
    address: Optional[str] = None
    timestamp: Optional[datetime] = None

class VitalSigns(BaseModel):
    heart_rate: int  # bpm
    oxygen: int  # SpO2 %
    temperature: float  # Fahrenheit
    blood_pressure: Optional[Dict[str, int]] = None  # {"systolic": 120, "diastolic": 80}
    timestamp: Optional[datetime] = None

class VitalStatus(BaseModel):
    heart_rate_status: str
    oxygen_status: str
    temperature_status: str
    overall_status: str
    alerts: List[str]

class EmergencyContact(BaseModel):
    name: str
    relationship: str
    phone: str
    email: Optional[str] = None
    priority: int = 1

# In-memory storage
emergency_contacts_db = {}
vitals_history_db = []
sos_alerts_db = []

# SOS System
@router.post("/sos")
async def trigger_sos(alert: SOSAlert):
    """
    Trigger SOS emergency alert
    """
    if alert.timestamp is None:
        alert.timestamp = datetime.now()
    
    sos_alerts_db.append(alert.dict())
    
    # Simulate sending to emergency contacts
    contacts = list(emergency_contacts_db.values())
    notifications_sent = []
    
    for contact in sorted(contacts, key=lambda x: x.get("priority", 999)):
        notifications_sent.append({
            "contact_name": contact["name"],
            "contact_phone": contact["phone"],
            "method": "SMS",
            "message": f"🚨 EMERGENCY ALERT: {alert.user_id} needs help at {alert.address or 'unknown location'}. Location: {alert.location['lat']}, {alert.location['lng']}"
        })
    
    return {
        "status": "SOS activated",
        "alert_id": len(sos_alerts_db),
        "timestamp": alert.timestamp,
        "notifications_sent": len(notifications_sent),
        "contacts_notified": notifications_sent,
        "emergency_services": "Call 108 has been suggested",
        "location_shared": True
    }

@router.post("/sos/cancel")
async def cancel_sos(alert_id: int, user_id: str):
    """
    Cancel false alarm SOS
    """
    if alert_id > len(sos_alerts_db):
        raise HTTPException(status_code=404, detail="SOS alert not found")
    
    # Send cancellation to contacts
    contacts = list(emergency_contacts_db.values())
    
    return {
        "status": "SOS cancelled",
        "message": "False alarm notification sent to all contacts",
        "contacts_notified": len(contacts)
    }

@router.get("/sos/history")
async def get_sos_history():
    """
    Get SOS alert history
    """
    return sos_alerts_db

# Vitals Monitoring
@router.post("/vitals", response_model=VitalStatus)
async def submit_vitals(vitals: VitalSigns):
    """
    Submit vital signs and get health status evaluation
    """
    if vitals.timestamp is None:
        vitals.timestamp = datetime.now()
    
    vitals_history_db.append(vitals.dict())
    
    # Evaluate vitals
    alerts = []
    
    # Heart rate evaluation
    if vitals.heart_rate < 60:
        hr_status = "low"
        alerts.append("⚠️ Heart rate below normal (bradycardia). Monitor closely.")
    elif vitals.heart_rate > 100:
        hr_status = "high"
        alerts.append("⚠️ Heart rate above normal (tachycardia). Consider rest.")
    else:
        hr_status = "normal"
    
    # Oxygen evaluation
    if vitals.oxygen < 90:
        o2_status = "critical"
        alerts.append("🚨 CRITICAL: Oxygen saturation dangerously low! Seek immediate medical attention!")
    elif vitals.oxygen < 95:
        o2_status = "low"
        alerts.append("⚠️ Oxygen saturation below normal. See doctor if persistent.")
    else:
        o2_status = "normal"
    
    # Temperature evaluation
    if vitals.temperature > 100.4:
        temp_status = "fever"
        alerts.append("🌡️ Fever detected. Stay hydrated and rest.")
    elif vitals.temperature < 95:
        temp_status = "hypothermia"
        alerts.append("🥶 Body temperature too low. Warm up and seek medical care.")
    else:
        temp_status = "normal"
    
    # Overall status
    if o2_status == "critical" or temp_status == "hypothermia":
        overall = "critical"
    elif hr_status == "high" or hr_status == "low" or o2_status == "low" or temp_status == "fever":
        overall = "concerning"
    else:
        overall = "healthy"
    
    return VitalStatus(
        heart_rate_status=hr_status,
        oxygen_status=o2_status,
        temperature_status=temp_status,
        overall_status=overall,
        alerts=alerts if alerts else ["All vitals within normal range ✅"]
    )

@router.get("/vitals")
async def get_vitals_history(limit: int = 10):
    """
    Get vitals history
    """
    return vitals_history_db[-limit:]

@router.get("/vitals/latest")
async def get_latest_vitals():
    """
    Get most recent vital signs
    """
    if not vitals_history_db:
        raise HTTPException(status_code=404, detail="No vitals recorded yet")
    
    return vitals_history_db[-1]

@router.get("/vitals/trends")
async def get_vitals_trends():
    """
    Get trends in vital signs over time
    """
    if len(vitals_history_db) < 2:
        return {"message": "Not enough data for trend analysis"}
    
    recent = vitals_history_db[-7:]  # Last 7 readings
    
    avg_hr = sum(v["heart_rate"] for v in recent) / len(recent)
    avg_o2 = sum(v["oxygen"] for v in recent) / len(recent)
    avg_temp = sum(v["temperature"] for v in recent) / len(recent)
    
    return {
        "averages": {
            "heart_rate": round(avg_hr, 1),
            "oxygen": round(avg_o2, 1),
            "temperature": round(avg_temp, 1)
        },
        "readings_analyzed": len(recent),
        "trend": "stable" if 60 <= avg_hr <= 100 and avg_o2 >= 95 else "needs_monitoring"
    }

# Emergency Contacts Management
@router.post("/contacts")
async def add_emergency_contact(contact: EmergencyContact):
    """
    Add emergency contact
    """
    contact_id = f"contact_{len(emergency_contacts_db) + 1}"
    emergency_contacts_db[contact_id] = contact.dict()
    
    return {
        "id": contact_id,
        "message": f"Emergency contact {contact.name} added successfully",
        "contact": contact.dict()
    }

@router.get("/contacts")
async def get_emergency_contacts():
    """
    Get all emergency contacts
    """
    contacts = list(emergency_contacts_db.values())
    return sorted(contacts, key=lambda x: x["priority"])

@router.delete("/contacts/{contact_id}")
async def remove_emergency_contact(contact_id: str):
    """
    Remove emergency contact
    """
    if contact_id not in emergency_contacts_db:
        raise HTTPException(status_code=404, detail="Contact not found")
    
    removed = emergency_contacts_db.pop(contact_id)
    
    return {
        "message": f"Contact {removed['name']} removed successfully"
    }

@router.put("/contacts/{contact_id}")
async def update_emergency_contact(contact_id: str, contact: EmergencyContact):
    """
    Update emergency contact
    """
    if contact_id not in emergency_contacts_db:
        raise HTTPException(status_code=404, detail="Contact not found")
    
    emergency_contacts_db[contact_id] = contact.dict()
    
    return {
        "message": "Contact updated successfully",
        "contact": contact.dict()
    }

# Alerts Feed
@router.get("/alerts")
async def get_alerts_feed():
    """
    Get all health alerts and notifications
    """
    alerts = []
    
    # Check recent vitals for alerts
    if vitals_history_db:
        latest = vitals_history_db[-1]
        if latest["oxygen"] < 95:
            alerts.append({
                "type": "vital-alert",
                "severity": "high",
                "message": f"Low oxygen detected: {latest['oxygen']}%",
                "timestamp": latest["timestamp"],
                "action": "Check oxygen levels again"
            })
    
    # SOS alerts
    for sos in sos_alerts_db[-5:]:
        alerts.append({
            "type": "sos",
            "severity": "critical",
            "message": "SOS alert was triggered",
            "timestamp": sos["timestamp"],
            "location": sos["address"]
        })
    
    return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)

# Location Tracking (for SOS)
@router.post("/location/update")
async def update_location(user_id: str, location: Dict[str, float]):
    """
    Update user location for emergency tracking
    """
    return {
        "user_id": user_id,
        "location": location,
        "last_updated": datetime.now().isoformat(),
        "tracking_active": True
    }
