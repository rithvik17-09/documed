# Documed Backend - API Quick Reference

## Base URL
```
http://localhost:8000
```

## System Endpoints

### Health Check
```bash
GET /health
```

### API Info
```bash
GET /api/info
```

---

## DocMate - First Aid Assistant

### Analyze Emergency
```bash
POST /api/docmate/emergency
Content-Type: application/json

{
  "description": "person not breathing",
  "urgency": "high"
}
```

### Get All Guides
```bash
GET /api/docmate/guides
```

### Get Specific Guide
```bash
GET /api/docmate/guides/CPR
```

### Ask Question
```bash
POST /api/docmate/ask
Content-Type: application/json

{
  "description": "how to treat a burn"
}
```

---

## MediMate - Medical Assistant

### Check Symptoms
```bash
POST /api/medimate/symptoms
Content-Type: application/json

{
  "symptoms": ["fever", "cough", "headache"],
  "duration": "3 days",
  "severity": 6
}
```

### Add Medication
```bash
POST /api/medimate/medications
Content-Type: application/json

{
  "name": "Ibuprofen",
  "dosage": "400mg",
  "frequency": "twice daily",
  "time": "08:00",
  "duration_days": 7
}
```

### Get Reminders
```bash
GET /api/medimate/reminders
```

### Book Appointment
```bash
POST /api/medimate/appointments
Content-Type: application/json

{
  "doctor_name": "Dr. Smith",
  "specialty": "Cardiology",
  "date": "2024-02-15",
  "time": "14:00",
  "reason": "Annual checkup"
}
```

### Check Drug Interactions
```bash
POST /api/medimate/interactions
Content-Type: application/json

["Warfarin", "Aspirin", "Ibuprofen"]
```

---

## MediMood - Mental Wellness

### Record Mood
```bash
POST /api/medimood/mood
Content-Type: application/json

{
  "mood": "Happy",
  "intensity": 8,
  "notes": "Had a great day!"
}
```

### Get Mood History
```bash
GET /api/medimood/mood/history?days=7
```

### Get Mood Analysis
```bash
GET /api/medimood/analysis
```

### Create Journal Entry
```bash
POST /api/medimood/journal
Content-Type: application/json

{
  "mood": "Calm",
  "content": "Today I practiced mindfulness...",
  "tags": ["meditation", "self-care"]
}
```

### Get Content Suggestions
```bash
GET /api/medimood/suggestions?mood=Anxious
```

### Crisis Check
```bash
POST /api/medimood/crisis-check
Content-Type: application/json

{
  "text": "user input text to check"
}
```

---

## PulseChain - Emergency Response

### Trigger SOS
```bash
POST /api/pulsechain/sos
Content-Type: application/json

{
  "user_id": "user123",
  "location": {"lat": 40.7128, "lng": -74.0060},
  "address": "123 Main St, New York"
}
```

### Submit Vitals
```bash
POST /api/pulsechain/vitals
Content-Type: application/json

{
  "heart_rate": 75,
  "oxygen": 98,
  "temperature": 98.6,
  "blood_pressure": {"systolic": 120, "diastolic": 80}
}
```

### Get Vitals History
```bash
GET /api/pulsechain/vitals?limit=10
```

### Get Latest Vitals
```bash
GET /api/pulsechain/vitals/latest
```

### Get Vitals Trends
```bash
GET /api/pulsechain/vitals/trends
```

### Add Emergency Contact
```bash
POST /api/pulsechain/contacts
Content-Type: application/json

{
  "name": "John Doe",
  "relationship": "Spouse",
  "phone": "+1234567890",
  "email": "john@example.com",
  "priority": 1
}
```

### Get Emergency Contacts
```bash
GET /api/pulsechain/contacts
```

### Get Alerts Feed
```bash
GET /api/pulsechain/alerts
```

---

## X-Ray Analyser - Medical Imaging

### Analyze Image
```bash
POST /api/xray/analyze
Content-Type: multipart/form-data

file: [image file]
scan_type: xray  # or "mri"
```

### Upload Image
```bash
POST /api/xray/upload
Content-Type: multipart/form-data

file: [image file]
```

### Get Report
```bash
GET /api/xray/report/{report_id}
```

### Save Report
```bash
POST /api/xray/save-report
Content-Type: application/json

{
  "status": "Defective",
  "confidence": 94.3,
  "findings": [...],
  "recommendation": "Consult radiologist",
  "scan_type": "xray"
}
```

### List Reports
```bash
GET /api/xray/reports
```

### Get Statistics
```bash
GET /api/xray/stats
```

---

## Example Usage with cURL

### Check Symptoms
```bash
curl -X POST http://localhost:8000/api/medimate/symptoms \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["fever", "cough"], "severity": 5}'
```

### Trigger SOS
```bash
curl -X POST http://localhost:8000/api/pulsechain/sos \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "location": {"lat": 40.7128, "lng": -74.0060}}'
```

### Analyze X-Ray
```bash
curl -X POST http://localhost:8000/api/xray/analyze \
  -F "file=@xray_image.jpg" \
  -F "scan_type=xray"
```

---

## Response Format

### Success Response
```json
{
  "status": "success",
  "data": {...},
  "message": "Operation successful"
}
```

### Error Response
```json
{
  "error": "Error description",
  "status_code": 400
}
```

---

## Interactive Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive API documentation where you can test endpoints directly from your browser.
