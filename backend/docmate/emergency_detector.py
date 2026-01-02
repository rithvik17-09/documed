"""
DocMate - First Aid Assistant Backend
Emergency detection and first-aid guidance system
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import re

router = APIRouter()
EMERGENCY_KNOWLEDGE_BASE = {
    "CPR": {
        "keywords": ["cpr", "not breathing", "stopped breathing", "unconscious", "cardiac arrest", "heart stopped"],
        "severity": "critical",
        "steps": [
            "Call 108 or emergency services immediately",
            "Check if person is responsive - tap shoulders and shout",
            "If no response, check breathing for 10 seconds",
            "Place person on firm, flat surface",
            "Position yourself beside the person's chest",
            "Place heel of one hand on center of chest, other hand on top",
            "Push hard and fast - compress chest 2 inches deep",
            "Give 30 chest compressions at rate of 100-120/minute",
            "Give 2 rescue breaths (tilt head, lift chin, pinch nose)",
            "Continue 30 compressions : 2 breaths cycle until help arrives"
        ],
        "warnings": ["Do not stop CPR unless person starts breathing or help arrives"],
        "call_emergency": True
    },
    "Choking": {
        "keywords": ["choking", "can't breathe", "food stuck", "airway blocked"],
        "severity": "critical",
        "steps": [
            "Ask 'Are you choking?' - if they can't speak, act immediately",
            "Call for help - have someone call 108",
            "Stand behind the person",
            "Make a fist with one hand, place above navel",
            "Grasp fist with other hand",
            "Give quick, upward thrusts (Heimlich maneuver)",
            "Repeat until object is expelled",
            "If person becomes unconscious, begin CPR"
        ],
        "warnings": ["For infants, use back blows and chest thrusts, NOT Heimlich"],
        "call_emergency": True
    },
    "Burns": {
        "keywords": ["burn", "burned", "scalded", "fire", "hot water", "chemical burn"],
        "severity": "high",
        "steps": [
            "Remove person from source of burn immediately",
            "Cool the burn with cool (not cold) running water for 10-20 minutes",
            "Remove jewelry and tight clothing before swelling starts",
            "Cover burn with clean, non-stick bandage or cloth",
            "Do NOT apply ice, butter, or ointments",
            "Take over-the-counter pain reliever if needed",
            "For large burns (bigger than 3 inches), seek medical care",
            "For chemical burns, flush with water for at least 20 minutes"
        ],
        "warnings": ["Never pop blisters", "Seek immediate care for face/hand burns"],
        "call_emergency": False
    },
    "Bleeding": {
        "keywords": ["bleeding", "blood", "cut", "wound", "laceration", "hemorrhage"],
        "severity": "high",
        "steps": [
            "Wash your hands if possible",
            "Apply direct pressure with clean cloth or bandage",
            "Maintain pressure for 15-20 minutes without checking",
            "If blood soaks through, add more cloth on top (don't remove first layer)",
            "Elevate injured area above heart if possible",
            "Once bleeding stops, apply clean bandage",
            "For severe bleeding that won't stop, call 108",
            "Check for signs of shock (pale, weak, rapid breathing)"
        ],
        "warnings": ["If object is embedded in wound, don't remove it"],
        "call_emergency": False
    },
    "Fracture": {
        "keywords": ["fracture", "broken bone", "bone break", "can't move", "deformed limb"],
        "severity": "medium",
        "steps": [
            "Do not move the injured area",
            "Immobilize the injured area with splint or padding",
            "Apply ice pack wrapped in cloth (not directly on skin)",
            "Keep ice on for 15-20 minutes at a time",
            "Elevate injured area if possible",
            "Give over-the-counter pain reliever",
            "Seek medical attention",
            "For open fracture (bone visible), cover wound with clean cloth"
        ],
        "warnings": ["Never try to realign the bone", "Don't test if bone is broken by moving it"],
        "call_emergency": False
    },
    "Heart Attack": {
        "keywords": ["heart attack", "chest pain", "chest pressure", "left arm pain", "jaw pain", "shortness of breath"],
        "severity": "critical",
        "steps": [
            "Call 108 immediately",
            "Have person sit down and rest",
            "Loosen any tight clothing",
            "Give aspirin (if no allergies) - have person chew it slowly",
            "Stay with person and keep them calm",
            "If person becomes unconscious and stops breathing, begin CPR",
            "Do not leave person alone",
            "Note time symptoms started for emergency responders"
        ],
        "warnings": ["Never delay calling emergency services"],
        "call_emergency": True
    },
    "Stroke": {
        "keywords": ["stroke", "facial drooping", "arm weakness", "speech difficulty", "FAST"],
        "severity": "critical",
        "steps": [
            "Call 108 immediately",
            "Note time symptoms started - critical for treatment",
            "FAST test: Face drooping, Arm weakness, Speech difficulty, Time to call",
            "Keep person calm and comfortable",
            "Do not give food or water",
            "Lay person on side if vomiting",
            "Stay with person until help arrives"
        ],
        "warnings": ["Every minute counts - call emergency immediately"],
        "call_emergency": True
    },
    "Allergic Reaction": {
        "keywords": ["allergic reaction", "anaphylaxis", "swelling", "hives", "difficulty breathing", "throat swelling"],
        "severity": "critical",
        "steps": [
            "Check if person has epinephrine auto-injector (EpiPen)",
            "If yes and severe reaction, use it immediately",
            "Call 108",
            "Help person lie down with legs elevated",
            "Loosen tight clothing",
            "Do not give oral medications if person has trouble breathing",
            "Be prepared to perform CPR if needed",
            "Stay with person until help arrives"
        ],
        "warnings": ["Use EpiPen even if unsure - benefits outweigh risks"],
        "call_emergency": True
    },
    "Seizure": {
        "keywords": ["seizure", "convulsion", "shaking", "epilepsy", "fitting"],
        "severity": "high",
        "steps": [
            "Stay calm and time the seizure",
            "Protect person from injury - move harmful objects away",
            "Cushion head with something soft",
            "Turn person on their side if possible",
            "Do NOT hold person down or restrain them",
            "Do NOT put anything in their mouth",
            "Stay with person until fully conscious",
            "Call 108 if seizure lasts more than 5 minutes or person is injured"
        ],
        "warnings": ["Never restrain person during seizure", "Most seizures stop on their own"],
        "call_emergency": False
    },
    "Poisoning": {
        "keywords": ["poisoning", "swallowed", "overdose", "toxic", "poison"],
        "severity": "critical",
        "steps": [
            "Call Poison Control (1-800-222-1222) or 108",
            "Identify substance if possible",
            "Do NOT induce vomiting unless told to by professional",
            "If person is unconscious, not breathing, or having seizures, call 108",
            "Save container or substance for identification",
            "Follow instructions from poison control exactly",
            "Keep person calm and still"
        ],
        "warnings": ["Never give anything by mouth unless instructed"],
        "call_emergency": True
    },
    "Heatstroke": {
        "keywords": ["heat stroke", "heat exhaustion", "overheating", "hot", "dizzy from heat"],
        "severity": "critical",
        "steps": [
            "Move person to cool, shaded area immediately",
            "Call 108 if confusion, unconsciousness, or very high temperature",
            "Remove excess clothing",
            "Cool person rapidly - apply cool wet cloths",
            "Fan person while applying cool water",
            "Give cool water to drink if conscious and able",
            "Apply ice packs to armpits, groin, neck, and back",
            "Continue cooling until help arrives"
        ],
        "warnings": ["Heatstroke is life-threatening - act fast"],
        "call_emergency": True
    },
    "Hypothermia": {
        "keywords": ["hypothermia", "too cold", "freezing", "shivering", "frostbite"],
        "severity": "high",
        "steps": [
            "Move person to warm, dry area",
            "Remove any wet clothing",
            "Warm person gradually with blankets",
            "Cover head and neck (heat loss areas)",
            "Give warm (not hot) beverages if conscious",
            "Do not use direct heat (heating pad, hot water)",
            "Do not rub or massage person",
            "Seek medical attention"
        ],
        "warnings": ["Warm gradually - rapid warming can cause heart problems"],
        "call_emergency": False
    },
    "Nosebleed": {
        "keywords": ["nosebleed", "nose bleeding", "bloody nose"],
        "severity": "low",
        "steps": [
            "Sit upright and lean slightly forward",
            "Do NOT tilt head back (can cause blood to go down throat)",
            "Pinch soft part of nose firmly for 10 minutes",
            "Breathe through mouth while pinching",
            "After 10 minutes, release and check if bleeding stopped",
            "If still bleeding, pinch for another 10 minutes",
            "Apply cold compress to nose bridge",
            "Avoid blowing nose for several hours after"
        ],
        "warnings": ["If bleeding continues after 20 minutes, seek medical care"],
        "call_emergency": False
    },
    "Sprain": {
        "keywords": ["sprain", "twisted ankle", "twisted wrist", "swollen joint"],
        "severity": "low",
        "steps": [
            "Follow RICE method:",
            "Rest - Stop activity, don't put weight on injury",
            "Ice - Apply ice pack for 15-20 minutes every 2-3 hours",
            "Compression - Wrap with elastic bandage (not too tight)",
            "Elevation - Raise injured area above heart level",
            "Take over-the-counter pain reliever",
            "Seek medical care if severe pain or can't bear weight"
        ],
        "warnings": ["If unable to move joint or severe deformity, may be fracture - seek care"],
        "call_emergency": False
    }
}
class EmergencyRequest(BaseModel):
    description: str
    urgency: Optional[str] = None

class FirstAidGuide(BaseModel):
    emergency_type: str
    severity: str
    steps: List[str]
    warnings: List[str]
    call_emergency: bool

class EmergencyResponse(BaseModel):
    emergency_type: str
    severity: str
    confidence: float
    steps: List[str]
    warnings: List[str]
    call_emergency: bool
    additional_info: Optional[str] = None
def detect_emergency(description: str) -> Dict:
    """
    Detect emergency type from user description using keyword matching
    """
    description_lower = description.lower()
    matches = []
    
    for emergency_type, data in EMERGENCY_KNOWLEDGE_BASE.items():
        match_score = 0
        for keyword in data["keywords"]:
            if keyword in description_lower:
                match_score += 1
        
        if match_score > 0:
            matches.append({
                "type": emergency_type,
                "score": match_score,
                "data": data
            })
    if matches:
        matches.sort(key=lambda x: x["score"], reverse=True)
        best_match = matches[0]
        return {
            "emergency_type": best_match["type"],
            "confidence": min(best_match["score"] * 0.3, 0.95),  # Cap at 95%
            **best_match["data"]
        }
    
    return None
@router.post("/emergency", response_model=EmergencyResponse)
async def analyze_emergency(request: EmergencyRequest):
    """
    Analyze emergency situation and provide first-aid guidance
    """
    result = detect_emergency(request.description)
    
    if not result:
        raise HTTPException(
            status_code=404,
            detail="Could not identify specific emergency type. Please call emergency services if unsure."
        )
    
    return EmergencyResponse(
        emergency_type=result["emergency_type"],
        severity=result["severity"],
        confidence=result["confidence"],
        steps=result["steps"],
        warnings=result["warnings"],
        call_emergency=result["call_emergency"],
        additional_info="This is AI-generated guidance. For serious emergencies, call 108 immediately."
    )

@router.get("/guides", response_model=List[FirstAidGuide])
async def get_all_guides():
    """
    Get all available first-aid guides
    """
    guides = []
    for emergency_type, data in EMERGENCY_KNOWLEDGE_BASE.items():
        guides.append(FirstAidGuide(
            emergency_type=emergency_type,
            severity=data["severity"],
            steps=data["steps"],
            warnings=data["warnings"],
            call_emergency=data["call_emergency"]
        ))
    return guides

@router.get("/guides/{emergency_type}", response_model=FirstAidGuide)
async def get_guide(emergency_type: str):
    """
    Get specific first-aid guide by emergency type
    """
    data = EMERGENCY_KNOWLEDGE_BASE.get(emergency_type)
    if not data:
        raise HTTPException(status_code=404, detail="Emergency type not found")
    
    return FirstAidGuide(
        emergency_type=emergency_type,
        severity=data["severity"],
        steps=data["steps"],
        warnings=data["warnings"],
        call_emergency=data["call_emergency"]
    )

@router.post("/ask")
async def ask_question(request: EmergencyRequest):
    """
    Ask a first-aid question using AI (Gemini fallback for complex cases)
    """
    result = detect_emergency(request.description)
    
    if result and result["confidence"] > 0.6:
        return {
            "source": "knowledge_base",
            "emergency_type": result["emergency_type"],
            "response": result["steps"],
            "confidence": result["confidence"]
        }
    
    return {
        "source": "general",
        "response": [
            "For medical emergencies, always call 108 or your local emergency number",
            "If someone is unconscious or not breathing, begin CPR immediately",
            "For serious injuries, bleeding, or severe pain, seek immediate medical attention",
            "Keep first-aid kit stocked and accessible",
            "Learn CPR and basic first aid through certified courses"
        ],
        "note": "This is general guidance. For specific emergencies, please call emergency services."
    }
