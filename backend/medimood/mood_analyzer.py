"""
MediMood - Mental Wellness Platform Backend
Mood analysis, sentiment analysis, and content recommendations
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import random

router = APIRouter()

MOODS = ["Happy", "Calm", "Neutral", "Sad", "Angry", "Anxious", "Tired", "Grateful", "Excited", "Overwhelmed"]
class MoodEntry(BaseModel):
    mood: str
    intensity: int  
    notes: Optional[str] = None
    timestamp: Optional[datetime] = None

class JournalEntry(BaseModel):
    mood: str
    content: str
    tags: Optional[List[str]] = None
    timestamp: Optional[datetime] = None

class MoodAnalysis(BaseModel):
    dominant_mood: str
    mood_variety: int
    trend: str  
    insights: List[str]
    concerns: Optional[str] = None

class ContentSuggestion(BaseModel):
    type: str  
    title: str
    description: str
    duration: Optional[str] = None

mood_entries_db = []
journal_entries_db = []
@router.post("/mood")
async def record_mood(entry: MoodEntry):
    """
    Record daily mood entry
    """
    if entry.mood not in MOODS:
        raise HTTPException(status_code=400, detail=f"Invalid mood. Choose from: {', '.join(MOODS)}")
    
    if entry.timestamp is None:
        entry.timestamp = datetime.now()
    
    mood_entries_db.append(entry.dict())
    
    return {
        "message": "Mood recorded successfully",
        "entry": entry.dict(),
        "streak": calculate_streak()
    }
@router.get("/mood/history")
async def get_mood_history(days: int = 7):
    """
    Get mood history for specified number of days
    """
    cutoff_date = datetime.now() - timedelta(days=days)
    recent_entries = [
        entry for entry in mood_entries_db
        if datetime.fromisoformat(str(entry["timestamp"])) > cutoff_date
    ]
    
    return recent_entries
@router.get("/analysis", response_model=MoodAnalysis)
async def analyze_mood_patterns():
    """
    Analyze mood patterns and generate insights
    """
    if not mood_entries_db:
        raise HTTPException(status_code=404, detail="No mood entries found. Start tracking your mood first!")
    
    recent = mood_entries_db[-7:]
    mood_counts = {}
    for entry in recent:
        mood = entry["mood"]
        mood_counts[mood] = mood_counts.get(mood, 0) + 1
    dominant_mood = max(mood_counts, key=mood_counts.get)
    mood_variety = len(mood_counts)
    if len(recent) >= 3:
        recent_moods = [entry["mood"] for entry in recent[-3:]]
        positive_moods = ["Happy", "Calm", "Grateful", "Excited"]
        positive_count = sum(1 for m in recent_moods if m in positive_moods)
        
        if positive_count >= 2:
            trend = "improving"
        elif positive_count == 1:
            trend = "stable"
        else:
            trend = "declining"
    else:
        trend = "stable"
    insights = []
    if dominant_mood in ["Happy", "Grateful", "Excited"]:
        insights.append(f"You've been feeling {dominant_mood.lower()} most often - that's wonderful!")
    elif dominant_mood in ["Sad", "Anxious", "Overwhelmed"]:
        insights.append(f"You've been feeling {dominant_mood.lower()} frequently. Consider self-care activities.")
    
    if mood_variety > 5:
        insights.append("You're experiencing a wide range of emotions - that's normal and healthy.")
    
    if trend == "declining":
        insights.append("Your mood has been declining lately. Consider reaching out to someone you trust.")
    
    concerns = None
    negative_streak = 0
    for entry in reversed(recent):
        if entry["mood"] in ["Sad", "Anxious", "Overwhelmed"]:
            negative_streak += 1
        else:
            break
    
    if negative_streak >= 5:
        concerns = "You've been feeling down for several days. Consider speaking with a mental health professional."
    
    return MoodAnalysis(
        dominant_mood=dominant_mood,
        mood_variety=mood_variety,
        trend=trend,
        insights=insights,
        concerns=concerns
    )

@router.post("/journal")
async def create_journal_entry(entry: JournalEntry):
    """
    Create journal entry with mood
    """
    if entry.timestamp is None:
        entry.timestamp = datetime.now()
    positive_words = ["happy", "joy", "good", "great", "wonderful", "love", "grateful"]
    negative_words = ["sad", "angry", "bad", "terrible", "hate", "awful", "worried"]
    
    content_lower = entry.content.lower()
    positive_count = sum(1 for word in positive_words if word in content_lower)
    negative_count = sum(1 for word in negative_words if word in content_lower)
    
    sentiment = "positive" if positive_count > negative_count else "negative" if negative_count > positive_count else "neutral"
    
    journal_entries_db.append(entry.dict())
    
    return {
        "message": "Journal entry created",
        "sentiment_detected": sentiment,
        "word_count": len(entry.content.split()),
        "entry_id": len(journal_entries_db)
    }

@router.get("/journal")
async def get_journal_entries(limit: int = 10):
    """
    Get recent journal entries
    """
    return journal_entries_db[-limit:]

@router.get("/suggestions", response_model=List[ContentSuggestion])
async def get_content_suggestions(mood: Optional[str] = None):
    """
    Get personalized wellness content based on mood
    """
    suggestions = []
    if not mood and mood_entries_db:
        mood = mood_entries_db[-1]["mood"]
    
    if not mood:
        mood = "Neutral"
    if mood in ["Anxious", "Overwhelmed"]:
        suggestions.extend([
            ContentSuggestion(
                type="breathing",
                title="4-7-8 Breathing Technique",
                description="Calm anxiety with this simple breathing exercise",
                duration="5 minutes"
            ),
            ContentSuggestion(
                type="meditation",
                title="Grounding Meditation",
                description="Bring yourself back to the present moment",
                duration="10 minutes"
            ),
            ContentSuggestion(
                type="activity",
                title="Take a Walk",
                description="Fresh air and movement can reduce anxiety",
                duration="15 minutes"
            )
        ])
    elif mood in ["Sad"]:
        suggestions.extend([
            ContentSuggestion(
                type="article",
                title="Understanding Sadness",
                description="Learn why sadness is a natural emotion and how to process it",
                duration="8 minutes"
            ),
            ContentSuggestion(
                type="activity",
                title="Reach Out to a Friend",
                description="Social connection can help lift your spirits",
                duration=None
            ),
            ContentSuggestion(
                type="meditation",
                title="Self-Compassion Meditation",
                description="Be kind to yourself during difficult times",
                duration="12 minutes"
            )
        ])
    
    elif mood in ["Tired"]:
        suggestions.extend([
            ContentSuggestion(
                type="article",
                title="Sleep Hygiene Tips",
                description="Improve your sleep quality with these evidence-based tips",
                duration="6 minutes"
            ),
            ContentSuggestion(
                type="activity",
                title="Power Nap",
                description="A short 20-minute nap can boost energy",
                duration="20 minutes"
            )
        ])
    
    else:  
        suggestions.extend([
            ContentSuggestion(
                type="activity",
                title="Gratitude Practice",
                description="Write down 3 things you're grateful for today",
                duration="5 minutes"
            ),
            ContentSuggestion(
                type="meditation",
                title="Mindfulness Meditation",
                description="Enhance your positive state with mindfulness",
                duration="10 minutes"
            )
        ])
    
    return suggestions[:3]

def calculate_streak():
    """Calculate current mood tracking streak"""
    if not mood_entries_db:
        return 0
    
    streak = 1
    today = datetime.now().date()
    
    for i in range(len(mood_entries_db) - 2, -1, -1):
        entry_date = datetime.fromisoformat(str(mood_entries_db[i]["timestamp"])).date()
        expected_date = today - timedelta(days=streak)
        
        if entry_date == expected_date:
            streak += 1
        else:
            break
    
    return streak
@router.post("/crisis-check")
async def check_for_crisis(text: str):
    """
    Check if text contains crisis indicators
    """
    crisis_keywords = ["suicide", "kill myself", "end it all", "no point living", "self-harm"]
    
    text_lower = text.lower()
    crisis_detected = any(keyword in text_lower for keyword in crisis_keywords)
    
    if crisis_detected:
        return {
            "crisis_detected": True,
            "message": "We're concerned about you. Please reach out for help.",
            "resources": {
                "national_suicide_hotline": "988",
                "crisis_text_line": "Text HOME to 741741",
                "emergency": "Call 108 immediately"
            }
        }
    
    return {
        "crisis_detected": False,
        "message": "No crisis indicators detected"
    }
