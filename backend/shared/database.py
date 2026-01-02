"""
Shared database utilities for Documed backend
"""

from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection (async)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/documed")

class Database:
    """
    MongoDB database handler
    """
    client: AsyncIOMotorClient = None
    
    @classmethod
    async def connect_db(cls):
        """Connect to MongoDB"""
        cls.client = AsyncIOMotorClient(MONGODB_URI)
        print("✅ Connected to MongoDB")
    
    @classmethod
    async def close_db(cls):
        """Close MongoDB connection"""
        cls.client.close()
        print("❌ MongoDB connection closed")
    
    @classmethod
    def get_database(cls):
        """Get database instance"""
        return cls.client.documed
    
    @classmethod
    def get_collection(cls, collection_name: str):
        """Get collection from database"""
        db = cls.get_database()
        return db[collection_name]

# Collections
def get_users_collection():
    """Get users collection"""
    return Database.get_collection("users")

def get_mood_entries_collection():
    """Get mood entries collection"""
    return Database.get_collection("mood_entries")

def get_vitals_collection():
    """Get vitals collection"""
    return Database.get_collection("vitals")

def get_medications_collection():
    """Get medications collection"""
    return Database.get_collection("medications")

def get_appointments_collection():
    """Get appointments collection"""
    return Database.get_collection("appointments")

def get_emergency_contacts_collection():
    """Get emergency contacts collection"""
    return Database.get_collection("emergency_contacts")

def get_analysis_reports_collection():
    """Get analysis reports collection"""
    return Database.get_collection("analysis_reports")
