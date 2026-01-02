"""
Shared utility functions for Documed backend
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional
import re

def generate_id(prefix: str = "") -> str:
    """Generate unique ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_part = secrets.token_hex(4)
    return f"{prefix}_{timestamp}_{random_part}" if prefix else f"{timestamp}_{random_part}"

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == hashed

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone: str) -> bool:
    """Validate phone number format"""
    # Remove common separators
    phone_clean = re.sub(r'[-()\s]', '', phone)
    # Check if 10-15 digits
    return phone_clean.isdigit() and 10 <= len(phone_clean) <= 15

def format_datetime(dt: datetime) -> str:
    """Format datetime for display"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def get_time_ago(dt: datetime) -> str:
    """Get human-readable time ago string"""
    now = datetime.now()
    diff = now - dt
    
    if diff.days > 365:
        return f"{diff.days // 365} year(s) ago"
    elif diff.days > 30:
        return f"{diff.days // 30} month(s) ago"
    elif diff.days > 0:
        return f"{diff.days} day(s) ago"
    elif diff.seconds > 3600:
        return f"{diff.seconds // 3600} hour(s) ago"
    elif diff.seconds > 60:
        return f"{diff.seconds // 60} minute(s) ago"
    else:
        return "just now"

def calculate_age(birthdate: datetime) -> int:
    """Calculate age from birthdate"""
    today = datetime.now()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))

def sanitize_string(text: str) -> str:
    """Remove potentially harmful characters from string"""
    # Remove control characters
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    return text.strip()

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

class ResponseFormatter:
    """Format API responses consistently"""
    
    @staticmethod
    def success(data, message: str = "Success"):
        return {
            "status": "success",
            "message": message,
            "data": data
        }
    
    @staticmethod
    def error(message: str, code: Optional[str] = None):
        response = {
            "status": "error",
            "message": message
        }
        if code:
            response["error_code"] = code
        return response
    
    @staticmethod
    def paginated(items: list, page: int, total: int, per_page: int):
        return {
            "status": "success",
            "data": items,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": (total + per_page - 1) // per_page
            }
        }
