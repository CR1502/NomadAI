"""
Utility helper functions for the Lifestyle Discovery Assistant.
"""

import re
import hashlib
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import yaml


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file {config_path} not found")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        return {}


def clean_text(text: str) -> str:
    """Clean and normalize text for processing."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.,!?-]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_location_mentions(text: str) -> List[str]:
    """Extract potential location mentions from text."""
    # Simple regex patterns for common location formats
    location_patterns = [
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}\b',  # City, State
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z][a-z]+\b',  # City, Country
        r'in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # "in Location"
        r'at\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # "at Location"
    ]

    locations = []
    for pattern in location_patterns:
        matches = re.findall(pattern, text)
        locations.extend(matches)

    return list(set(locations))  # Remove duplicates


def generate_hash(text: str) -> str:
    """Generate a hash for text deduplication."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def categorize_content(text: str, title: str = "") -> str:
    """Categorize content into travel, food, or events."""
    combined_text = f"{title} {text}".lower()

    # Define keywords for each category
    travel_keywords = [
        'travel', 'trip', 'vacation', 'hotel', 'flight', 'destination',
        'visit', 'sightseeing', 'tourist', 'backpack', 'itinerary'
    ]

    food_keywords = [
        'restaurant', 'food', 'eat', 'dining', 'menu', 'chef', 'cuisine',
        'meal', 'breakfast', 'lunch', 'dinner', 'coffee', 'bar', 'drink'
    ]

    event_keywords = [
        'concert', 'festival', 'event', 'show', 'performance', 'music',
        'theater', 'exhibition', 'conference', 'meetup', 'party', 'nightlife'
    ]

    # Count keyword matches
    travel_score = sum(1 for keyword in travel_keywords if keyword in combined_text)
    food_score = sum(1 for keyword in food_keywords if keyword in combined_text)
    event_score = sum(1 for keyword in event_keywords if keyword in combined_text)

    # Return category with highest score
    scores = {'travel': travel_score, 'food': food_score, 'events': event_score}
    max_category = max(scores, key=scores.get)

    # If all scores are 0 or tied, return 'general'
    if scores[max_category] == 0 or list(scores.values()).count(scores[max_category]) > 1:
        return 'general'

    return max_category


def validate_data_quality(data: Dict[str, Any]) -> bool:
    """Validate if data meets quality requirements."""
    required_fields = ['text', 'source', 'timestamp']

    # Check required fields
    for field in required_fields:
        if field not in data or not data[field]:
            return False

    # Check text length
    text_length = len(data.get('text', ''))
    if text_length < 10 or text_length > 5000:
        return False

    # Check if timestamp is valid
    try:
        if isinstance(data['timestamp'], str):
            datetime.fromisoformat(data['timestamp'])
    except (ValueError, TypeError):
        return False

    return True


def get_time_period(timestamp: datetime) -> str:
    """Categorize timestamp into time periods for analysis."""
    hour = timestamp.hour

    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'


def setup_logging(config_path: str = "config/logging.conf"):
    """Setup logging configuration."""
    import logging.config

    try:
        logging.config.fileConfig(config_path)
    except FileNotFoundError:
        # Fallback to basic logging if config file not found
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.warning(f"Logging config file {config_path} not found, using basic config")


if __name__ == "__main__":
    # Test the functions
    sample_text = "I had an amazing dinner at this restaurant in Austin, TX. The food was incredible!"

    print("Original text:", sample_text)
    print("Cleaned text:", clean_text(sample_text))
    print("Locations found:", extract_location_mentions(sample_text))
    print("Category:", categorize_content(sample_text))
    print("Hash:", generate_hash(sample_text))