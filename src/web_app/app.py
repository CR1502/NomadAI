"""
Enhanced Streamlit application with Google Places API integration and Smart Trip Planner.
Single-page view with restaurants, attractions, Reddit posts, and cost estimates.
100% Free AI using local ML and Reddit data analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple
import json
import os
import sys
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import googlemaps
import re
from datetime import datetime
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import logging

# Load environment variables
load_dotenv('docker/.env')

# Add the src directory to the path
if 'src' not in sys.path:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, '..', '..')
    if os.path.exists(src_dir):
        sys.path.insert(0, src_dir)

# Page configuration
st.set_page_config(
    page_title="Nomad AI - Enhanced Discovery",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with all styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #64b5f6 0%, #42a5f5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .nomad-logo {
        position: absolute;
        top: 20px;
        right: 30px;
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
        letter-spacing: 0.5px;
    }

    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #ffffff !important;
        margin: 2rem 0 1rem 0;
        padding: 1rem;
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        border-radius: 0.5rem;
        text-align: center;
    }

    .smart-ai-section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #ffffff !important;
        margin: 2rem 0 1rem 0;
        padding: 1rem;
        background: linear-gradient(135deg, #9c27b0 0%, #673ab7 100%);
        border-radius: 0.5rem;
        text-align: center;
    }

    .recommendation-card {
        background: #ffffff;
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4a90e2;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
    }

    .place-name {
        color: #0d47a1 !important;
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .place-details {
        color: #424242 !important;
        line-height: 1.5;
        margin-bottom: 0.8rem;
    }

    .rating-badge {
        background: #4caf50;
        color: white !important;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 8px;
    }

    .price-badge {
        background: #ff9800;
        color: white !important;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }

    .reddit-post-card {
        background: #f8fafe;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #4a90e2;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .positive-post {
        border-left-color: #4caf50 !important;
        background: #f1f8e9;
    }

    .negative-post {
        border-left-color: #f44336 !important;
        background: #ffebee;
    }

    .post-title {
        color: #0d47a1 !important;
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 0.8rem;
    }

    .post-content {
        color: #212121 !important;
        line-height: 1.6;
        margin-bottom: 0.8rem;
    }

    .post-meta {
        color: #666666 !important;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .reddit-link {
        color: #4a90e2 !important;
        font-weight: bold;
        text-decoration: none;
        background: rgba(74, 144, 226, 0.1);
        padding: 6px 12px;
        border-radius: 6px;
        margin-top: 10px;
        display: inline-block;
        transition: all 0.3s ease;
        border: 1px solid rgba(74, 144, 226, 0.3);
    }

    .reddit-link:hover {
        background: rgba(74, 144, 226, 0.2);
        text-decoration: none;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(74, 144, 226, 0.3);
    }

    .negative-reddit-link {
        color: #f44336 !important;
        background: rgba(244, 67, 54, 0.1);
        border: 1px solid rgba(244, 67, 54, 0.3);
    }

    .negative-reddit-link:hover {
        background: rgba(244, 67, 54, 0.2);
        box-shadow: 0 2px 4px rgba(244, 67, 54, 0.3);
    }

    .smart-itinerary-card {
        background: linear-gradient(135deg, #9c27b0 0%, #673ab7 100%);
        color: white !important;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 2rem;
    }

    .smart-day-section {
        background: #f3e5f5;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid #9c27b0;
    }

    .smart-activity-item {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.8rem;
        border-left: 3px solid #9c27b0;
        margin-left: 2rem;
    }

    .expense-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white !important;
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        text-align: center;
    }

    .expense-item {
        background: #ffffff;
        color: #333333 !important;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .quality-badge {
        background: #9c27b0;
        color: white !important;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 8px;
    }

    .method-badge {
        background: #607d8b;
        color: white !important;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 10px;
    }

    /* Sidebar styling */
    .stSelectbox label, .stSlider label, .stSubheader, .stCheckbox label, .stMultiselect label {
        color: #ffffff !important;
    }

    .css-1d391kg, .css-1d391kg p, .css-1d391kg span, .css-1d391kg div {
        color: #ffffff !important;
    }

    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #ffffff !important;
    }

    .main-content-description {
        color: #b0b0b0 !important;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }

    /* Form styling */
    .stForm {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# Data Quality Enhancer Class (Embedded)
class DataQualityEnhancer:
    """Advanced data quality enhancement using local ML and NLP."""

    def __init__(self):
        """Initialize the data quality enhancer."""
        self.nlp = None
        self.load_nlp_model()

    def load_nlp_model(self):
        """Load spaCy NLP model."""
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            st.warning("spaCy model not available. Some enhanced features will be limited.")
            self.nlp = None

    def enhanced_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Advanced sentiment analysis using multiple methods."""

        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment.polarity

        # Custom travel-specific sentiment
        travel_sentiment = self._calculate_travel_sentiment(text)

        # Combined sentiment score
        combined_score = (textblob_sentiment + travel_sentiment) / 2

        # Confidence score
        confidence = abs(combined_score)

        # Label assignment
        if combined_score > 0.15:
            label = 'positive'
        elif combined_score < -0.15:
            label = 'negative'
        else:
            label = 'neutral'

        return {
            'sentiment_score': combined_score,
            'sentiment_label': label,
            'confidence': confidence,
            'textblob_score': textblob_sentiment,
            'travel_specific_score': travel_sentiment
        }

    def _calculate_travel_sentiment(self, text: str) -> float:
        """Calculate travel-specific sentiment score."""
        text_lower = text.lower()

        # Travel-specific positive words with weights
        positive_words = {
            'amazing': 1.0, 'incredible': 1.0, 'fantastic': 0.9, 'beautiful': 0.8,
            'stunning': 0.9, 'perfect': 0.8, 'love': 0.7, 'great': 0.6,
            'wonderful': 0.8, 'excellent': 0.8, 'must visit': 1.0, 'highly recommend': 1.0,
            'hidden gem': 0.9, 'authentic': 0.7, 'local favorite': 0.8, 'worth it': 0.7
        }

        # Travel-specific negative words with weights
        negative_words = {
            'tourist trap': -1.0, 'overrated': -0.8, 'overpriced': -0.7, 'crowded': -0.5,
            'disappointing': -0.9, 'waste of money': -1.0, 'avoid': -1.0, 'terrible': -0.9,
            'awful': -0.8, 'dirty': -0.6, 'rude': -0.6, 'scam': -1.0
        }

        score = 0.0
        word_count = 0

        # Calculate weighted sentiment
        for word, weight in positive_words.items():
            if word in text_lower:
                score += weight
                word_count += 1

        for word, weight in negative_words.items():
            if word in text_lower:
                score += weight
                word_count += 1

        # Normalize by word count
        if word_count > 0:
            score = score / word_count

        return max(-1.0, min(1.0, score))

    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy."""
        if not self.nlp:
            return {'locations': [], 'organizations': [], 'persons': [], 'money': [], 'dates': []}

        doc = self.nlp(text)

        entities = {
            'locations': [],
            'organizations': [],
            'persons': [],
            'money': [],
            'dates': []
        }

        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC']:
                entities['locations'].append(ent.text.strip())
            elif ent.label_ in ['ORG']:
                entities['organizations'].append(ent.text.strip())
            elif ent.label_ in ['PERSON']:
                entities['persons'].append(ent.text.strip())
            elif ent.label_ in ['MONEY']:
                entities['money'].append(ent.text.strip())
            elif ent.label_ in ['DATE', 'TIME']:
                entities['dates'].append(ent.text.strip())

        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities


# Smart Trip Planner Class (100% Free)
class SmartTripPlanner:
    """Intelligent trip planner using local ML and Reddit data analysis."""

    def __init__(self):
        """Initialize the smart trip planner."""
        self.nlp = None
        self.load_nlp()

    def load_nlp(self):
        """Load spaCy for text analysis."""
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = None

    def analyze_reddit_for_recommendations(self, posts: List[Dict[str, Any]], location: str) -> Dict[str, Any]:
        """Analyze Reddit posts to extract intelligent recommendations."""

        recommendations = {
            'morning_activities': [],
            'restaurants': [],
            'attractions': [],
            'evening_activities': [],
            'tips': [],
            'hidden_gems': [],
            'budget_advice': []
        }

        for post in posts:
            text = f"{post.get('title', '')} {post.get('text', '')} {post.get('summary', '')}"
            score = post.get('score', 0)

            # Extract different types of recommendations
            recommendations['morning_activities'].extend(
                self._extract_time_specific_activities(text, 'morning', score)
            )
            recommendations['restaurants'].extend(
                self._extract_restaurant_mentions(text, score)
            )
            recommendations['attractions'].extend(
                self._extract_attraction_mentions(text, score)
            )
            recommendations['evening_activities'].extend(
                self._extract_time_specific_activities(text, 'evening', score)
            )
            recommendations['tips'].extend(
                self._extract_travel_tips(text, score)
            )
            recommendations['hidden_gems'].extend(
                self._extract_hidden_gems(text, score)
            )
            recommendations['budget_advice'].extend(
                self._extract_budget_advice(text, score)
            )

        # Rank and filter recommendations
        for category in recommendations:
            recommendations[category] = self._rank_recommendations(recommendations[category])

        return recommendations

    def _extract_time_specific_activities(self, text: str, time_period: str, score: int) -> List[Dict]:
        """Extract activities for specific times of day."""
        activities = []
        text_lower = text.lower()

        time_indicators = {
            'morning': ['morning', 'early', 'sunrise', 'breakfast', '9am', '10am', 'am'],
            'evening': ['evening', 'night', 'sunset', 'dinner', '7pm', '8pm', 'pm', 'nightlife']
        }

        if any(indicator in text_lower for indicator in time_indicators.get(time_period, [])):
            # Extract activity mentions
            activity_patterns = [
                r'(?:go|visit|check out|try|see)\s+([A-Z][a-zA-Z\s]{2,30})',
                r'([A-Z][a-zA-Z\s]{2,30})\s+(?:is|was)\s+(?:amazing|great|perfect|beautiful)',
                r'recommend\s+([A-Z][a-zA-Z\s]{2,30})'
            ]

            for pattern in activity_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if self._is_valid_activity(match, text):
                        activities.append({
                            'name': match.strip().title(),
                            'context': self._extract_context(match, text),
                            'score': score,
                            'time_period': time_period
                        })

        return activities

    def _extract_restaurant_mentions(self, text: str, score: int) -> List[Dict]:
        """Extract restaurant mentions with context."""
        restaurants = []

        # Restaurant-specific patterns
        patterns = [
            r'restaurant\s+called\s+([A-Z][a-zA-Z\s&\'-]{2,35})',
            r'([A-Z][a-zA-Z\s&\'-]{2,35})\s+restaurant',
            r'ate\s+at\s+([A-Z][a-zA-Z\s&\'-]{2,35})',
            r'food\s+at\s+([A-Z][a-zA-Z\s&\'-]{2,35})',
            r'try\s+([A-Z][a-zA-Z\s&\'-]{2,35})',
            r'([A-Z][a-zA-Z\s&\'-]{2,35})\s+(?:has|serves)\s+(?:amazing|great|delicious)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if self._is_valid_restaurant_name(match, text):
                    restaurants.append({
                        'name': match.strip().title(),
                        'context': self._extract_context(match, text),
                        'score': score,
                        'type': 'restaurant'
                    })

        return restaurants

    def _extract_attraction_mentions(self, text: str, score: int) -> List[Dict]:
        """Extract attraction mentions with context."""
        attractions = []

        # Attraction-specific patterns
        attraction_keywords = ['museum', 'cathedral', 'palace', 'temple', 'park', 'monument', 'gallery', 'tower',
                               'bridge', 'square']

        patterns = [
            r'visit\s+(?:the\s+)?([A-Z][a-zA-Z\s&\'-]{2,40})',
            r'see\s+(?:the\s+)?([A-Z][a-zA-Z\s&\'-]{2,40})',
            r'([A-Z][a-zA-Z\s&\'-]{2,40})\s+(?:' + '|'.join(attraction_keywords) + ')',
            r'must\s+see\s+([A-Z][a-zA-Z\s&\'-]{2,40})',
            r'don\'t\s+miss\s+([A-Z][a-zA-Z\s&\'-]{2,40})'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if self._is_valid_attraction_name(match, text):
                    attractions.append({
                        'name': match.strip().title(),
                        'context': self._extract_context(match, text),
                        'score': score,
                        'type': 'attraction'
                    })

        return attractions

    def _extract_travel_tips(self, text: str, score: int) -> List[Dict]:
        """Extract practical travel tips."""
        tips = []

        tip_patterns = [
            r'tip[s]?[:\-]\s*([^.!?]{15,120})',
            r'(?:make sure|remember to|don\'t forget)\s+([^.!?]{10,100})',
            r'(?:pro tip|advice)[:\-]\s*([^.!?]{15,120})',
            r'(?:i recommend|would recommend)\s+([^.!?]{10,100})',
            r'(?:important|helpful)[:\-]\s*([^.!?]{15,120})'
        ]

        for pattern in tip_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                tip = match.strip()
                if len(tip) > 10 and self._is_useful_tip(tip):
                    tips.append({
                        'tip': tip,
                        'score': score,
                        'type': 'tip'
                    })

        return tips

    def _extract_hidden_gems(self, text: str, score: int) -> List[Dict]:
        """Extract hidden gems and local favorites."""
        gems = []
        text_lower = text.lower()

        gem_indicators = ['hidden gem', 'local favorite', 'locals only', 'off the beaten path',
                          'secret spot', 'not touristy', 'locals go', 'authentic', 'insider tip']

        if any(indicator in text_lower for indicator in gem_indicators):
            # Extract the gem mention
            sentences = text.split('.')
            for sentence in sentences:
                if (any(indicator in sentence.lower() for indicator in gem_indicators) and
                        len(sentence.strip()) > 20):
                    gems.append({
                        'description': sentence.strip(),
                        'score': score,
                        'type': 'hidden_gem'
                    })

        return gems

    def _extract_budget_advice(self, text: str, score: int) -> List[Dict]:
        """Extract budget and money-saving advice."""
        budget_advice = []
        text_lower = text.lower()

        budget_indicators = ['cheap', 'budget', 'free', 'expensive', 'cost', 'price', 'money',
                             'affordable', 'save money', 'deal', 'discount', '$']

        if any(indicator in text_lower for indicator in budget_indicators):
            sentences = text.split('.')
            for sentence in sentences:
                if (any(indicator in sentence.lower() for indicator in budget_indicators) and
                        len(sentence.strip()) > 20 and len(sentence.strip()) < 150):
                    budget_advice.append({
                        'advice': sentence.strip(),
                        'score': score,
                        'type': 'budget'
                    })

        return budget_advice

    def _is_valid_activity(self, name: str, text: str) -> bool:
        """Validate if extracted text is a real activity."""
        invalid_words = {'Most', 'Best', 'All', 'Very', 'Really', 'Just', 'Only', 'Also', 'When', 'Where', 'What',
                         'This', 'That'}
        return (len(name) > 3 and len(name) < 40 and name not in invalid_words and
                any(word in text.lower() for word in ['visit', 'go', 'see', 'check', 'try']))

    def _is_valid_restaurant_name(self, name: str, text: str) -> bool:
        """Validate restaurant name extraction."""
        invalid_words = {'Most', 'Best', 'All', 'Very', 'Really', 'Just', 'Only', 'Also', 'Good', 'Great', 'Nice'}
        return (len(name) > 2 and len(name) < 40 and name not in invalid_words and
                any(word in text.lower() for word in
                    ['food', 'eat', 'restaurant', 'delicious', 'meal', 'lunch', 'dinner']))

    def _is_valid_attraction_name(self, name: str, text: str) -> bool:
        """Validate attraction name extraction."""
        invalid_words = {'Most', 'Best', 'All', 'Very', 'Really', 'Just', 'Only', 'Also', 'Good', 'Great', 'Nice'}
        return (len(name) > 2 and len(name) < 50 and name not in invalid_words and
                any(word in text.lower() for word in ['visit', 'see', 'beautiful', 'historic', 'museum', 'attraction']))

    def _is_useful_tip(self, tip: str) -> bool:
        """Check if tip is useful and actionable."""
        return (len(tip) > 15 and len(tip) < 150 and
                any(word in tip.lower() for word in
                    ['book', 'avoid', 'bring', 'wear', 'go', 'try', 'don\'t', 'make sure']))

    def _extract_context(self, mention: str, text: str) -> str:
        """Extract context around a mention."""
        text_lower = text.lower()
        mention_pos = text_lower.find(mention.lower())

        if mention_pos == -1:
            return "Recommended by the community"

        start = max(0, mention_pos - 60)
        end = min(len(text), mention_pos + len(mention) + 80)
        context = text[start:end].strip()

        # Find the most relevant sentence
        sentences = context.split('.')
        for sentence in sentences:
            if mention.lower() in sentence.lower() and len(sentence.strip()) > 15:
                return sentence.strip()

        return "Highly recommended by travelers"

    def _rank_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Rank recommendations by score and remove duplicates."""
        if not recommendations:
            return []

        # Group by name/content
        grouped = {}
        for rec in recommendations:
            key = rec.get('name', rec.get('tip', rec.get('description', '')))[:30].lower()

            if key in grouped:
                grouped[key]['score'] += rec['score']
                grouped[key]['count'] = grouped[key].get('count', 1) + 1
            else:
                grouped[key] = rec.copy()
                grouped[key]['count'] = 1

        # Sort by score and return top items
        ranked = sorted(grouped.values(), key=lambda x: x['score'], reverse=True)
        return ranked[:8]

    def generate_smart_itinerary(
            self,
            location: str,
            reddit_posts: List[Dict[str, Any]],
            user_preferences: Dict[str, Any],
            restaurants: List[Dict] = None,
            attractions: List[Dict] = None
    ) -> Dict[str, Any]:
        """Generate intelligent itinerary using Reddit analysis and smart templates."""

        # Analyze Reddit posts for recommendations
        reddit_recommendations = self.analyze_reddit_for_recommendations(reddit_posts, location)

        # Extract user preferences
        duration = user_preferences.get('duration', 3)
        budget = user_preferences.get('budget', 'medium')
        interests = user_preferences.get('interests', [])
        travel_style = user_preferences.get('travel_style', 'balanced')

        # Generate days
        days = []

        for day_num in range(1, duration + 1):
            day_title, activities = self._generate_day_activities(
                day_num, duration, location, reddit_recommendations,
                restaurants, attractions, interests, travel_style
            )

            days.append({
                'day': day_num,
                'title': day_title,
                'activities': activities
            })

        # Compile tips and advice
        reddit_tips = [tip['tip'] for tip in reddit_recommendations['tips'][:5]]
        budget_notes = [advice['advice'] for advice in reddit_recommendations['budget_advice'][:3]]
        hidden_gems = [gem['description'] for gem in reddit_recommendations['hidden_gems'][:3]]

        return {
            'location': location,
            'user_preferences': user_preferences,
            'days': days,
            'reddit_tips': reddit_tips,
            'budget_notes': budget_notes,
            'hidden_gems': hidden_gems,
            'generated_at': datetime.now().isoformat(),
            'ai_generated': True,
            'method': 'Smart Analysis + Reddit Data',
            'reddit_insights_used': len(reddit_posts)
        }

    def _generate_day_activities(
            self,
            day_num: int,
            total_days: int,
            location: str,
            reddit_recs: Dict[str, List],
            restaurants: List[Dict],
            attractions: List[Dict],
            interests: List[str],
            travel_style: str
    ) -> Tuple[str, List[Dict]]:
        """Generate activities for a specific day."""

        # Day themes based on travel style and day number
        day_themes = {
            1: "Arrival & City Center Exploration",
            2: "Cultural Immersion & Local Favorites",
            3: "Hidden Gems & Authentic Experiences",
            4: "Adventure & Deep Exploration",
            5: "Local Life & Unique Discoveries",
            6: "Relaxation & Special Experiences",
            7: "Final Highlights & Departure Prep"
        }

        day_title = day_themes.get(day_num, f"Day {day_num} - {location} Discovery")
        activities = []

        # Morning activity (9:00-10:00 AM)
        morning_time = ["9:00 AM", "9:30 AM", "10:00 AM"][min(day_num - 1, 2)]

        # Choose morning activity intelligently
        if day_num == 1:
            # First day - major attraction
            if attractions:
                activity = attractions[0]
                reddit_insight = self._get_reddit_insight(activity['name'], reddit_recs)
                activities.append({
                    'time': morning_time,
                    'activity': activity['name'],
                    'description': f"Start your {location} adventure at this top-rated attraction ({activity.get('rating', 'N/A')}⭐). {reddit_insight}",
                    'type': 'attraction'
                })
        else:
            # Use Reddit community recommendations
            if reddit_recs['attractions'] and len(reddit_recs['attractions']) >= day_num - 1:
                reddit_attraction = reddit_recs['attractions'][day_num - 2]
                activities.append({
                    'time': morning_time,
                    'activity': reddit_attraction['name'],
                    'description': f"{reddit_attraction['context']} (Discovered from Reddit community with {reddit_attraction['score']} upvotes)",
                    'type': 'attraction'
                })
            elif attractions and len(attractions) > day_num - 1:
                activity = attractions[day_num - 1]
                activities.append({
                    'time': morning_time,
                    'activity': activity['name'],
                    'description': f"Explore this {activity.get('rating', 'highly-rated')}⭐ attraction. Perfect for your {travel_style} travel style.",
                    'type': 'attraction'
                })

        # Lunch (12:00-1:00 PM)
        lunch_time = ["12:00 PM", "12:30 PM", "1:00 PM"][min(day_num - 1, 2)]

        if restaurants:
            restaurant_idx = min(day_num - 1, len(restaurants) - 1)
            restaurant = restaurants[restaurant_idx]

            # Add Reddit insights about food
            food_insight = self._get_food_insight(reddit_recs, interests)
            price_desc = self._get_price_description(restaurant.get('price_level', 2))

            activities.append({
                'time': lunch_time,
                'activity': restaurant['name'],
                'description': f"Lunch at this {price_desc} restaurant ({restaurant.get('rating', 'N/A')}⭐). {food_insight}",
                'type': 'restaurant'
            })

        # Afternoon activity (2:00-3:00 PM)
        afternoon_time = ["2:00 PM", "2:30 PM", "3:00 PM"][min(day_num - 1, 2)]

        # Choose based on interests and day
        if 'culture' in interests and attractions and len(attractions) > day_num:
            activity = attractions[day_num]
            activities.append({
                'time': afternoon_time,
                'activity': activity['name'],
                'description': f"Perfect for culture enthusiasts. {self._get_reddit_insight(activity['name'], reddit_recs)} Rating: {activity.get('rating', 'N/A')}⭐",
                'type': 'attraction'
            })
        elif reddit_recs['hidden_gems'] and day_num > 1:
            gem_idx = min(day_num - 2, len(reddit_recs['hidden_gems']) - 1)
            gem = reddit_recs['hidden_gems'][gem_idx]
            activities.append({
                'time': afternoon_time,
                'activity': "Local Hidden Gem Experience",
                'description': f"{gem['description']} (Community score: {gem['score']} upvotes)",
                'type': 'hidden_gem'
            })
        elif 'shopping' in interests:
            activities.append({
                'time': afternoon_time,
                'activity': f"{location} Local Markets",
                'description': f"Explore local markets and shopping areas. Great for finding authentic {location} souvenirs and experiencing local life.",
                'type': 'shopping'
            })

        # Evening activity (6:00-8:00 PM)
        evening_time = ["6:00 PM", "7:00 PM", "8:00 PM"][min(day_num - 1, 2)]

        if 'nightlife' in interests and reddit_recs['evening_activities']:
            evening_activity = reddit_recs['evening_activities'][0]
            activities.append({
                'time': evening_time,
                'activity': evening_activity['name'],
                'description': f"{evening_activity['context']} Perfect for experiencing {location}'s nightlife scene.",
                'type': 'nightlife'
            })
        elif restaurants and len(restaurants) > day_num:
            restaurant = restaurants[day_num]
            activities.append({
                'time': evening_time,
                'activity': restaurant['name'],
                'description': f"Dinner at this {self._get_price_description(restaurant.get('price_level', 2))} restaurant ({restaurant.get('rating', 'N/A')}⭐). End your day with excellent {location} cuisine!",
                'type': 'restaurant'
            })

        return day_title, activities

    def _get_reddit_insight(self, place_name: str, reddit_recs: Dict) -> str:
        """Get relevant Reddit insight for a place."""
        # Look for mentions in Reddit recommendations
        for category in reddit_recs.values():
            if isinstance(category, list):
                for item in category:
                    if isinstance(item, dict) and place_name.lower() in item.get('context', '').lower():
                        return f"Reddit community says: {item['context'][:100]}..."

        return "Highly recommended by travelers."

    def _get_food_insight(self, reddit_recs: Dict, interests: List[str]) -> str:
        """Get food-related insights from Reddit."""
        if reddit_recs['restaurants']:
            return f"Community insight: {reddit_recs['restaurants'][0]['context'][:80]}..."
        elif 'food' in interests:
            return "Great for food enthusiasts based on community recommendations."
        else:
            return "Popular choice among travelers for authentic local cuisine."

    def _get_price_description(self, price_level: int) -> str:
        """Convert price level to description."""
        descriptions = {
            0: "very budget-friendly",
            1: "budget-friendly",
            2: "moderately priced",
            3: "upscale",
            4: "high-end luxury"
        }
        return descriptions.get(price_level, "moderately priced")


@st.cache_resource
def get_s3_client():
    """Initialize S3 client."""
    try:
        return boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )
    except Exception as e:
        st.error(f"Failed to initialize S3 client: {e}")
        return None


@st.cache_resource
def get_google_places_client():
    """Initialize Google Places client."""
    try:
        api_key = os.getenv('GOOGLE_PLACES_API_KEY')
        if not api_key:
            st.error("Google Places API key not found.")
            return None
        return googlemaps.Client(key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Google Places client: {e}")
        return None


@st.cache_data
def load_extraction_summary():
    """Load the extraction summary from S3."""
    s3_client = get_s3_client()
    if not s3_client:
        return {}

    try:
        bucket_name = os.getenv('S3_BUCKET_NAME')
        response = s3_client.get_object(Bucket=bucket_name, Key='extraction_summary.json')
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        try:
            with open('data/summaries/extraction_summary.json', 'r') as f:
                return json.load(f)
        except:
            return {}


@st.cache_data
def load_location_data(location: str, category: str):
    """Load Reddit data for a specific location and category."""
    s3_client = get_s3_client()
    if not s3_client:
        return []

    try:
        bucket_name = os.getenv('S3_BUCKET_NAME')
        s3_key = f"{location.lower().replace(' ', '_')}/{category}/reddit_posts.json"

        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except ClientError as e:
        if e.response['Error']['Code'] != 'NoSuchKey':
            st.error(f"Error loading {location} {category} data: {e}")
        return []
    except Exception as e:
        return []


@st.cache_data
def get_google_places_data(location: str):
    """Get verified restaurants and attractions from Google Places API."""
    gmaps = get_google_places_client()
    if not gmaps:
        return {'restaurants': [], 'attractions': []}

    try:
        # Get location coordinates
        geocode_result = gmaps.geocode(location)
        if not geocode_result:
            st.error(f"Could not find coordinates for {location}")
            return {'restaurants': [], 'attractions': []}

        lat_lng = geocode_result[0]['geometry']['location']

        # Search for restaurants
        restaurants_result = gmaps.places_nearby(
            location=lat_lng,
            radius=8000,
            type='restaurant',
            language='en'
        )

        # Search for tourist attractions
        attractions_result = gmaps.places_nearby(
            location=lat_lng,
            radius=8000,
            type='tourist_attraction',
            language='en'
        )

        # Get details for top results
        restaurants = []
        for place in restaurants_result.get('results', [])[:12]:
            try:
                details = gmaps.place(
                    place_id=place['place_id'],
                    fields=['name', 'rating', 'user_ratings_total', 'price_level',
                            'formatted_address', 'website', 'formatted_phone_number']
                )
                restaurants.append({
                    **place,
                    'details': details.get('result', {})
                })
            except Exception:
                continue

        attractions = []
        for place in attractions_result.get('results', [])[:12]:
            try:
                details = gmaps.place(
                    place_id=place['place_id'],
                    fields=['name', 'rating', 'user_ratings_total', 'formatted_address',
                            'website', 'formatted_phone_number']
                )
                attractions.append({
                    **place,
                    'details': details.get('result', {})
                })
            except Exception:
                continue

        return {
            'restaurants': restaurants,
            'attractions': attractions
        }

    except Exception as e:
        st.error(f"Error fetching Google Places data: {e}")
        return {'restaurants': [], 'attractions': []}


@st.cache_data
def enhance_reddit_data(posts: List[Dict[str, Any]], location: str) -> Dict[str, Any]:
    """Apply data quality enhancements to Reddit posts."""

    if not posts:
        return {'enhanced_posts': [], 'positive': [], 'negative': []}

    enhancer = DataQualityEnhancer()

    # Apply quality enhancements
    enhanced_posts = []
    for post in posts:
        text = f"{post.get('title', '')} {post.get('text', '')} {post.get('summary', '')}"

        # Enhanced sentiment
        sentiment = enhancer.enhanced_sentiment_analysis(text)

        # Named entities
        entities = enhancer.extract_named_entities(text)

        # Update post
        enhanced_post = post.copy()
        enhanced_post.update({
            'enhanced_sentiment': sentiment,
            'entities': entities
        })

        enhanced_posts.append(enhanced_post)

    # Analyze sentiment for positive/negative classification
    positive_posts = []
    negative_posts = []

    for post in enhanced_posts:
        sentiment = post.get('enhanced_sentiment', {})

        if sentiment.get('sentiment_label') == 'positive' and sentiment.get('confidence', 0) > 0.3:
            positive_posts.append(post)
        elif sentiment.get('sentiment_label') == 'negative' and sentiment.get('confidence', 0) > 0.2:
            negative_posts.append(post)

    # Sort by confidence and score
    positive_posts.sort(key=lambda x: (x.get('enhanced_sentiment', {}).get('confidence', 0), x.get('score', 0)),
                        reverse=True)
    negative_posts.sort(key=lambda x: (x.get('enhanced_sentiment', {}).get('confidence', 0), x.get('score', 0)),
                        reverse=True)

    return {
        'enhanced_posts': enhanced_posts,
        'positive': positive_posts[:5],
        'negative': negative_posts[:3]
    }


def estimate_costs(location: str, restaurants: List[Dict], attractions: List[Dict]) -> Dict[str, Any]:
    """Estimate costs for visiting the location."""

    cost_estimates = {
        'Paris': {'accommodation': 120, 'food': 60, 'transport': 25, 'attractions': 30},
        'London': {'accommodation': 150, 'food': 70, 'transport': 30, 'attractions': 35},
        'Tokyo': {'accommodation': 100, 'food': 50, 'transport': 20, 'attractions': 25},
        'New York': {'accommodation': 180, 'food': 80, 'transport': 35, 'attractions': 40},
        'Rome': {'accommodation': 90, 'food': 45, 'transport': 20, 'attractions': 25},
        'Barcelona': {'accommodation': 85, 'food': 40, 'transport': 18, 'attractions': 22},
        'Amsterdam': {'accommodation': 110, 'food': 55, 'transport': 25, 'attractions': 28},
        'Berlin': {'accommodation': 70, 'food': 35, 'transport': 15, 'attractions': 20},
        'Prague': {'accommodation': 50, 'food': 25, 'transport': 12, 'attractions': 15},
        'Vienna': {'accommodation': 80, 'food': 40, 'transport': 18, 'attractions': 22},
        'Istanbul': {'accommodation': 45, 'food': 20, 'transport': 8, 'attractions': 12},
        'Dubai': {'accommodation': 130, 'food': 65, 'transport': 25, 'attractions': 35},
        'Bangkok': {'accommodation': 40, 'food': 15, 'transport': 8, 'attractions': 10},
        'Singapore': {'accommodation': 95, 'food': 45, 'transport': 20, 'attractions': 25},
        'Hong Kong': {'accommodation': 120, 'food': 55, 'transport': 22, 'attractions': 28},
        'Sydney': {'accommodation': 130, 'food': 65, 'transport': 30, 'attractions': 35},
        'Los Angeles': {'accommodation': 140, 'food': 70, 'transport': 40, 'attractions': 35},
        'Chicago': {'accommodation': 120, 'food': 60, 'transport': 25, 'attractions': 30},
        'Las Vegas': {'accommodation': 90, 'food': 60, 'transport': 20, 'attractions': 50},
        'Miami': {'accommodation': 130, 'food': 65, 'transport': 25, 'attractions': 30},
        'San Francisco': {'accommodation': 160, 'food': 75, 'transport': 30, 'attractions': 35},
        'Venice': {'accommodation': 110, 'food': 50, 'transport': 20, 'attractions': 25},
        'Florence': {'accommodation': 95, 'food': 45, 'transport': 15, 'attractions': 22},
        'Athens': {'accommodation': 60, 'food': 30, 'transport': 12, 'attractions': 18},
        'Lisbon': {'accommodation': 70, 'food': 35, 'transport': 15, 'attractions': 20}
    }

    base_costs = cost_estimates.get(location, {
        'accommodation': 100, 'food': 50, 'transport': 20, 'attractions': 25
    })

    # Calculate restaurant costs
    restaurant_cost = 0
    for restaurant in restaurants[:8]:
        price_level = restaurant.get('price_level', 2)
        daily_meal_cost = [15, 25, 45, 70, 100][price_level] if price_level <= 4 else 50
        restaurant_cost += daily_meal_cost

    avg_restaurant_cost = restaurant_cost / len(restaurants) if restaurants else base_costs['food']
    daily_total = base_costs['accommodation'] + avg_restaurant_cost + base_costs['transport'] + base_costs[
        'attractions']

    return {
        'daily_breakdown': base_costs,
        'daily_total': daily_total,
        'trip_3_days': daily_total * 3,
        'trip_1_week': daily_total * 7,
        'avg_restaurant_cost': avg_restaurant_cost,
        'currency': 'USD'
    }


def display_restaurants(restaurants: List[Dict], location: str):
    """Display verified restaurants from Google Places."""
    if not restaurants:
        st.warning("No restaurant data available from Google Places API.")
        return

    st.markdown('<div class="section-header">🍽️ Top Restaurants</div>', unsafe_allow_html=True)

    for i, restaurant in enumerate(restaurants[:8], 1):
        details = restaurant.get('details', {})

        rating = details.get('rating', restaurant.get('rating', 0))
        price_level = restaurant.get('price_level', 2)
        price_symbols = ['$', '$', '$$', '$$$', '$$$$'][price_level] if price_level <= 4 else '$$'

        website = details.get('website', 'Not available')
        website_display = website if website != 'Not available' else 'Not available'
        if len(website_display) > 50:
            website_display = website_display[:47] + "..."

        restaurant_card = f"""
        <div class="recommendation-card">
            <div class="place-name">
                {i}. {restaurant['name']}
            </div>
            <div class="place-details">
                <span class="rating-badge">⭐ {rating:.1f}</span>
                <span class="price-badge">{price_symbols}</span>
                <br><br>
                <strong>📍 Address:</strong> {details.get('formatted_address', 'Address not available')}<br>
                <strong>📞 Phone:</strong> {details.get('formatted_phone_number', 'Not available')}<br>
                <strong>🌐 Website:</strong> {website_display}
            </div>
        </div>
        """

        st.markdown(restaurant_card, unsafe_allow_html=True)


def display_attractions(attractions: List[Dict], location: str):
    """Display verified attractions from Google Places."""
    if not attractions:
        st.warning("No attraction data available from Google Places API.")
        return

    st.markdown('<div class="section-header">🏛️ Top Attractions</div>', unsafe_allow_html=True)

    for i, attraction in enumerate(attractions[:8], 1):
        details = attraction.get('details', {})

        rating = details.get('rating', attraction.get('rating', 0))

        website = details.get('website', 'Not available')
        website_display = website if website != 'Not available' else 'Not available'
        if len(website_display) > 50:
            website_display = website_display[:47] + "..."

        attraction_card = f"""
        <div class="recommendation-card">
            <div class="place-name">
                {i}. {attraction['name']}
            </div>
            <div class="place-details">
                <span class="rating-badge">⭐ {rating:.1f}</span>
                <br><br>
                <strong>📍 Address:</strong> {details.get('formatted_address', 'Address not available')}<br>
                <strong>📞 Phone:</strong> {details.get('formatted_phone_number', 'Not available')}<br>
                <strong>🌐 Website:</strong> {website_display}
            </div>
        </div>
        """

        st.markdown(attraction_card, unsafe_allow_html=True)


def display_reddit_insights(sentiment_analysis: Dict[str, List[Dict]], location: str):
    """Display positive and negative Reddit posts with links and enhanced data."""

    # Positive recommendations
    if sentiment_analysis.get('positive'):
        st.markdown('<div class="section-header">✅ Community Favorites</div>', unsafe_allow_html=True)

        for i, post in enumerate(sentiment_analysis['positive'], 1):
            # Create Reddit URL
            post_url = post.get('url', '')
            if post_url.startswith('/r/'):
                reddit_url = f"https://reddit.com{post_url}"
            elif post_url.startswith('https://'):
                reddit_url = post_url
            else:
                reddit_url = f"https://reddit.com/r/{post.get('subreddit', 'travel')}/comments/{post.get('id', '')}"

            # Get enhanced sentiment data
            enhanced_sentiment = post.get('enhanced_sentiment', {})
            entities = post.get('entities', {})

            # Build quality badges
            confidence = enhanced_sentiment.get('confidence', 0)
            quality_badge = f'<span class="quality-badge">Quality: {confidence:.1f}</span>' if confidence > 0 else ''

            post_card = f"""
            <div class="reddit-post-card positive-post">
                <div class="post-title">
                    👍 {post['title']}
                    {quality_badge}
                </div>
                <div class="post-content">
                    {post.get('text', post.get('summary', 'Content not available'))}
                </div>
                <div class="post-meta">
                    📍 r/{post['subreddit']} • 👤 u/{post['author']} • ⬆️ {post['score']} upvotes • 
                    🎯 Relevancy: {post.get('relevancy_score', 0):.2f}
                    {f"• 🧠 Sentiment: {enhanced_sentiment.get('sentiment_score', 0):.2f}" if enhanced_sentiment else ""}
                    <br>
                    {f"📍 Mentions: {', '.join(entities.get('locations', [])[:3])}" if entities.get('locations') else ""}
                    <br>
                    <a href="{reddit_url}" target="_blank" class="reddit-link">
                        🔗 View Full Reddit Thread & All Comments
                    </a>
                </div>
            </div>
            """
            st.markdown(post_card, unsafe_allow_html=True)
    else:
        st.info("No positive community recommendations found for this location.")

    # Negative experiences
    if sentiment_analysis.get('negative'):
        st.markdown('<div class="section-header">⚠️ Places to Avoid / Cons</div>', unsafe_allow_html=True)

        for i, post in enumerate(sentiment_analysis['negative'], 1):
            post_url = post.get('url', '')
            if post_url.startswith('/r/'):
                reddit_url = f"https://reddit.com{post_url}"
            elif post_url.startswith('https://'):
                reddit_url = post_url
            else:
                reddit_url = f"https://reddit.com/r/{post.get('subreddit', 'travel')}/comments/{post.get('id', '')}"

            enhanced_sentiment = post.get('enhanced_sentiment', {})
            entities = post.get('entities', {})

            confidence = enhanced_sentiment.get('confidence', 0)
            quality_badge = f'<span class="quality-badge">Quality: {confidence:.1f}</span>' if confidence > 0 else ''

            post_card = f"""
            <div class="reddit-post-card negative-post">
                <div class="post-title">
                    ⚠️ {post['title']}
                    {quality_badge}
                </div>
                <div class="post-content">
                    {post.get('text', post.get('summary', 'Content not available'))}
                </div>
                <div class="post-meta">
                    📍 r/{post['subreddit']} • 👤 u/{post['author']} • ⬆️ {post['score']} upvotes • 
                    🎯 Relevancy: {post.get('relevancy_score', 0):.2f}
                    {f"• 🧠 Sentiment: {enhanced_sentiment.get('sentiment_score', 0):.2f}" if enhanced_sentiment else ""}
                    <br>
                    {f"📍 Mentions: {', '.join(entities.get('locations', [])[:3])}" if entities.get('locations') else ""}
                    <br>
                    <a href="{reddit_url}" target="_blank" class="reddit-link negative-reddit-link">
                        🔗 View Full Reddit Thread & All Comments
                    </a>
                </div>
            </div>
            """
            st.markdown(post_card, unsafe_allow_html=True)
    else:
        st.info("No negative experiences or warnings found for this location.")


def display_cost_estimates(cost_data: Dict[str, Any], location: str):
    """Display estimated costs for visiting the location."""
    st.markdown('<div class="section-header">💰 Estimated Costs</div>', unsafe_allow_html=True)

    # Main cost card
    cost_card = f"""
    <div class="expense-card">
        <h3>💸 {location} Daily Budget</h3>
        <h2>${cost_data['daily_total']:.0f} per day</h2>
        <p>3-day trip: ${cost_data['trip_3_days']:.0f} • 1-week trip: ${cost_data['trip_1_week']:.0f}</p>
    </div>
    """
    st.markdown(cost_card, unsafe_allow_html=True)

    # Detailed breakdown
    breakdown = cost_data['daily_breakdown']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Daily Cost Breakdown:**")
        for category, cost in breakdown.items():
            st.markdown(f"""
            <div class="expense-item">
                <span>{category.title()}</span>
                <span><strong>${cost}</strong></span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("**Trip Duration Costs:**")
        durations = [
            ('3 days', cost_data['trip_3_days']),
            ('1 week', cost_data['trip_1_week']),
            ('2 weeks', cost_data['trip_1_week'] * 2)
        ]

        for duration, total_cost in durations:
            st.markdown(f"""
            <div class="expense-item">
                <span>{duration}</span>
                <span><strong>${total_cost:.0f}</strong></span>
            </div>
            """, unsafe_allow_html=True)


def display_smart_trip_planner(location: str, reddit_posts: List[Dict], restaurants: List[Dict],
                               attractions: List[Dict]):
    """Display smart trip planner interface (100% free)."""

    st.markdown('<div class="smart-ai-section-header">🧠 Smart Trip Planner (Free AI)</div>', unsafe_allow_html=True)
    st.success("🆓 This Smart AI uses advanced algorithms + your Reddit data. No external APIs required!")

    # User preferences form
    with st.form("smart_trip_preferences"):
        st.markdown("**Customize Your Smart Trip:**")

        col1, col2 = st.columns(2)

        with col1:
            duration = st.selectbox("Trip Duration:", [2, 3, 4, 5, 7], index=1)
            budget = st.selectbox("Budget Level:", ["budget", "medium", "luxury"], index=1)

        with col2:
            interests = st.multiselect(
                "Interests:",
                ["food", "culture", "history", "art", "nightlife", "nature", "shopping", "architecture"],
                default=["food", "culture"]
            )
            travel_style = st.selectbox(
                "Travel Style:",
                ["relaxed", "balanced", "explorer", "luxury"],
                index=1
            )

        generate_button = st.form_submit_button("🎯 Generate Smart Itinerary", type="primary")

    if generate_button:
        with st.spinner("🧠 Smart AI is analyzing Reddit community data and creating your personalized itinerary..."):
            # Initialize smart planner
            smart_planner = SmartTripPlanner()

            # Prepare user preferences
            user_preferences = {
                'duration': duration,
                'budget': budget,
                'interests': interests,
                'travel_style': travel_style
            }

            # Generate smart itinerary
            smart_itinerary = smart_planner.generate_smart_itinerary(
                location=location,
                reddit_posts=reddit_posts,
                user_preferences=user_preferences,
                restaurants=restaurants,
                attractions=attractions
            )

            # Display the generated itinerary
            display_smart_itinerary(smart_itinerary)


def display_smart_itinerary(itinerary: Dict[str, Any]):
    """Display the smart-generated itinerary."""

    if not itinerary:
        st.error("Failed to generate itinerary. Please try again.")
        return

    # Header
    st.success("🎉 Your Personalized Smart Itinerary is Ready!")

    # Itinerary overview
    preferences = itinerary.get('user_preferences', {})

    # Build the overview HTML properly
    budget = preferences.get('budget', 'medium').title()
    style = preferences.get('travel_style', 'balanced').title()
    interests = ', '.join(preferences.get('interests', []))
    duration = preferences.get('duration', 3)
    method = itinerary.get('method', 'Smart Analysis')
    reddit_count = itinerary.get('reddit_insights_used', 0)

    overview_html = f"""
    <div class="smart-itinerary-card">
        <h3>🗺️ {itinerary['location']} - {duration} Day Smart Trip</h3>
        <p><strong>Budget:</strong> {budget} • 
           <strong>Style:</strong> {style} • 
           <strong>Interests:</strong> {interests}</p>
        <p><em>✨ {method} using {reddit_count} Reddit posts</em>
           <span class="method-badge">100% Free</span></p>
    </div>
    """
    st.markdown(overview_html, unsafe_allow_html=True)

    # Display each day
    for day_info in itinerary.get('days', []):
        day_html = f"""
        <div class="smart-day-section">
            <h4 style="color: #0d47a1; margin-bottom: 1rem;">
                📅 Day {day_info['day']}: {day_info.get('title', f'Day {day_info["day"]}')}
            </h4>
        </div>
        """
        st.markdown(day_html, unsafe_allow_html=True)

        # Display activities
        for activity in day_info.get('activities', []):
            activity_type = activity.get('type', 'activity')
            icon = {
                'restaurant': '🍽️',
                'attraction': '🏛️',
                'walking': '🚶',
                'shopping': '🛍️',
                'nightlife': '🌃',
                'hidden_gem': '💎',
                'activity': '🎯'
            }.get(activity_type, '📍')

            activity_html = f"""
            <div class="smart-activity-item">
                <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 0.5rem;">
                    <span style="background: #ff9800; color: white; padding: 4px 8px; 
                                border-radius: 15px; font-size: 0.8rem; font-weight: bold;">
                        {activity['time']}
                    </span>
                    <strong style="color: #0d47a1;">
                        {icon} {activity['activity']}
                    </strong>
                </div>
                <div style="color: #424242; line-height: 1.5; margin-left: 80px;">
                    {activity['description']}
                </div>
            </div>
            """
            st.markdown(activity_html, unsafe_allow_html=True)

    # Display Reddit tips
    if itinerary.get('reddit_tips'):
        st.markdown("### 💡 Community Tips from Reddit")
        for tip in itinerary['reddit_tips']:
            st.info(f"💬 {tip}")

    # Display budget notes
    if itinerary.get('budget_notes'):
        st.markdown("### 💰 Budget Insights from Community")
        for note in itinerary['budget_notes']:
            st.warning(f"💸 {note}")

    # Display hidden gems
    if itinerary.get('hidden_gems'):
        st.markdown("### 💎 Hidden Gems from Locals")
        for gem in itinerary['hidden_gems']:
            st.success(f"🗺️ {gem}")

def get_available_locations():
   """Get list of available locations."""
   return [
       "Paris", "London", "New York", "Tokyo", "Rome", "Barcelona",
       "Amsterdam", "Prague", "Vienna", "Berlin", "Istanbul", "Dubai",
       "Bangkok", "Singapore", "Hong Kong", "Sydney", "Los Angeles",
       "Chicago", "Las Vegas", "Miami", "San Francisco", "Venice",
       "Florence", "Athens", "Lisbon"
   ]

def main():
   """Main application function."""

   # Header with logo
   st.markdown("""
   <div style="position: relative;">
       <h1 class="main-header">🌍 Lifestyle Discovery Assistant</h1>
       <div class="nomad-logo">✈️ Nomad AI</div>
   </div>
   """, unsafe_allow_html=True)

   st.markdown('<p class="main-content-description"><strong>Discover Amazing Travel Destinations, Restaurants, and Events from Real Community Experiences</strong></p>', unsafe_allow_html=True)

   # Load data
   summary_data = load_extraction_summary()
   available_locations = get_available_locations()

   # Sidebar for location selection
   with st.sidebar:
       st.header("🎯 Destination Explorer")

       # Location selection
       selected_location = st.selectbox(
           "Choose your destination:",
           available_locations,
           index=0 if available_locations else None
       )

       st.subheader("Display Options")

       show_restaurants = st.checkbox("🍽️ Show Restaurants", value=True)
       show_attractions = st.checkbox("🏛️ Show Attractions", value=True)
       show_reddit_posts = st.checkbox("📝 Show Reddit Insights", value=True)
       show_costs = st.checkbox("💰 Show Cost Estimates", value=True)

       st.subheader("AI Features")
       show_smart_planner = st.checkbox("🧠 Smart Trip Planner (Free)", value=False)
       enhance_data_quality = st.checkbox("⚡ Enhanced Data Quality", value=True)

       # Show location statistics
       if summary_data and 'by_location' in summary_data and selected_location in summary_data['by_location']:
           location_stats = summary_data['by_location'][selected_location]
           st.subheader(f"📊 {selected_location} Data")

           total_posts = sum(location_stats.values())
           st.metric("Community Posts", total_posts)
           st.metric("Travel Posts", location_stats.get('travel', 0))
           st.metric("Food Posts", location_stats.get('food', 0))

           # Show data quality info
           if enhance_data_quality:
               st.caption("⚡ Enhanced with ML sentiment analysis & NER")

   # Main content area
   if selected_location:

       # Load data
       with st.spinner(f"Loading comprehensive data for {selected_location}..."):
           # Get Google Places data
           places_data = get_google_places_data(selected_location)

           # Get Reddit data
           travel_posts = load_location_data(selected_location, 'travel')
           food_posts = load_location_data(selected_location, 'food')
           all_reddit_posts = travel_posts + food_posts

           # Apply data quality enhancements
           if enhance_data_quality and all_reddit_posts:
               enhanced_analysis = enhance_reddit_data(all_reddit_posts, selected_location)
               sentiment_analysis = {
                   'positive': enhanced_analysis['positive'],
                   'negative': enhanced_analysis['negative']
               }
               st.success(f"⚡ Enhanced {len(all_reddit_posts)} posts with ML sentiment analysis")
           else:
               # Basic sentiment analysis fallback
               sentiment_analysis = {'positive': [], 'negative': []}

           # Estimate costs
           cost_estimates = estimate_costs(selected_location, places_data['restaurants'], places_data['attractions'])

       # Display sections based on user selection

       # Restaurants section
       if show_restaurants:
           display_restaurants(places_data['restaurants'], selected_location)

       # Attractions section
       if show_attractions:
           display_attractions(places_data['attractions'], selected_location)

       # Cost estimates section
       if show_costs:
           display_cost_estimates(cost_estimates, selected_location)

       # Reddit insights section
       if show_reddit_posts:
           display_reddit_insights(sentiment_analysis, selected_location)

       # Smart Trip Planner section
       if show_smart_planner:
           display_smart_trip_planner(selected_location, all_reddit_posts, places_data['restaurants'], places_data['attractions'])

   # Analytics Dashboard
   if summary_data:
       st.markdown('<h2 style="color: #ffffff; margin-top: 3rem;">📊 Global Community Insights</h2>', unsafe_allow_html=True)

       col1, col2, col3, col4 = st.columns(4)

       with col1:
           total_posts = summary_data.get('total_posts', 0)
           st.markdown(f"""
           <div style="background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%); 
                       color: white; padding: 1rem; border-radius: 0.5rem; text-align: center;">
               <h3>{total_posts:,}</h3>
               <p>Total Posts Analyzed</p>
           </div>
           """, unsafe_allow_html=True)

       with col2:
           destinations = summary_data.get('destinations_processed', 0)
           st.markdown(f"""
           <div style="background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%); 
                       color: white; padding: 1rem; border-radius: 0.5rem; text-align: center;">
               <h3>{destinations}</h3>
               <p>Destinations Covered</p>
           </div>
           """, unsafe_allow_html=True)

       with col3:
           travel_posts = summary_data.get('by_category', {}).get('travel', 0)
           st.markdown(f"""
           <div style="background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%); 
                       color: white; padding: 1rem; border-radius: 0.5rem; text-align: center;">
               <h3>{travel_posts:,}</h3>
               <p>Travel Insights</p>
           </div>
           """, unsafe_allow_html=True)

       with col4:
           food_posts = summary_data.get('by_category', {}).get('food', 0)
           st.markdown(f"""
           <div style="background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%); 
                       color: white; padding: 1rem; border-radius: 0.5rem; text-align: center;">
               <h3>{food_posts:,}</h3>
               <p>Food Recommendations</p>
           </div>
           """, unsafe_allow_html=True)

       # Charts section
       st.markdown("### 📈 Data Analytics")

       col1, col2 = st.columns(2)

       with col1:
           if 'by_category' in summary_data:
               fig_cat = px.pie(
                   values=list(summary_data['by_category'].values()),
                   names=list(summary_data['by_category'].keys()),
                   title="Community Data by Category",
                   color_discrete_map={
                       'travel': '#4a90e2',
                       'food': '#ff6b6b',
                       'events': '#4ecdc4'
                   }
               )
               fig_cat.update_layout(
                   plot_bgcolor='white',
                   paper_bgcolor='white',
                   font=dict(color='#333333')
               )
               st.plotly_chart(fig_cat, use_container_width=True)

       with col2:
           if 'by_location' in summary_data:
               # Create location chart
               location_data = []
               for location, categories in summary_data['by_location'].items():
                   total_posts = sum(categories.values())
                   location_data.append({
                       'Location': location,
                       'Total Posts': total_posts,
                       'Travel': categories.get('travel', 0),
                       'Food': categories.get('food', 0),
                       'Events': categories.get('events', 0)
                   })

               df = pd.DataFrame(location_data)
               df = df.sort_values('Total Posts', ascending=False).head(10)

               fig_loc = px.bar(
                   df,
                   x='Location',
                   y=['Travel', 'Food', 'Events'],
                   title="Top 10 Destinations by Community Posts",
                   color_discrete_map={
                       'Travel': '#4a90e2',
                       'Food': '#ff6b6b',
                       'Events': '#4ecdc4'
                   }
               )
               fig_loc.update_layout(
                   xaxis_tickangle=-45,
                   plot_bgcolor='white',
                   paper_bgcolor='white',
                   font=dict(color='#333333')
               )
               st.plotly_chart(fig_loc, use_container_width=True)


if __name__ == "__main__":
   main()