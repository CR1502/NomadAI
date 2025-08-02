"""
Nomad AI - Lifestyle Discovery Assistant
Complete single-file application for Streamlit Cloud deployment.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Any

# Import optional dependencies with error handling
try:
    import boto3
    from botocore.exceptions import ClientError
    HAS_AWS = True
except ImportError:
    HAS_AWS = False

try:
    import googlemaps
    HAS_GOOGLE_MAPS = True
except ImportError:
    HAS_GOOGLE_MAPS = False

try:
    import praw
    HAS_REDDIT = True
except ImportError:
    HAS_REDDIT = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

# Page configuration
st.set_page_config(
    page_title="Nomad AI - Lifestyle Discovery",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
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
        display: inline-block;
        margin-top: 10px;
        border: 1px solid rgba(74, 144, 226, 0.3);
        transition: all 0.3s ease;
    }
    
    .reddit-link:hover {
        background: rgba(74, 144, 226, 0.2);
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(74, 144, 226, 0.3);
        text-decoration: none;
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
    
    .expense-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white !important;
        padding: 1.2rem;
        border-radius: 0.8rem;
        text-align: center;
        margin-bottom: 1rem;
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
    
    .data-source-badge {
        background: #9c27b0;
        color: white !important;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 8px;
    }
    
    .comment-card {
        background: #ffffff;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #4caf50;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .comment-meta {
        color: #666666 !important;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .comment-text {
        color: #333333 !important;
        line-height: 1.4;
        font-size: 0.9rem;
    }
    
    /* Sidebar styling */
    .stSelectbox label, .stCheckbox label, .stSubheader {
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
    
    .metric-card {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        color: white !important;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .status-indicator {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 8px;
    }
    
    .status-connected {
        background: #4caf50;
        color: white;
    }
    
    .status-missing {
        background: #ff9800;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
DESTINATIONS = [
    "Paris", "London", "New York", "Tokyo", "Rome", "Barcelona",
    "Amsterdam", "Berlin", "Prague", "Vienna", "Dubai", "Bangkok",
    "Singapore", "Sydney", "Los Angeles", "Miami", "Venice",
    "Florence", "Athens", "Lisbon", "Istanbul", "Chicago", "Las Vegas"
]

# Cost data for all destinations
COST_DATA = {
    "Paris": {"daily": 235, "accommodation": 120, "food": 60, "transport": 25, "attractions": 30},
    "London": {"daily": 285, "accommodation": 150, "food": 70, "transport": 30, "attractions": 35},
    "Tokyo": {"daily": 195, "accommodation": 100, "food": 50, "transport": 20, "attractions": 25},
    "New York": {"daily": 335, "accommodation": 180, "food": 80, "transport": 35, "attractions": 40},
    "Rome": {"daily": 180, "accommodation": 90, "food": 45, "transport": 20, "attractions": 25},
    "Barcelona": {"daily": 165, "accommodation": 85, "food": 40, "transport": 18, "attractions": 22},
    "Amsterdam": {"daily": 218, "accommodation": 110, "food": 55, "transport": 25, "attractions": 28},
    "Berlin": {"daily": 140, "accommodation": 70, "food": 35, "transport": 15, "attractions": 20},
    "Prague": {"daily": 102, "accommodation": 50, "food": 25, "transport": 12, "attractions": 15},
    "Vienna": {"daily": 158, "accommodation": 80, "food": 40, "transport": 18, "attractions": 22},
    "Dubai": {"daily": 280, "accommodation": 130, "food": 65, "transport": 25, "attractions": 60},
    "Bangkok": {"daily": 73, "accommodation": 40, "food": 15, "transport": 8, "attractions": 10},
    "Singapore": {"daily": 185, "accommodation": 95, "food": 45, "transport": 20, "attractions": 25},
    "Sydney": {"daily": 285, "accommodation": 130, "food": 65, "transport": 30, "attractions": 60},
    "Istanbul": {"daily": 125, "accommodation": 45, "food": 20, "transport": 8, "attractions": 12},
    "Athens": {"daily": 118, "accommodation": 60, "food": 30, "transport": 12, "attractions": 18},
    "Lisbon": {"daily": 135, "accommodation": 70, "food": 35, "transport": 15, "attractions": 20},
    "Venice": {"daily": 220, "accommodation": 110, "food": 50, "transport": 20, "attractions": 25},
    "Florence": {"daily": 185, "accommodation": 95, "food": 45, "transport": 15, "attractions": 22},
    "Chicago": {"daily": 245, "accommodation": 120, "food": 60, "transport": 25, "attractions": 30},
    "Las Vegas": {"daily": 210, "accommodation": 90, "food": 60, "transport": 20, "attractions": 50},
    "Los Angeles": {"daily": 275, "accommodation": 140, "food": 70, "transport": 40, "attractions": 35},
    "Miami": {"daily": 260, "accommodation": 130, "food": 65, "transport": 25, "attractions": 30},
    "San Francisco": {"daily": 315, "accommodation": 160, "food": 75, "transport": 30, "attractions": 35},
}

def get_environment_value(key: str) -> str:
    """Get environment variable from Streamlit secrets or docker/.env."""

    # Try Streamlit secrets first (for cloud deployment)
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    # Try loading from docker/.env (for local development)
    if HAS_DOTENV:
        try:
            load_dotenv('docker/.env')
            value = os.getenv(key)
            if value:
                return value
        except Exception:
            pass

    # Fallback to regular environment variables
    return os.getenv(key, "")

def check_api_connections():
    """Check if API keys are properly loaded."""

    api_status = {}

    # Check Reddit API
    reddit_id = get_environment_value("REDDIT_CLIENT_ID")
    reddit_secret = get_environment_value("REDDIT_CLIENT_SECRET")
    api_status['reddit'] = bool(reddit_id and reddit_secret and HAS_REDDIT)

    # Check Google Places API
    google_key = get_environment_value("GOOGLE_PLACES_API_KEY")
    api_status['google'] = bool(google_key and HAS_GOOGLE_MAPS)

    # Check AWS S3
    aws_key = get_environment_value("AWS_ACCESS_KEY_ID")
    aws_secret = get_environment_value("AWS_SECRET_ACCESS_KEY")
    api_status['aws'] = bool(aws_key and aws_secret and HAS_AWS)

    return api_status

@st.cache_resource
def get_s3_client():
    """Initialize S3 client."""
    if not HAS_AWS:
        return None

    aws_key = get_environment_value("AWS_ACCESS_KEY_ID")
    aws_secret = get_environment_value("AWS_SECRET_ACCESS_KEY")

    if not aws_key or not aws_secret:
        return None

    try:
        return boto3.client(
            's3',
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            region_name=get_environment_value("AWS_DEFAULT_REGION") or 'us-east-1'
        )
    except Exception as e:
        st.error(f"S3 client initialization failed: {e}")
        return None

@st.cache_resource
def get_reddit_client():
    """Initialize Reddit client."""
    if not HAS_REDDIT:
        return None

    client_id = get_environment_value("REDDIT_CLIENT_ID")
    client_secret = get_environment_value("REDDIT_CLIENT_SECRET")
    user_agent = get_environment_value("REDDIT_USER_AGENT") or "nomad_ai_lifestyle_discovery_v1.0"

    if not client_id or not client_secret:
        return None

    try:
        return praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    except Exception as e:
        st.error(f"Reddit client initialization failed: {e}")
        return None

@st.cache_resource
def get_google_places_client():
    """Initialize Google Places client."""
    if not HAS_GOOGLE_MAPS:
        return None

    api_key = get_environment_value("GOOGLE_PLACES_API_KEY")

    if not api_key:
        return None

    try:
        return googlemaps.Client(key=api_key)
    except Exception as e:
        st.error(f"Google Places client initialization failed: {e}")
        return None

@st.cache_data
def get_google_places_data(location: str):
    """Get real Google Places data."""
    gmaps = get_google_places_client()

    if not gmaps:
        return None

    try:
        # Geocode the location
        geocode = gmaps.geocode(location)

        if not geocode:
            st.warning(f"Could not find coordinates for {location}")
            return None

        lat_lng = geocode[0]['geometry']['location']

        # Get restaurants
        restaurants_result = gmaps.places_nearby(
            location=lat_lng,
            radius=8000,
            type='restaurant',
            language='en'
        )

        # Get attractions
        attractions_result = gmaps.places_nearby(
            location=lat_lng,
            radius=8000,
            type='tourist_attraction',
            language='en'
        )

        # Process restaurants with detailed information
        restaurants = []
        for place in restaurants_result.get('results', [])[:10]:
            try:
                details = gmaps.place(
                    place_id=place['place_id'],
                    fields=['name', 'rating', 'user_ratings_total', 'price_level',
                           'formatted_address', 'website', 'formatted_phone_number']
                )

                detail_info = details.get('result', {})
                price_level = place.get('price_level', 2)
                price_symbols = ['$', '$', '$$', '$$$', '$$$$'][min(price_level, 4)]

                restaurants.append({
                    'name': place['name'],
                    'rating': detail_info.get('rating', place.get('rating', 4.0)),
                    'user_ratings_total': detail_info.get('user_ratings_total', 0),
                    'price': price_symbols,
                    'price_level': price_level,
                    'address': detail_info.get('formatted_address', place.get('vicinity', 'Address not available')),
                    'website': detail_info.get('website', 'Not available'),
                    'phone': detail_info.get('formatted_phone_number', 'Not available')
                })
            except Exception:
                continue

        # Process attractions with detailed information
        attractions = []
        for place in attractions_result.get('results', [])[:10]:
            try:
                details = gmaps.place(
                    place_id=place['place_id'],
                    fields=['name', 'rating', 'user_ratings_total', 'formatted_address',
                           'website', 'formatted_phone_number']
                )

                detail_info = details.get('result', {})

                attractions.append({
                    'name': place['name'],
                    'rating': detail_info.get('rating', place.get('rating', 4.0)),
                    'user_ratings_total': detail_info.get('user_ratings_total', 0),
                    'address': detail_info.get('formatted_address', place.get('vicinity', 'Address not available')),
                    'website': detail_info.get('website', 'Not available'),
                    'phone': detail_info.get('formatted_phone_number', 'Not available')
                })
            except Exception:
                continue

        return {'restaurants': restaurants, 'attractions': attractions}

    except Exception as e:
        st.error(f"Google Places API error: {e}")
        return None

@st.cache_data
def load_reddit_data_from_s3(location: str):
    """Load real Reddit data from S3 storage."""
    s3_client = get_s3_client()
    if not s3_client:
        return []

    bucket_name = get_environment_value("S3_BUCKET_NAME")
    if not bucket_name:
        return []

    all_posts = []

    # Load travel and food data
    for category in ['travel', 'food', 'events']:
        try:
            s3_key = f"{location.lower().replace(' ', '_')}/{category}/reddit_posts.json"
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            posts = json.loads(response['Body'].read().decode('utf-8'))
            all_posts.extend(posts)
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchKey':
                st.warning(f"Error loading {category} data for {location}")
        except Exception:
            continue

    return all_posts

def extract_fresh_reddit_data(location: str, max_posts: int = 30):
    """Extract fresh Reddit data for a location using real Reddit API."""
    reddit_client = get_reddit_client()
    if not reddit_client:
        return []

    posts = []
    subreddits = ['travel', 'solotravel', 'backpacking', 'food', 'AskCulinary', 'streetfood']

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        for idx, subreddit_name in enumerate(subreddits):
            status_text.text(f"Searching r/{subreddit_name} for {location} posts...")

            try:
                subreddit = reddit_client.subreddit(subreddit_name)

                # Search for location mentions
                search_results = list(subreddit.search(location, limit=max_posts//len(subreddits)))

                for submission in search_results:
                    full_text = f"{submission.title} {submission.selftext}".lower()

                    # Check if location is actually mentioned meaningfully
                    if location.lower() not in full_text:
                        continue

                    # Calculate relevancy score
                    location_mentions = full_text.count(location.lower())
                    relevancy = min(location_mentions * 0.2, 0.8)

                    # Bonus for travel context
                    travel_context = ['visit', 'trip', 'travel', 'went', 'been to', 'staying', 'vacation']
                    if any(word in full_text for word in travel_context):
                        relevancy += 0.3

                    if relevancy < 0.3:
                        continue

                    # Extract top comments
                    comments = []
                    try:
                        submission.comments.replace_more(limit=0)
                        all_comments = submission.comments.list()

                        # Filter and sort comments
                        valid_comments = [c for c in all_comments
                                        if hasattr(c, 'body') and hasattr(c, 'score')
                                        and c.score > 1 and len(c.body) > 20]
                        valid_comments.sort(key=lambda x: x.score, reverse=True)

                        for comment in valid_comments[:3]:
                            comments.append({
                                'author': str(comment.author) if comment.author else 'deleted',
                                'body': comment.body[:400] if len(comment.body) > 400 else comment.body,
                                'score': comment.score,
                                'created_utc': comment.created_utc
                            })
                    except Exception:
                        pass

                    # Create comprehensive post data
                    post_text = submission.selftext or submission.title
                    summary = post_text[:300] + '...' if len(post_text) > 300 else post_text

                    post_data = {
                        'id': submission.id,
                        'title': submission.title,
                        'text': post_text,
                        'summary': summary,
                        'subreddit': subreddit_name,
                        'author': str(submission.author) if submission.author else 'deleted',
                        'score': submission.score,
                        'num_comments': submission.num_comments,
                        'url': f"https://reddit.com{submission.permalink}",
                        'relevancy_score': relevancy,
                        'top_comments': comments,
                        'target_location': location,
                        'timestamp': datetime.fromtimestamp(submission.created_utc).isoformat(),
                        'category': 'travel' if subreddit_name in ['travel', 'solotravel', 'backpacking'] else 'food'
                    }

                    posts.append(post_data)

                    # Rate limiting
                    time.sleep(0.3)

            except Exception as e:
                st.warning(f"Error searching r/{subreddit_name}: {e}")
                continue

            # Update progress
            progress_bar.progress((idx + 1) / len(subreddits))
            time.sleep(1)  # Rate limiting between subreddits

    except Exception as e:
        st.error(f"Error extracting Reddit data: {e}")
    finally:
        progress_bar.empty()
        status_text.empty()

    # Sort by relevancy and score
    posts.sort(key=lambda x: (x.get('relevancy_score', 0), x.get('score', 0)), reverse=True)

    return posts

def analyze_reddit_sentiment(posts: List[Dict]) -> Dict[str, List[Dict]]:
    """Advanced sentiment analysis of Reddit posts."""
    positive_posts = []
    negative_posts = []

    # Enhanced sentiment indicators
    positive_indicators = [
        'amazing', 'incredible', 'fantastic', 'love', 'perfect', 'recommend',
        'beautiful', 'wonderful', 'excellent', 'must visit', 'hidden gem',
        'authentic', 'worth it', 'favorite', 'brilliant', 'stunning',
        'unforgettable', 'magical', 'breathtaking', 'outstanding'
    ]

    negative_indicators = [
        'avoid', 'terrible', 'worst', 'disappointing', 'overrated',
        'tourist trap', 'waste of money', 'not worth', 'skip', 'bad',
        'awful', 'overpriced', 'crowded', 'dirty', 'rude', 'scam',
        'boring', 'mediocre', 'expensive', 'poor service'
    ]

    for post in posts:
        text = f"{post.get('title', '')} {post.get('text', '')} {post.get('summary', '')}".lower()

        # TextBlob sentiment analysis if available
        sentiment_score = 0
        if HAS_TEXTBLOB:
            try:
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity
            except:
                sentiment_score = 0

        # Count sentiment indicators
        pos_count = sum(1 for word in positive_indicators if word in text)
        neg_count = sum(1 for word in negative_indicators if word in text)

        # Enhanced classification with multiple criteria
        if (sentiment_score > 0.1 and pos_count > neg_count) or pos_count >= 2:
            post['sentiment_score'] = sentiment_score
            post['positive_indicators'] = pos_count
            post['sentiment_confidence'] = abs(sentiment_score) + (pos_count * 0.1)
            positive_posts.append(post)
        elif (sentiment_score < -0.1 and neg_count > pos_count) or neg_count >= 1:
            post['sentiment_score'] = sentiment_score
            post['negative_indicators'] = neg_count
            post['sentiment_confidence'] = abs(sentiment_score) + (neg_count * 0.1)
            negative_posts.append(post)

    # Sort by combined relevancy, score, and sentiment confidence
    positive_posts.sort(
        key=lambda x: (
            x.get('relevancy_score', 0) * 100 +
            x.get('score', 0) +
            x.get('sentiment_confidence', 0) * 50
        ),
        reverse=True
    )

    negative_posts.sort(
        key=lambda x: (
            x.get('relevancy_score', 0) * 100 +
            x.get('score', 0) +
            x.get('sentiment_confidence', 0) * 50
        ),
        reverse=True
    )

    return {
        'positive': positive_posts[:5],
        'negative': negative_posts[:3]
    }

def display_restaurants(restaurants: List[Dict], location: str, data_source: str = "Google Places"):
    """Display restaurant recommendations with enhanced details."""
    st.markdown(f'<div class="section-header">🍽️ Top Restaurants <span class="data-source-badge">{data_source}</span></div>', unsafe_allow_html=True)

    if not restaurants:
        st.warning("No restaurant data available for this location.")
        return

    for i, restaurant in enumerate(restaurants, 1):
        rating = restaurant.get('rating', 4.0)
        price = restaurant.get('price', '$$')
        address = restaurant.get('address', 'Address not available')
        website = restaurant.get('website', 'Not available')
        phone = restaurant.get('phone', 'Not available')
        total_ratings = restaurant.get('user_ratings_total', 0)

        # Format website display
        website_display = website
        if website != 'Not available' and len(website) > 50:
            website_display = website[:47] + "..."

        st.markdown(f"""
        <div class="recommendation-card">
            <div class="place-name">{i}. {restaurant['name']}</div>
            <div class="place-details">
                <span class="rating-badge">⭐ {rating:.1f}</span>
                <span class="price-badge">{price}</span>
                {f'<small style="color: #666;">({total_ratings:,} reviews)</small>' if total_ratings > 0 else ''}<br><br>
                <strong>📍 Address:</strong> {address}<br>
                <strong>📞 Phone:</strong> {phone}<br>
                <strong>🌐 Website:</strong> {website_display}
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_attractions(attractions: List[Dict], location: str, data_source: str = "Google Places"):
    """Display attraction recommendations with enhanced details."""
    st.markdown(f'<div class="section-header">🏛️ Top Attractions <span class="data-source-badge">{data_source}</span></div>', unsafe_allow_html=True)

    if not attractions:
        st.warning("No attraction data available for this location.")
        return

    for i, attraction in enumerate(attractions, 1):
        rating = attraction.get('rating', 4.0)
        address = attraction.get('address', 'Address not available')
        website = attraction.get('website', 'Not available')
        phone = attraction.get('phone', 'Not available')
        total_ratings = attraction.get('user_ratings_total', 0)

        # Format website display
        website_display = website
        if website != 'Not available' and len(website) > 50:
            website_display = website[:47] + "..."

        st.markdown(f"""
        <div class="recommendation-card">
            <div class="place-name">{i}. {attraction['name']}</div>
            <div class="place-details">
                <span class="rating-badge">⭐ {rating:.1f}</span>
                {f'<small style="color: #666;">({total_ratings:,} reviews)</small>' if total_ratings > 0 else ''}<br><br>
                <strong>📍 Address:</strong> {address}<br>
                <strong>📞 Phone:</strong> {phone}<br>
                <strong>🌐 Website:</strong> {website_display}
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_reddit_insights(sentiment_data: Dict[str, List[Dict]], location: str, data_source: str = "Reddit API"):
    """Display comprehensive Reddit community insights."""

    # Positive posts
    if sentiment_data.get('positive'):
        st.markdown(f'<div class="section-header">✅ Community Favorites <span class="data-source-badge">{data_source}</span></div>', unsafe_allow_html=True)

        for i, post in enumerate(sentiment_data['positive'], 1):
            reddit_url = post.get('url', '#')

            st.markdown(f"""
            <div class="reddit-post-card positive-post">
                <div class="post-title">👍 {post['title']}</div>
                <div class="post-content">{post.get('text', post.get('summary', ''))}</div>
                <div class="post-meta">
                    📍 r/{post['subreddit']} • 👤 u/{post['author']} • ⬆️ {post['score']} upvotes • 
                    🎯 Relevancy: {post.get('relevancy_score', 0):.2f} • 
                    💭 {post.get('num_comments', 0)} comments<br>
                    {f"🧠 Sentiment: {post.get('sentiment_score', 0):.2f}" if post.get('sentiment_score') else ''}
                    <br>
                    <a href="{reddit_url}" target="_blank" class="reddit-link">
                        🔗 View Full Reddit Thread & All Comments
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Display top comments if available
            if post.get('top_comments'):
                st.markdown("**💬 Top Community Responses:**")
                for comment in post['top_comments'][:2]:
                    st.markdown(f"""
                    <div class="comment-card">
                        <div class="comment-meta">
                            <strong>u/{comment['author']}</strong> • ⬆️ {comment['score']} upvotes
                        </div>
                        <div class="comment-text">
                            {comment['body']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("No positive community recommendations found. Try extracting fresh data or check API connections.")

    # Negative posts
    if sentiment_data.get('negative'):
        st.markdown('<div class="section-header">⚠️ Things to Consider</div>', unsafe_allow_html=True)

        for i, post in enumerate(sentiment_data['negative'], 1):
            reddit_url = post.get('url', '#')

            st.markdown(f"""
            <div class="reddit-post-card negative-post">
                <div class="post-title">⚠️ {post['title']}</div>
                <div class="post-content">{post.get('text', post.get('summary', ''))}</div>
                <div class="post-meta">
                    📍 r/{post['subreddit']} • 👤 u/{post['author']} • ⬆️ {post['score']} upvotes • 
                    🎯 Relevancy: {post.get('relevancy_score', 0):.2f} • 
                    💭 {post.get('num_comments', 0)} comments<br>
                    {f"🧠 Sentiment: {post.get('sentiment_score', 0):.2f}" if post.get('sentiment_score') else ''}
                    <br>
                    <a href="{reddit_url}" target="_blank" class="reddit-link negative-reddit-link">
                        🔗 View Full Reddit Thread & All Comments
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Display top comments for negative posts too
            if post.get('top_comments'):
                st.markdown("**💬 Community Discussion:**")
                for comment in post['top_comments'][:2]:
                    st.markdown(f"""
                    <div class="comment-card">
                        <div class="comment-meta">
                            <strong>u/{comment['author']}</strong> • ⬆️ {comment['score']} upvotes
                        </div>
                        <div class="comment-text">
                            {comment['body']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

def display_costs(location: str):
    """Display comprehensive cost estimates."""
    st.markdown('<div class="section-header">💰 Estimated Costs</div>', unsafe_allow_html=True)

    default_costs = {"daily": 200, "accommodation": 100, "food": 50, "transport": 20, "attractions": 30}
    costs = COST_DATA.get(location, default_costs)

    trip_3_days = costs['daily'] * 3
    trip_1_week = costs['daily'] * 7
    trip_2_weeks = costs['daily'] * 14
    trip_1_month = costs['daily'] * 30

    st.markdown(f"""
    <div class="expense-card">
        <h3>💸 {location} Daily Budget</h3>
        <h2>${costs['daily']} per day</h2>
        <p>3-day trip: ${trip_3_days} • 1-week trip: ${trip_1_week}</p>
    </div>
    """, unsafe_allow_html=True)

    # Detailed breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Daily Cost Breakdown:**")
        breakdown_items = [
            ("🏨 Accommodation", costs.get('accommodation', costs['daily']//2)),
            ("🍽️ Food & Dining", costs.get('food', costs['daily']//4)),
            ("🚇 Transportation", costs.get('transport', 20)),
            ("🎫 Attractions & Activities", costs.get('attractions', costs['daily']//6))
        ]

        for item, cost in breakdown_items:
            st.markdown(f"""
            <div class="expense-item">
                <span>{item}</span>
                <span><strong>${cost}</strong></span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("**Trip Duration Costs:**")
        duration_costs = [
            ("3 days", trip_3_days),
            ("1 week", trip_1_week),
            ("2 weeks", trip_2_weeks),
            ("1 month", trip_1_month)
        ]

        for duration, cost in duration_costs:
            st.markdown(f"""
            <div class="expense-item">
                <span>{duration}</span>
                <span><strong>${cost:,}</strong></span>
            </div>
            """, unsafe_allow_html=True)

def load_extraction_summary():
    """Load extraction summary for analytics."""
    try:
        # Try S3 first
        s3_client = get_s3_client()
        if s3_client:
            bucket_name = get_environment_value("S3_BUCKET_NAME")
            if bucket_name:
                try:
                    response = s3_client.get_object(Bucket=bucket_name, Key='extraction_summary.json')
                    return json.loads(response['Body'].read().decode('utf-8'))
                except:
                    pass

        # Try local file
        if os.path.exists('data/summaries/extraction_summary.json'):
            with open('data/summaries/extraction_summary.json', 'r') as f:
                return json.load(f)
    except:
        pass

    return None

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

    # Check API connections
    api_status = check_api_connections()

    # Show dependency status
    if not all([HAS_AWS, HAS_GOOGLE_MAPS, HAS_REDDIT, HAS_TEXTBLOB]):
        missing_deps = []
        if not HAS_AWS: missing_deps.append("boto3")
        if not HAS_GOOGLE_MAPS: missing_deps.append("googlemaps")
        if not HAS_REDDIT: missing_deps.append("praw")
        if not HAS_TEXTBLOB: missing_deps.append("textblob")

        if missing_deps:
            st.warning(f"Some features may be limited. Missing dependencies: {', '.join(missing_deps)}")

    # Sidebar
    with st.sidebar:
        st.header("🎯 Destination Explorer")

        selected_location = st.selectbox(
            "Choose your destination:",
            DESTINATIONS,
            index=0
        )

        st.subheader("Display Options")
        show_restaurants = st.checkbox("🍽️ Show Restaurants", value=True)
        show_attractions = st.checkbox("🏛️ Show Attractions", value=True)
        show_reddit = st.checkbox("📝 Show Reddit Insights", value=True)
        show_costs = st.checkbox("💰 Show Cost Estimates", value=True)

        st.subheader("Data Options")

        # Reddit data options
        if api_status['reddit']:
            data_source_option = st.radio(
                "Reddit Data Source:",
                ["Use Stored Data", "Extract Fresh Data"],
                help="Stored data is faster, Fresh data is more current"
            )
            use_fresh_reddit = (data_source_option == "Extract Fresh Data")

            if use_fresh_reddit:
                max_posts = st.slider("Max posts to extract:", 10, 50, 20)
        else:
            use_fresh_reddit = False
            max_posts = 20

        # API status indicators
        st.subheader("📊 Data Sources")

        # Google Places status
        if api_status['google']:
            st.markdown('<span class="status-indicator status-connected">✅ Google Places API</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-missing">⚠️ Google Places API Missing</span>', unsafe_allow_html=True)

        # Reddit status
        if api_status['reddit']:
            st.markdown('<span class="status-indicator status-connected">✅ Reddit API</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-missing">⚠️ Reddit API Missing</span>', unsafe_allow_html=True)

        # AWS status
        if api_status['aws']:
            st.markdown('<span class="status-indicator status-connected">✅ AWS S3</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-missing">📝 Local Storage Only</span>', unsafe_allow_html=True)

        # Location statistics
        st.subheader(f"📈 {selected_location} Data")

        # Load and display summary statistics
        summary = load_extraction_summary()
        if summary and selected_location in summary.get('by_location', {}):
            location_stats = summary['by_location'][selected_location]
            st.metric("Travel Posts", location_stats.get('travel', 0))
            st.metric("Food Posts", location_stats.get('food', 0))
            st.metric("Events Posts", location_stats.get('events', 0))
            st.metric("Total Posts", sum(location_stats.values()))
        else:
            st.metric("Available Data", "Ready")
            if api_status['reddit']:
                st.caption("Extract fresh data to see statistics")

    # Main content area
    if selected_location:

        # Load all data with progress indicators
        with st.spinner(f"Loading comprehensive data for {selected_location}..."):

            # Get Google Places data
            places_data = None
            if api_status['google']:
                places_data = get_google_places_data(selected_location)
                data_source_places = "Live Google Places API"
            else:
                data_source_places = "API Key Required"

            # Get Reddit data
            reddit_posts = []
            data_source_reddit = "No Data"

            if show_reddit and api_status['reddit']:
                if use_fresh_reddit:
                    st.info("🔄 Extracting fresh Reddit data... This will take 1-2 minutes.")
                    reddit_posts = extract_fresh_reddit_data(selected_location, max_posts)
                    data_source_reddit = f"Fresh Reddit API ({len(reddit_posts)} posts)"
                else:
                    reddit_posts = load_reddit_data_from_s3(selected_location)
                    data_source_reddit = f"Stored Reddit Data ({len(reddit_posts)} posts)" if reddit_posts else "No Stored Data"

                # Analyze sentiment if we have posts
                if reddit_posts:
                    sentiment_analysis = analyze_reddit_sentiment(reddit_posts)
                else:
                    sentiment_analysis = {'positive': [], 'negative': []}
            else:
                sentiment_analysis = {'positive': [], 'negative': []}

        # Display sections based on user selection
        if show_restaurants:
            if places_data and places_data['restaurants']:
                display_restaurants(places_data['restaurants'], selected_location, data_source_places)
            else:
                st.warning("🔑 Google Places API key required to show real restaurant data")
                st.info("Add your Google Places API key in the Streamlit Cloud app settings to see verified restaurants with ratings, addresses, and contact information.")

        if show_attractions:
            if places_data and places_data['attractions']:
                display_attractions(places_data['attractions'], selected_location, data_source_places)
            else:
                st.warning("🔑 Google Places API key required to show real attraction data")
                st.info("Add your Google Places API key in the Streamlit Cloud app settings to see verified attractions with ratings and details.")

        if show_costs:
            display_costs(selected_location)

        if show_reddit:
            if api_status['reddit']:
                if reddit_posts:
                    display_reddit_insights(sentiment_analysis, selected_location, data_source_reddit)
                else:
                    st.warning("No Reddit data found for this location. Try extracting fresh data.")
                    if st.button("🔄 Extract Fresh Reddit Data"):
                        st.rerun()
            else:
                st.warning("🔑 Reddit API credentials required to show community insights")
                with st.expander("📖 How to get Reddit API credentials"):
                    st.info("""
                    **To get Reddit API credentials:**
                    1. Go to [reddit.com/prefs/apps](https://reddit.com/prefs/apps)
                    2. Click "Create App" or "Create Another App"
                    3. Choose "script" as the app type
                    4. Copy your Client ID and Client Secret
                    5. Add them to your Streamlit Cloud app settings
                    """)

    # Analytics dashboard
    st.markdown('<h2 style="color: #ffffff; margin-top: 3rem;">📊 Global Community Insights</h2>', unsafe_allow_html=True)

    # Load summary for global stats
    summary = load_extraction_summary()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_posts = summary.get('total_posts', 0) if summary else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_posts:,}</h3>
            <p>Total Posts</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        destinations_count = len(DESTINATIONS)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{destinations_count}</h3>
            <p>Destinations</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        travel_posts = summary.get('by_category', {}).get('travel', 0) if summary else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>{travel_posts:,}</h3>
            <p>Travel Posts</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        food_posts = summary.get('by_category', {}).get('food', 0) if summary else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>{food_posts:,}</h3>
            <p>Food Posts</p>
        </div>
        """, unsafe_allow_html=True)

    # Show data source summary
    st.markdown("### 🔧 Current Data Configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        google_status = "✅ Live Data" if api_status['google'] else "🔑 API Key Needed"
        st.info(f"**Restaurants & Attractions**\n{google_status}")
        if api_status['google']:
            st.caption("Showing verified Google Places data")

    with col2:
        reddit_status = "✅ Live Data" if api_status['reddit'] else "🔑 API Key Needed"
        st.info(f"**Community Insights**\n{reddit_status}")
        if api_status['reddit']:
            st.caption("Real Reddit posts with sentiment analysis")

    with col3:
        aws_status = "✅ Cloud Storage" if api_status['aws'] else "💾 Local Storage"
        st.info(f"**Data Storage**\n{aws_status}")
        if api_status['aws']:
            st.caption("Data stored in AWS S3")

    # Show global analytics if we have summary data
    if summary and 'by_location' in summary:
        st.markdown("### 📈 Top Destinations by Community Activity")

        # Create chart data
        location_data = []
        for location, categories in summary['by_location'].items():
            total_posts = sum(categories.values())
            if total_posts > 0:  # Only show locations with data
                location_data.append({
                    'Location': location,
                    'Total Posts': total_posts,
                    'Travel': categories.get('travel', 0),
                    'Food': categories.get('food', 0),
                    'Events': categories.get('events', 0)
                })

        if location_data:
            df = pd.DataFrame(location_data)
            df = df.sort_values('Total Posts', ascending=False).head(10)

            fig = px.bar(
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
            fig.update_layout(
                xaxis_tickangle=-45,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#333333'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    # Footer with app info
    st.markdown("---")
    st.markdown("### 🚀 About Nomad AI")
    st.info("""
    **Nomad AI Lifestyle Discovery Assistant** combines real community insights from Reddit with verified business data 
    from Google Places to provide authentic travel recommendations. Built with advanced sentiment analysis and 
    machine learning to surface the most valuable travel experiences shared by real travelers.
    """)


if __name__ == "__main__":
    main()