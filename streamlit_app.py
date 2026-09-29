"""
Nomad AI - Lifestyle Discovery Assistant
Streamlit dashboard backed by shared ingestion and content logic.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from math import log1p
from typing import Dict, List

from src.data_pipeline.storage import deduplicate_posts, load_location_posts, location_slug, validate_extraction_summary
from src.utils.content import calculate_costs, classify_sentiment, normalize_place, safe_html, safe_url
from src.utils.helpers import PROJECT_ROOT, load_config
from src.services.demo_data import demo_places
from src.web.trip_planner import render_trip_planner

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
APP_CONFIG = load_config()
DESTINATIONS = APP_CONFIG['destinations']
COST_DATA = {location: calculate_costs(costs) for location, costs in APP_CONFIG['costs'].items()}
if HAS_DOTENV:
    load_dotenv(PROJECT_ROOT / '.env')
    load_dotenv(PROJECT_ROOT / 'docker' / '.env')

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
            load_dotenv(PROJECT_ROOT / 'docker' / '.env')
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
    api_status['aws'] = bool(get_environment_value('S3_BUCKET_NAME') and HAS_AWS)

    return api_status

@st.cache_resource
def get_s3_client():
    """Initialize S3 client."""
    if not HAS_AWS:
        return None

    aws_key = get_environment_value("AWS_ACCESS_KEY_ID")
    aws_secret = get_environment_value("AWS_SECRET_ACCESS_KEY")

    if not get_environment_value('S3_BUCKET_NAME'):
        return None

    try:
        credentials = {}
        if aws_key and aws_secret:
            credentials = {'aws_access_key_id': aws_key, 'aws_secret_access_key': aws_secret}
            token = get_environment_value('AWS_SESSION_TOKEN')
            if token:
                credentials['aws_session_token'] = token
        return boto3.client(
            's3',
            region_name=get_environment_value("AWS_DEFAULT_REGION") or 'us-east-1',
            **credentials,
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

@st.cache_data(ttl=300)
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
                restaurants.append(normalize_place(place, detail_info))
            except Exception as error:
                st.warning(f"Could not load restaurant details for {place.get('name', 'a place')}: {error}")
                restaurants.append(normalize_place(place))

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

                attractions.append(normalize_place(place, detail_info))
            except Exception as error:
                st.warning(f"Could not load attraction details for {place.get('name', 'a place')}: {error}")
                attractions.append(normalize_place(place))

        return {'restaurants': restaurants, 'attractions': attractions}

    except Exception as e:
        st.error(f"Google Places API error: {e}")
        return None

@st.cache_data(ttl=300)
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
            s3_key = f"{location_slug(location)}/{category}/reddit_posts.json"
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            posts = json.loads(response['Body'].read().decode('utf-8'))
            if not isinstance(posts, list) or any(not isinstance(post, dict) for post in posts):
                raise ValueError('Stored posts must be a JSON list of post objects')
            all_posts.extend(deduplicate_posts(posts))
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchKey':
                st.warning(f"Error loading {category} data for {location}")
        except Exception as error:
            st.warning(f"Could not read stored {category} posts for {location}: {error}")

    return deduplicate_posts(all_posts)


def get_data_directory() -> Path:
    return Path(get_environment_value('NOMADAI_DATA_DIR') or PROJECT_ROOT / 'data')


@st.cache_data(ttl=300)
def load_stored_reddit_data(location: str, data_directory: str, demo_mode: bool = False):
    """Read S3 or local data without requiring a Reddit API connection."""
    if not demo_mode:
        posts = load_reddit_data_from_s3(location)
        if posts:
            return posts
    root = Path(data_directory) / 'demo' if demo_mode else Path(data_directory)
    try:
        return load_location_posts(location, root / 'by_location', include_demo=demo_mode)
    except (OSError, ValueError) as error:
        st.warning(f'Could not load local posts: {error}')
        return []

def extract_fresh_reddit_data(location: str, max_posts: int = 30):
    """Extract fresh Reddit data for a location using real Reddit API."""
    reddit_client = get_reddit_client()
    if not reddit_client:
        return []

    posts = []
    seen_ids = set()
    subreddits = ['travel', 'solotravel', 'backpacking', 'food', 'AskCulinary', 'streetfood']

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        for idx, subreddit_name in enumerate(subreddits):
            status_text.text(f"Searching r/{subreddit_name} for {location} posts...")

            try:
                subreddit = reddit_client.subreddit(subreddit_name)

                # Search for location mentions
                quota, remainder = divmod(max_posts, len(subreddits))
                search_results = list(subreddit.search(location, limit=quota + (idx < remainder)))

                for submission in search_results:
                    if submission.id in seen_ids:
                        continue
                    seen_ids.add(submission.id)
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
                        'relevancy_score': min(relevancy, 1.0),
                        'top_comments': comments,
                        'target_location': location,
                        'timestamp': datetime.fromtimestamp(submission.created_utc, timezone.utc).isoformat(),
                        'category': 'travel' if subreddit_name in ['travel', 'solotravel', 'backpacking'] else 'food',
                        'source': 'reddit',
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
    groups = {'positive': [], 'negative': [], 'neutral': []}
    for post in posts:
        text = f"{post.get('title', '')} {post.get('text') or post.get('summary', '')}"
        polarity = 0.0
        if HAS_TEXTBLOB:
            try:
                polarity = TextBlob(text).sentiment.polarity
            except Exception:
                pass
        sentiment = classify_sentiment(text, polarity)
        groups[sentiment['sentiment_label']].append({**post, **sentiment})

    def rank(post):
        engagement = min(log1p(max(post.get('score', 0), 0)) / log1p(1000), 1.0)
        return 0.6 * post.get('relevancy_score', 0) + 0.3 * engagement + 0.1 * abs(post['sentiment_score'])

    for group in groups.values():
        group.sort(key=rank, reverse=True)
    return {'positive': groups['positive'][:5], 'negative': groups['negative'][:3], 'neutral': groups['neutral'][:3]}

def display_restaurants(restaurants: List[Dict], location: str, data_source: str = "Google Places"):
    """Display restaurant recommendations with enhanced details."""
    st.markdown(f'<div class="section-header">🍽️ Restaurants <span class="data-source-badge">{safe_html(data_source)}</span></div>', unsafe_allow_html=True)

    if not restaurants:
        st.warning("No restaurant data available for this location.")
        return

    for i, restaurant in enumerate(restaurants, 1):
        rating = restaurant.get('rating')
        rating_display = f'⭐ {rating:.1f}' if rating is not None else 'Rating unavailable'
        price = safe_html(restaurant.get('price', 'Price unavailable'))
        address = safe_html(restaurant.get('address', 'Address not available'))
        website = safe_html(restaurant.get('website', 'Not available'))
        phone = safe_html(restaurant.get('phone', 'Not available'))
        total_ratings = restaurant.get('user_ratings_total') or 0

        # Format website display
        website_display = website
        if website != 'Not available' and len(website) > 50:
            website_display = website[:47] + "..."

        st.markdown(f"""
        <div class="recommendation-card">
            <div class="place-name">{i}. {safe_html(restaurant['name'])}</div>
            <div class="place-details">
                <span class="rating-badge">{rating_display}</span>
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
    st.markdown(f'<div class="section-header">🏛️ Attractions <span class="data-source-badge">{safe_html(data_source)}</span></div>', unsafe_allow_html=True)

    if not attractions:
        st.warning("No attraction data available for this location.")
        return

    for i, attraction in enumerate(attractions, 1):
        rating = attraction.get('rating')
        rating_display = f'⭐ {rating:.1f}' if rating is not None else 'Rating unavailable'
        address = safe_html(attraction.get('address', 'Address not available'))
        website = safe_html(attraction.get('website', 'Not available'))
        phone = safe_html(attraction.get('phone', 'Not available'))
        total_ratings = attraction.get('user_ratings_total') or 0

        # Format website display
        website_display = website
        if website != 'Not available' and len(website) > 50:
            website_display = website[:47] + "..."

        st.markdown(f"""
        <div class="recommendation-card">
            <div class="place-name">{i}. {safe_html(attraction['name'])}</div>
            <div class="place-details">
                <span class="rating-badge">{rating_display}</span>
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
        st.markdown(f'<div class="section-header">✅ Community Favorites <span class="data-source-badge">{safe_html(data_source)}</span></div>', unsafe_allow_html=True)

        for i, post in enumerate(sentiment_data['positive'], 1):
            reddit_url = safe_url(post.get('url'))

            st.markdown(f"""
            <div class="reddit-post-card positive-post">
                <div class="post-title">👍 {safe_html(post.get('title', 'Untitled'))}</div>
                <div class="post-content">{safe_html(post.get('text') or post.get('summary', ''))}</div>
                <div class="post-meta">
                    📍 r/{safe_html(post.get('subreddit', 'unknown'))} • 👤 u/{safe_html(post.get('author', 'deleted'))} • ⬆️ {safe_html(post.get('score', 0))} upvotes •
                    🎯 Relevancy: {post.get('relevancy_score', 0):.2f} • 
                    💭 {safe_html(post.get('num_comments', 0))} comments<br>
                    {f"🧠 Sentiment: {post.get('sentiment_score', 0):.2f}" if post.get('sentiment_score') else ''}
                    <br>
                    <a href="{reddit_url}" target="_blank" rel="noopener noreferrer" class="reddit-link">
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
                            <strong>u/{safe_html(comment.get('author', 'deleted'))}</strong> • ⬆️ {safe_html(comment.get('score', 0))} upvotes
                        </div>
                        <div class="comment-text">
                            {safe_html(comment.get('body', ''))}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("No positive community recommendations found. Try extracting fresh data or check API connections.")

    # Negative posts
    if sentiment_data.get('negative'):
        st.markdown('<div class="section-header">⚠️ Things to Consider</div>', unsafe_allow_html=True)

        for i, post in enumerate(sentiment_data['negative'], 1):
            reddit_url = safe_url(post.get('url'))

            st.markdown(f"""
            <div class="reddit-post-card negative-post">
                <div class="post-title">⚠️ {safe_html(post.get('title', 'Untitled'))}</div>
                <div class="post-content">{safe_html(post.get('text') or post.get('summary', ''))}</div>
                <div class="post-meta">
                    📍 r/{safe_html(post.get('subreddit', 'unknown'))} • 👤 u/{safe_html(post.get('author', 'deleted'))} • ⬆️ {safe_html(post.get('score', 0))} upvotes •
                    🎯 Relevancy: {post.get('relevancy_score', 0):.2f} • 
                    💭 {safe_html(post.get('num_comments', 0))} comments<br>
                    {f"🧠 Sentiment: {post.get('sentiment_score', 0):.2f}" if post.get('sentiment_score') else ''}
                    <br>
                    <a href="{reddit_url}" target="_blank" rel="noopener noreferrer" class="reddit-link negative-reddit-link">
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
                            <strong>u/{safe_html(comment.get('author', 'deleted'))}</strong> • ⬆️ {safe_html(comment.get('score', 0))} upvotes
                        </div>
                        <div class="comment-text">
                            {safe_html(comment.get('body', ''))}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

def display_costs(location: str):
    """Display comprehensive cost estimates."""
    st.markdown('<div class="section-header">💰 Estimated Costs</div>', unsafe_allow_html=True)

    costs = COST_DATA.get(location)
    if costs is None:
        st.info(f'No budget estimate is available for {location} yet.')
        return
    st.caption('Illustrative estimates in USD; actual prices vary by date and travel style.')

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

@st.cache_data(ttl=300)
def load_extraction_summary(demo_mode: bool = False):
    """Load extraction summary for analytics."""
    s3_client = None if demo_mode else get_s3_client()
    if s3_client:
        bucket_name = get_environment_value('S3_BUCKET_NAME')
        if bucket_name:
            try:
                response = s3_client.get_object(Bucket=bucket_name, Key='extraction_summary.json')
                return validate_extraction_summary(json.loads(response['Body'].read().decode('utf-8')))
            except ClientError as error:
                if error.response['Error']['Code'] != 'NoSuchKey':
                    st.warning('Could not load the S3 extraction summary')
            except Exception as error:
                st.warning(f'Could not read the S3 extraction summary: {error}')

    root = get_data_directory() / 'demo' if demo_mode else get_data_directory()
    summary_path = root / 'summaries' / 'extraction_summary.json'
    try:
        if summary_path.is_file():
            with summary_path.open(encoding='utf-8') as f:
                return validate_extraction_summary(json.load(f), include_demo=demo_mode)
    except (OSError, ValueError) as error:
        st.warning(f'Could not read the local extraction summary: {error}')

    return None


def request_reddit_refresh():
    """Widget callback runs before the next script, so mode changes are safe."""
    st.session_state['reddit_data_source'] = 'Extract Fresh Data'
    st.session_state['refresh_reddit'] = True


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
            index=0, key='destination',
        )

        st.subheader("Display Options")
        show_restaurants = st.checkbox("🍽️ Show Restaurants", value=True, key='show_restaurants')
        show_attractions = st.checkbox("🏛️ Show Attractions", value=True, key='show_attractions')
        show_reddit = st.checkbox("📝 Show Reddit Insights", value=True, key='show_reddit')
        show_costs = st.checkbox("💰 Show Cost Estimates", value=True, key='show_costs')

        st.subheader("Data Options")
        demo_mode = st.checkbox('Use demo data', value=False, key='demo_mode')
        show_planner = st.checkbox('Show Trip Planner', value=True, key='show_planner')

        data_source_option = st.radio(
            'Reddit Data Source:', ['Use Stored Data', 'Extract Fresh Data'],
            key='reddit_data_source', disabled=not api_status['reddit'] or demo_mode,
            help='Stored posts work without a Reddit connection. Fresh collection runs only when requested.',
        )
        use_fresh_reddit = data_source_option == 'Extract Fresh Data' and api_status['reddit'] and not demo_mode
        max_posts = st.slider('Max posts to extract:', 10, 50, 20) if use_fresh_reddit else 20
        st.button(
            '🔄 Extract Fresh Reddit Data', key='fetch_reddit', on_click=request_reddit_refresh,
            disabled=not api_status['reddit'] or demo_mode or not (show_reddit or show_planner),
        )
        fresh_request = st.session_state.pop('refresh_reddit', False)

        # API status indicators
        st.subheader("📊 Data Sources")

        # Google Places status
        if api_status['google']:
            st.markdown('<span class="status-indicator status-connected">Google Places configured</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-missing">⚠️ Google Places API Missing</span>', unsafe_allow_html=True)

        # Reddit status
        if api_status['reddit']:
            st.markdown('<span class="status-indicator status-connected">Reddit configured</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-missing">⚠️ Reddit API Missing</span>', unsafe_allow_html=True)

        # AWS status
        if api_status['aws']:
            st.markdown('<span class="status-indicator status-connected">S3 configured</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-missing">📝 Local Storage Only</span>', unsafe_allow_html=True)

        # Location statistics
        st.subheader(f"📈 {selected_location} Data")

        # Load and display summary statistics
        summary = load_extraction_summary(demo_mode)
        if summary and selected_location in summary.get('by_location', {}):
            location_stats = summary['by_location'][selected_location]
            st.metric("Travel Posts", location_stats.get('travel', 0))
            st.metric("Food Posts", location_stats.get('food', 0))
            st.metric("Events Posts", location_stats.get('events', 0))
            st.metric("Total Posts", sum(location_stats.values()))
        else:
            st.metric("Available Data", "No summary")
            if api_status['reddit']:
                st.caption("Extract fresh data to see statistics")

    # Main content area
    if selected_location:
        if demo_mode:
            st.warning('Demo mode: community posts are fictional samples, not real traveler reports.')

        # Load all data with progress indicators
        with st.spinner(f"Loading comprehensive data for {selected_location}..."):

            # Get Google Places data
            places_data = None
            if demo_mode:
                places_data = demo_places(selected_location)
                data_source_places = 'Fictional demo places'
            elif api_status['google'] and (show_restaurants or show_attractions or show_planner):
                places_data = get_google_places_data(selected_location)
                data_source_places = "Google Places"
            else:
                data_source_places = "API Key Required"

            # Get Reddit data
            reddit_posts = []
            data_source_reddit = "No Data"

            if show_reddit or show_planner:
                if use_fresh_reddit:
                    snapshots = st.session_state.setdefault('fresh_reddit_posts', {})
                    snapshot_key = (selected_location, max_posts)
                    if fresh_request:
                        snapshots[snapshot_key] = extract_fresh_reddit_data(selected_location, max_posts)
                    reddit_posts = snapshots.get(snapshot_key, [])
                    data_source_reddit = f"Fresh Reddit snapshot ({len(reddit_posts)} posts)"
                else:
                    reddit_posts = load_stored_reddit_data(selected_location, str(get_data_directory()), demo_mode)
                    data_source_reddit = f"{'Demo' if demo_mode else 'Stored Reddit'} Data ({len(reddit_posts)} posts)"

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
            elif not api_status['google']:
                st.info('Restaurant recommendations are unavailable until Google Places is configured.')
            else:
                st.info('No restaurant recommendations are available for this destination right now.')

        if show_attractions:
            if places_data and places_data['attractions']:
                display_attractions(places_data['attractions'], selected_location, data_source_places)
            elif not api_status['google']:
                st.info('Attraction recommendations are unavailable until Google Places is configured.')
            else:
                st.info('No attraction recommendations are available for this destination right now.')

        if show_costs:
            display_costs(selected_location)

        if show_planner:
            render_trip_planner(
                selected_location, reddit_posts, places_data, demo_mode=demo_mode,
                api_key=get_environment_value('OPENAI_API_KEY'),
                model_name=get_environment_value('OPENAI_MODEL') or 'gpt-4.1-mini',
            )

        if show_reddit:
            if reddit_posts:
                display_reddit_insights(sentiment_analysis, selected_location, data_source_reddit)
                if sentiment_analysis.get('neutral'):
                    with st.expander('Community discussions with neutral or mixed sentiment'):
                        for post in sentiment_analysis['neutral']:
                            st.write(post.get('title', 'Untitled'))
                            st.write(post.get('text') or post.get('summary', ''))
            else:
                st.info('No community posts are available for this destination in the selected data source.')
                if api_status['reddit'] and not demo_mode:
                    st.button('Collect posts for this destination', key='collect_empty_reddit', on_click=request_reddit_refresh)

    # Analytics dashboard
    st.markdown('<h2 style="color: #ffffff; margin-top: 3rem;">📊 Global Community Insights</h2>', unsafe_allow_html=True)

    # Load summary for global stats
    summary = load_extraction_summary(demo_mode)

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
        google_status = "Configured" if api_status['google'] else "Not configured"
        st.info(f"**Restaurants & Attractions**\n{google_status}")
        if api_status['google']:
            st.caption("Showing verified Google Places data")

    with col2:
        reddit_status = 'Demo data' if demo_mode else 'Stored data available without Reddit credentials'
        st.info(f"**Community Insights**\n{reddit_status}")
        if api_status['reddit']:
            st.caption("Fresh collection is available on request")

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
