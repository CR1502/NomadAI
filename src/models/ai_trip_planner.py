"""
AI-powered trip planner using OpenAI API with Reddit data as RAG context.
"""

import openai
import json
import logging
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import random

load_dotenv('docker/.env')
logger = logging.getLogger(__name__)


class AITripPlanner:
    """AI-powered trip planner using Reddit community insights."""

    def __init__(self):
        """Initialize the AI trip planner."""
        self.setup_openai()

    def setup_openai(self):
        """Setup OpenAI client."""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OpenAI API key not found. AI features will be limited.")
                self.client = None
                return

            openai.api_key = api_key
            self.client = openai
            logger.info("✅ OpenAI client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            self.client = None

    def create_reddit_context(self, posts: List[Dict[str, Any]], location: str) -> str:
        """Create context from Reddit posts for AI prompting."""

        # Select high-quality posts
        quality_posts = [p for p in posts if p.get('enhanced_quality_score', 0) > 60]
        quality_posts = quality_posts[:15]  # Top 15 posts

        context_parts = [
            f"Community insights about {location} from Reddit:",
            ""
        ]

        for i, post in enumerate(quality_posts, 1):
            sentiment = post.get('enhanced_sentiment', {})
            entities = post.get('entities', {})

            context_part = f"""
{i}. {post['title']} (Score: {post['score']}, Sentiment: {sentiment.get('sentiment_label', 'neutral')})
{post.get('summary', post.get('text', '')[:200])}...
Locations mentioned: {', '.join(entities.get('locations', [])[:3])}
Organizations mentioned: {', '.join(entities.get('organizations', [])[:2])}
"""
            context_parts.append(context_part)

        return '\n'.join(context_parts)

    def generate_personalized_itinerary(
            self,
            location: str,
            reddit_posts: List[Dict[str, Any]],
            user_preferences: Dict[str, Any],
            restaurants: List[Dict] = None,
            attractions: List[Dict] = None
    ) -> Dict[str, Any]:
        """Generate AI-powered personalized itinerary."""

        if not self.client:
            return self._generate_fallback_itinerary(location, user_preferences, restaurants, attractions)

        # Create Reddit context
        reddit_context = self.create_reddit_context(reddit_posts, location)

        # Build the prompt
        prompt = self._build_itinerary_prompt(
            location, user_preferences, reddit_context, restaurants, attractions
        )

        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are an expert travel planner who specializes in creating personalized itineraries based on authentic community experiences from Reddit and verified business data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )

            ai_response = response.choices[0].message.content

            # Parse the AI response into structured format
            parsed_itinerary = self._parse_ai_response(ai_response, location, user_preferences)

            return parsed_itinerary

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return self._generate_fallback_itinerary(location, user_preferences, restaurants, attractions)

    def _build_itinerary_prompt(
            self,
            location: str,
            preferences: Dict[str, Any],
            reddit_context: str,
            restaurants: List[Dict] = None,
            attractions: List[Dict] = None
    ) -> str:
        """Build the prompt for AI itinerary generation."""

        # Extract preferences
        duration = preferences.get('duration', 3)
        budget = preferences.get('budget', 'medium')
        interests = preferences.get('interests', [])
        travel_style = preferences.get('travel_style', 'balanced')

        # Build restaurant and attraction context
        places_context = ""
        if restaurants:
            restaurant_names = [r['name'] for r in restaurants[:8]]
            places_context += f"Verified restaurants: {', '.join(restaurant_names)}\n"

        if attractions:
            attraction_names = [a['name'] for a in attractions[:8]]
            places_context += f"Verified attractions: {', '.join(attraction_names)}\n"

        prompt = f"""
Create a personalized {duration}-day itinerary for {location} based on the following:

USER PREFERENCES:
- Budget: {budget}
- Interests: {', '.join(interests) if interests else 'General sightseeing'}
- Travel style: {travel_style}
- Duration: {duration} days

VERIFIED PLACES DATA:
{places_context}

REDDIT COMMUNITY INSIGHTS:
{reddit_context}

Please create a detailed itinerary that:
1. Incorporates specific restaurants and attractions from the verified data
2. Uses insights from Reddit community experiences
3. Matches the user's budget and interests
4. Includes practical tips from the Reddit community
5. Suggests specific times and logistics

Format the response as:
DAY 1: [Title]
- 9:00 AM: [Activity] - [Description with reasoning]
- 12:00 PM: [Restaurant] - [Why this choice based on data]
- 2:00 PM: [Attraction] - [Community insights]
...

REDDIT TIPS:
- [Specific tip from community]
- [Another community insight]

BUDGET NOTES:
- [Cost considerations]
- [Money-saving tips from Reddit]
"""

        return prompt

    def _parse_ai_response(self, ai_response: str, location: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Parse AI response into structured itinerary format."""

        # Extract days using regex
        day_pattern = r'DAY (\d+):(.*?)(?=DAY \d+:|REDDIT TIPS:|BUDGET NOTES:|$)'
        day_matches = re.findall(day_pattern, ai_response, re.DOTALL | re.IGNORECASE)

        days = []
        for day_num, day_content in day_matches:
            # Extract activities from day content
            activity_pattern = r'- (\d{1,2}:\d{2} [AP]M): (.*?)(?=- \d{1,2}:\d{2} [AP]M|$)'
            activities = re.findall(activity_pattern, day_content, re.DOTALL)

            day_activities = []
            for time, activity_desc in activities:
                # Split activity description
                parts = activity_desc.split(' - ', 1)
                activity_name = parts[0].strip()
                activity_detail = parts[1].strip() if len(parts) > 1 else ""

                day_activities.append({
                    'time': time,
                    'activity': activity_name,
                    'description': activity_detail,
                    'type': self._classify_activity_type(activity_name)
                })

            days.append({
                'day': int(day_num),
                'title': day_content.split('\n')[0].strip(),
                'activities': day_activities
            })

        # Extract tips
        tips_match = re.search(r'REDDIT TIPS:(.*?)(?=BUDGET NOTES:|$)', ai_response, re.DOTALL | re.IGNORECASE)
        reddit_tips = []
        if tips_match:
            tips_text = tips_match.group(1)
            tips = re.findall(r'- (.*)', tips_text)
            reddit_tips = [tip.strip() for tip in tips if tip.strip()]

        # Extract budget notes
        budget_match = re.search(r'BUDGET NOTES:(.*?)$', ai_response, re.DOTALL | re.IGNORECASE)
        budget_notes = []
        if budget_match:
            budget_text = budget_match.group(1)
            notes = re.findall(r'- (.*)', budget_text)
            budget_notes = [note.strip() for note in notes if note.strip()]

        return {
            'location': location,
            'user_preferences': preferences,
            'days': days,
            'reddit_tips': reddit_tips,
            'budget_notes': budget_notes,
            'generated_at': datetime.now().isoformat(),
            'ai_generated': True
        }

    def _classify_activity_type(self, activity_name: str) -> str:
        """Classify activity type for icons."""
        activity_lower = activity_name.lower()

        if any(word in activity_lower for word in ['restaurant', 'eat', 'lunch', 'dinner', 'cafe', 'food']):
            return 'restaurant'
        elif any(word in activity_lower for word in ['museum', 'gallery', 'monument', 'temple', 'church', 'palace']):
            return 'attraction'
        elif any(word in activity_lower for word in ['walk', 'stroll', 'explore', 'wander']):
            return 'walking'
        elif any(word in activity_lower for word in ['market', 'shopping', 'shop']):
            return 'shopping'
        else:
            return 'activity'

    def _generate_fallback_itinerary(
            self,
            location: str,
            preferences: Dict[str, Any],
            restaurants: List[Dict] = None,
            attractions: List[Dict] = None
    ) -> Dict[str, Any]:
        """Generate fallback itinerary when AI is not available."""

        duration = preferences.get('duration', 3)

        # Create basic structure
        days = []
        for day_num in range(1, duration + 1):
            activities = []

            # Morning attraction
            if attractions and len(attractions) >= day_num:
                activities.append({
                    'time': '9:30 AM',
                    'activity': attractions[day_num - 1]['name'],
                    'description': f"Visit this highly-rated attraction. Community recommended.",
                    'type': 'attraction'
                })

            # Lunch
            if restaurants and len(restaurants) >= day_num:
                activities.append({
                    'time': '12:30 PM',
                    'activity': restaurants[day_num - 1]['name'],
                    'description': f"Lunch at this popular restaurant. Rating: {restaurants[day_num - 1].get('rating', 'N/A')}",
                    'type': 'restaurant'
                })

            days.append({
                'day': day_num,
                'title': f'Day {day_num} in {location}',
                'activities': activities
            })

        return {
            'location': location,
            'user_preferences': preferences,
            'days': days,
            'reddit_tips': ['Plan ahead for popular attractions', 'Try local food markets'],
            'budget_notes': ['Book accommodations early for better rates'],
            'generated_at': datetime.now().isoformat(),
            'ai_generated': False
        }


if __name__ == "__main__":
    # Test the AI trip planner
    planner = AITripPlanner()

    # Sample preferences
    preferences = {
        'duration': 3,
        'budget': 'medium',
        'interests': ['food', 'culture', 'history'],
        'travel_style': 'explorer'
    }

    # Load sample Reddit data
    with open('data/processed/all_processed_posts.json', 'r') as f:
        sample_posts = json.load(f)[:20]

    # Generate itinerary
    itinerary = planner.generate_personalized_itinerary(
        location='Tokyo',
        reddit_posts=sample_posts,
        user_preferences=preferences
    )

    print("Generated itinerary:")
    print(json.dumps(itinerary, indent=2))