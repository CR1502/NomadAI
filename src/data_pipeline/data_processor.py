"""
Data processing module for cleaning and enriching extracted content.
Handles text cleaning, location extraction, sentiment analysis, and categorization.
"""

import re
import json
import pandas as pd
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from textblob import TextBlob

from ..utils.helpers import clean_text, categorize_content, extract_location_mentions, validate_data_quality

logger = logging.getLogger(__name__)


@dataclass
class ProcessedPost:
    """Data class for processed post information."""
    id: str
    title: str
    text: str
    cleaned_text: str
    subreddit: str
    category: str
    locations: List[str]
    sentiment_score: float
    sentiment_label: str
    score: int
    timestamp: datetime
    word_count: int
    quality_score: float
    source: str
    hash: str


class DataProcessor:
    """Processes and enriches extracted lifestyle content."""

    def __init__(self):
        """Initialize the data processor."""
        self.setup_nltk()

        # Enhanced location patterns for better extraction
        self.location_patterns = [
            # City, State format (US)
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2})\b',
            # City, Country format
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            # "in Location" format
            r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,\s*[A-Z]{2})?)\b',
            # "at Location" format
            r'\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            # "visiting Location" format
            r'\bvisiting\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            # "traveled to Location" format
            r'\btraveled?\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            # Single capitalized words that might be places
            r'\b([A-Z][a-z]{3,}(?:\s+[A-Z][a-z]+)*)\b',
        ]

        # Known places to help validate extractions
        self.known_places = {
            'countries': ['Japan', 'Germany', 'France', 'Italy', 'Spain', 'UK', 'USA', 'Canada', 'Australia', 'Norway',
                          'Sweden', 'Denmark', 'Netherlands', 'Belgium', 'Switzerland', 'Austria', 'Portugal', 'Greece',
                          'Turkey', 'Thailand', 'Vietnam', 'Cambodia', 'India', 'China', 'South Korea', 'Brazil',
                          'Argentina', 'Mexico', 'Chile', 'Peru', 'Egypt', 'Morocco', 'South Africa', 'Kenya',
                          'Tanzania', 'Namibia'],
            'us_states': ['California', 'New York', 'Texas', 'Florida', 'Illinois', 'Pennsylvania', 'Ohio', 'Georgia',
                          'North Carolina', 'Michigan', 'New Jersey', 'Virginia', 'Washington', 'Arizona',
                          'Massachusetts', 'Tennessee', 'Indiana', 'Missouri', 'Maryland', 'Wisconsin', 'Colorado',
                          'Minnesota', 'South Carolina', 'Alabama', 'Louisiana', 'Kentucky', 'Oregon', 'Oklahoma',
                          'Connecticut', 'Utah', 'Iowa', 'Nevada', 'Arkansas', 'Mississippi', 'Kansas', 'New Mexico',
                          'Nebraska', 'West Virginia', 'Idaho', 'Hawaii', 'New Hampshire', 'Maine', 'Montana',
                          'Rhode Island', 'Delaware', 'South Dakota', 'North Dakota', 'Alaska', 'Vermont', 'Wyoming'],
            'major_cities': ['Tokyo', 'New York', 'London', 'Paris', 'Los Angeles', 'Chicago', 'Berlin', 'Madrid',
                             'Rome', 'Amsterdam', 'Barcelona', 'Vienna', 'Prague', 'Budapest', 'Warsaw', 'Stockholm',
                             'Oslo', 'Copenhagen', 'Helsinki', 'Zurich', 'Geneva', 'Munich', 'Frankfurt', 'Hamburg',
                             'Milan', 'Venice', 'Florence', 'Naples', 'Athens', 'Istanbul', 'Dubai', 'Singapore',
                             'Hong Kong', 'Seoul', 'Osaka', 'Kyoto', 'Sydney', 'Melbourne', 'Toronto', 'Vancouver',
                             'Montreal', 'Mexico City', 'Buenos Aires', 'Rio de Janeiro', 'São Paulo', 'Lima', 'Bogotá',
                             'Santiago', 'Cairo', 'Marrakech', 'Cape Town', 'Nairobi', 'Mumbai', 'Delhi', 'Bangkok',
                             'Ho Chi Minh City', 'Hanoi', 'Jakarta', 'Manila', 'Kuala Lumpur']
        }

        # Flatten all known places for quick lookup
        self.all_known_places = set()
        for place_list in self.known_places.values():
            self.all_known_places.update(place_list)

    def setup_nltk(self):
        """Download required NLTK data."""
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
        except Exception as e:
            logger.warning(f"Could not download NLTK data: {e}")

    def extract_locations_advanced(self, text: str) -> List[str]:
        """Extract location mentions using advanced patterns."""
        locations = set()

        # Apply each pattern
        for pattern in self.location_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    # For patterns with groups like "City, State"
                    location = f"{match.group(1)}, {match.group(2)}".strip()
                else:
                    # For simple patterns
                    location = match.group(1).strip()

                # Clean the location
                location = location.title()

                # Validate location
                if self.is_valid_location(location):
                    locations.add(location)

        # Also check for known places directly
        words = text.split()
        for i, word in enumerate(words):
            # Check single words
            clean_word = re.sub(r'[^\w]', '', word).title()
            if clean_word in self.all_known_places:
                locations.add(clean_word)

            # Check two-word combinations
            if i < len(words) - 1:
                two_word = f"{clean_word} {re.sub(r'[^\w]', '', words[i + 1]).title()}"
                if two_word in self.all_known_places:
                    locations.add(two_word)

        return list(locations)

    def is_valid_location(self, location: str) -> bool:
        """Validate if extracted text is likely a real location."""
        # Skip very short or long strings
        if len(location) < 3 or len(location) > 50:
            return False

        # Skip common non-location words
        common_words = {'The', 'And', 'For', 'You', 'Are', 'But', 'Not', 'Can', 'All', 'Get', 'New', 'Now', 'Old',
                        'See', 'Him', 'Two', 'How', 'Its', 'Our', 'Out', 'Day', 'Had', 'Her', 'His', 'She', 'Use',
                        'Man', 'Way', 'Who', 'Boy', 'Did', 'May', 'Say', 'She', 'Too', 'Any', 'Old', 'Try'}
        if location in common_words:
            return False

        # Check if it's a known place
        if location in self.all_known_places:
            return True

        # Check if it has location-like characteristics
        # Contains common location suffixes
        location_suffixes = ['City', 'Town', 'Beach', 'Island', 'Mountain', 'Valley', 'River', 'Lake', 'Park', 'Square',
                             'Street', 'Avenue', 'Road']
        if any(suffix in location for suffix in location_suffixes):
            return True

        # If it's properly capitalized and not obviously wrong
        if location[0].isupper() and len(location) >= 4:
            return True

        return False

    def analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """Analyze sentiment of text using TextBlob."""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1

            # Convert to label
            if polarity > 0.1:
                label = 'positive'
            elif polarity < -0.1:
                label = 'negative'
            else:
                label = 'neutral'

            return polarity, label

        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return 0.0, 'neutral'

    def calculate_quality_score(self, post_data: Dict[str, Any]) -> float:
        """Calculate a quality score for the post."""
        score = 0.0

        # Text length score (0-30 points)
        text_length = len(post_data.get('cleaned_text', ''))
        if 50 <= text_length <= 500:
            score += 30
        elif 30 <= text_length < 50 or 500 < text_length <= 1000:
            score += 20
        elif 20 <= text_length < 30 or 1000 < text_length <= 2000:
            score += 10

        # Reddit score (0-25 points)
        reddit_score = post_data.get('score', 0)
        if reddit_score >= 100:
            score += 25
        elif reddit_score >= 50:
            score += 20
        elif reddit_score >= 20:
            score += 15
        elif reddit_score >= 5:
            score += 10

        # Comment engagement (0-15 points)
        num_comments = post_data.get('num_comments', 0)
        if num_comments >= 50:
            score += 15
        elif num_comments >= 20:
            score += 12
        elif num_comments >= 10:
            score += 8
        elif num_comments >= 5:
            score += 5

        # Location mentions (0-15 points)
        locations = post_data.get('locations', [])
        score += min(len(locations) * 5, 15)

        # Category relevance (0-15 points)
        category = post_data.get('category', 'general')
        if category in ['travel', 'food', 'events']:
            score += 15
        elif category == 'general':
            score += 5

        return min(score, 100.0)  # Cap at 100

    def process_reddit_posts(self, posts_data: List[Dict[str, Any]]) -> List[ProcessedPost]:
        """Process a list of Reddit posts."""
        processed_posts = []

        for i, post_data in enumerate(posts_data):
            try:
                # Clean and enhance text
                cleaned_text = clean_text(post_data.get('text', ''))
                if len(cleaned_text) < 10:  # Skip very short posts
                    continue

                # Extract locations from title and text combined
                full_text = f"{post_data.get('title', '')} {cleaned_text}"
                locations = self.extract_locations_advanced(full_text)

                # Analyze sentiment
                sentiment_score, sentiment_label = self.analyze_sentiment(cleaned_text)

                # Recategorize with cleaned text
                category = categorize_content(cleaned_text, post_data.get('title', ''))

                # Calculate quality score
                enhanced_data = {**post_data, 'cleaned_text': cleaned_text, 'locations': locations}
                quality_score = self.calculate_quality_score(enhanced_data)

                # Parse timestamp
                timestamp_str = post_data.get('timestamp', datetime.now().isoformat())
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except:
                    timestamp = datetime.now()

                # Create processed post
                processed_post = ProcessedPost(
                    id=post_data.get('id', f'post_{i}'),
                    title=post_data.get('title', ''),
                    text=post_data.get('text', ''),
                    cleaned_text=cleaned_text,
                    subreddit=post_data.get('subreddit', ''),
                    category=category,
                    locations=locations,
                    sentiment_score=sentiment_score,
                    sentiment_label=sentiment_label,
                    score=post_data.get('score', 0),
                    timestamp=timestamp,
                    word_count=len(cleaned_text.split()),
                    quality_score=quality_score,
                    source=post_data.get('source', 'reddit'),
                    hash=post_data.get('hash', '')
                )

                processed_posts.append(processed_post)

                # Progress indicator
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(posts_data)} posts")

            except Exception as e:
                logger.error(f"Error processing post {post_data.get('id', 'unknown')}: {e}")
                continue

        logger.info(f"Processed {len(processed_posts)} posts successfully")
        return processed_posts

    def filter_high_quality_posts(
            self,
            posts: List[ProcessedPost],
            min_quality_score: float = 40.0
    ) -> List[ProcessedPost]:
        """Filter posts by quality score."""
        high_quality = [post for post in posts if post.quality_score >= min_quality_score]
        logger.info(f"Filtered to {len(high_quality)} high-quality posts (min score: {min_quality_score})")
        return high_quality

    def save_processed_data(self, posts: List[ProcessedPost], filename: str):
        """Save processed posts to JSON file."""
        posts_data = []

        for post in posts:
            post_dict = {
                'id': post.id,
                'title': post.title,
                'text': post.text,
                'cleaned_text': post.cleaned_text,
                'subreddit': post.subreddit,
                'category': post.category,
                'locations': post.locations,
                'sentiment_score': post.sentiment_score,
                'sentiment_label': post.sentiment_label,
                'score': post.score,
                'timestamp': post.timestamp.isoformat(),
                'word_count': post.word_count,
                'quality_score': post.quality_score,
                'source': post.source,
                'hash': post.hash
            }
            posts_data.append(post_dict)

        # Save to file
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(posts_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(posts_data)} processed posts to {filename}")

    def create_analytics_summary(self, posts: List[ProcessedPost]) -> Dict[str, Any]:
        """Create analytics summary of processed posts."""
        if not posts:
            return {}

        # Category distribution
        categories = [post.category for post in posts]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}

        # Sentiment distribution
        sentiments = [post.sentiment_label for post in posts]
        sentiment_counts = {sent: sentiments.count(sent) for sent in set(sentiments)}

        # Location frequency
        all_locations = []
        for post in posts:
            all_locations.extend(post.locations)
        location_counts = {loc: all_locations.count(loc) for loc in set(all_locations)}
        location_counts = dict(sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:20])

        # Quality statistics
        quality_scores = [post.quality_score for post in posts]

        # Subreddit distribution
        subreddits = [post.subreddit for post in posts]
        subreddit_counts = {sub: subreddits.count(sub) for sub in set(subreddits)}

        summary = {
            'total_posts': len(posts),
            'category_distribution': category_counts,
            'sentiment_distribution': sentiment_counts,
            'subreddit_distribution': subreddit_counts,
            'top_locations': location_counts,
            'quality_stats': {
                'average_quality': sum(quality_scores) / len(quality_scores),
                'min_quality': min(quality_scores),
                'max_quality': max(quality_scores)
            },
            'word_count_stats': {
                'average_words': sum(post.word_count for post in posts) / len(posts),
                'total_words': sum(post.word_count for post in posts)
            }
        }

        return summary


if __name__ == "__main__":
    # Test the data processor with comprehensive data
    import logging

    logging.basicConfig(level=logging.INFO)

    processor = DataProcessor()

    # Load all extracted Reddit data
    with open('data/raw/all_posts.json', 'r') as f:
        all_data = json.load(f)

    print(f"Loaded {len(all_data)} raw posts")

    # Process all the data
    processed_posts = processor.process_reddit_posts(all_data)
    print(f"Successfully processed {len(processed_posts)} posts")

    # Filter high-quality posts (lower threshold since we have more data)
    high_quality = processor.filter_high_quality_posts(processed_posts, min_quality_score=25.0)
    print(f"Found {len(high_quality)} high-quality posts")

    # Save all processed data
    processor.save_processed_data(processed_posts, 'data/processed/all_processed_posts.json')

    # Save high-quality subset
    processor.save_processed_data(high_quality, 'data/processed/high_quality_posts.json')

    # Create comprehensive analytics
    analytics = processor.create_analytics_summary(processed_posts)

    print(f"\n=== Analytics Summary ===")
    print(f"Total processed posts: {analytics['total_posts']}")
    print(f"Average quality score: {analytics['quality_stats']['average_quality']:.1f}")
    print(f"Average word count: {analytics['word_count_stats']['average_words']:.1f}")

    print(f"\nCategory distribution:")
    for cat, count in analytics['category_distribution'].items():
        print(f"  {cat}: {count} posts")

    print(f"\nSentiment distribution:")
    for sent, count in analytics['sentiment_distribution'].items():
        print(f"  {sent}: {count} posts")

    print(f"\nTop 10 locations mentioned:")
    for location, count in list(analytics['top_locations'].items())[:10]:
        print(f"  {location}: {count} times")

    # Save analytics
    import json

    with open('data/processed/analytics_summary.json', 'w') as f:
        json.dump(analytics, f, indent=2)

    print(f"\nData processing complete!")
    print(f"Files created:")
    print(f"  - data/processed/all_processed_posts.json ({len(processed_posts)} posts)")
    print(f"  - data/processed/high_quality_posts.json ({len(high_quality)} posts)")
    print(f"  - data/processed/analytics_summary.json")