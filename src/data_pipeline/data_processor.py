"""
Data processing module for cleaning and enriching extracted content.
Handles text cleaning, location extraction, sentiment analysis, and categorization.
"""

import re
import json
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path
from textblob import TextBlob

from ..utils.helpers import PROJECT_ROOT, clean_text, categorize_content, load_config
from ..utils.content import classify_sentiment
from .storage import deduplicate_posts, load_extracted_posts, read_posts

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
    url: str = ''
    author: str = ''
    summary: str = ''
    target_location: str = ''
    detected_locations: List[str] = field(default_factory=list)
    top_comments: List[Dict[str, Any]] = field(default_factory=list)
    num_comments: int = 0
    relevancy_score: float = 0.0


class DataProcessor:
    """Processes and enriches extracted lifestyle content."""

    def __init__(self):
        """Initialize the data processor."""
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
        self.all_known_places.update(load_config()['destinations'])
        self.canonical_places = {place.casefold(): place for place in self.all_known_places}
        self.location_pattern = re.compile(
            r'(?<!\w)(?:' + '|'.join(re.escape(place) for place in sorted(self.all_known_places, key=len, reverse=True)) + r')(?!\w)',
            re.IGNORECASE,
        )

    def extract_locations_advanced(self, text: str) -> List[str]:
        """Match known places at word boundaries, without guessing from capitalization."""
        return sorted({self.canonical_places[match.group().casefold()]
                       for match in self.location_pattern.finditer(text)})

    def is_valid_location(self, location: str) -> bool:
        """Validate if extracted text is likely a real location."""
        return location.casefold() in self.canonical_places

    def analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """Analyze sentiment of text using TextBlob."""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1

            sentiment = classify_sentiment(text, polarity)
            return sentiment['sentiment_score'], sentiment['sentiment_label']

        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            sentiment = classify_sentiment(text)
            return sentiment['sentiment_score'], sentiment['sentiment_label']

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
                enhanced_data = {**post_data, 'cleaned_text': cleaned_text, 'locations': locations, 'category': category}
                quality_score = self.calculate_quality_score(enhanced_data)

                # Parse timestamp
                timestamp_str = post_data.get('timestamp', '')
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                except (AttributeError, ValueError, TypeError):
                    if post_data.get('created_utc') is None:
                        raise ValueError('Post must have a valid timestamp or created_utc')
                    timestamp = datetime.fromtimestamp(post_data['created_utc'], timezone.utc)

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
                    hash=post_data.get('hash', ''),
                    url=post_data.get('url', ''),
                    author=post_data.get('author', ''),
                    summary=post_data.get('summary', ''),
                    target_location=post_data.get('target_location', ''),
                    detected_locations=post_data.get('detected_locations', []),
                    top_comments=post_data.get('top_comments', []),
                    num_comments=post_data.get('num_comments', 0),
                    relevancy_score=post_data.get('relevancy_score', 0.0),
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
            post_dict = {**asdict(post), 'timestamp': post.timestamp.isoformat()}
            posts_data.append(post_dict)

        # Save to file
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
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


def main(argv=None):
    """Process the extractor's per-location files, preserving source evidence."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Optional legacy JSON post file')
    parser.add_argument('--demo', action='store_true', help='Read and write isolated demo data')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    processor = DataProcessor()
    root = PROJECT_ROOT / 'data' / 'demo' if args.demo else PROJECT_ROOT / 'data'
    try:
        all_data = (deduplicate_posts(read_posts(args.input), include_demo=args.demo) if args.input
                    else load_extracted_posts(root / 'by_location', include_demo=args.demo))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not all_data:
        parser.error('No extracted posts found. Run the extractor first, or use --demo for sample data.')
    output_dir = args.output_dir or root / 'processed'
    print(f"Loaded {len(all_data)} raw posts")

    # Process all the data
    processed_posts = processor.process_reddit_posts(all_data)
    print(f"Successfully processed {len(processed_posts)} posts")

    # Filter high-quality posts (lower threshold since we have more data)
    high_quality = processor.filter_high_quality_posts(
        processed_posts, min_quality_score=load_config()['ml']['quality_threshold']
    )
    print(f"Found {len(high_quality)} high-quality posts")

    # Save all processed data
    processor.save_processed_data(processed_posts, output_dir / 'all_processed_posts.json')

    # Save high-quality subset
    processor.save_processed_data(high_quality, output_dir / 'high_quality_posts.json')

    # Create comprehensive analytics
    analytics = processor.create_analytics_summary(processed_posts)
    if not analytics:
        parser.error('No posts passed processing; inspect the validation errors above.')

    print("\n=== Analytics Summary ===")
    print(f"Total processed posts: {analytics['total_posts']}")
    print(f"Average quality score: {analytics['quality_stats']['average_quality']:.1f}")
    print(f"Average word count: {analytics['word_count_stats']['average_words']:.1f}")

    print("\nCategory distribution:")
    for cat, count in analytics['category_distribution'].items():
        print(f"  {cat}: {count} posts")

    print("\nSentiment distribution:")
    for sent, count in analytics['sentiment_distribution'].items():
        print(f"  {sent}: {count} posts")

    print("\nTop 10 locations mentioned:")
    for location, count in list(analytics['top_locations'].items())[:10]:
        print(f"  {location}: {count} times")

    # Save analytics
    import json

    with (output_dir / 'analytics_summary.json').open('w', encoding='utf-8') as f:
        json.dump(analytics, f, indent=2)

    print("\nData processing complete!")
    print("Files created:")
    print(f"  - {output_dir / 'all_processed_posts.json'} ({len(processed_posts)} posts)")
    print(f"  - {output_dir / 'high_quality_posts.json'} ({len(high_quality)} posts)")
    print(f"  - {output_dir / 'analytics_summary.json'}")


if __name__ == "__main__":
    main()
