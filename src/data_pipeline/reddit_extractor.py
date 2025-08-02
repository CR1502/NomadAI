"""
Modular Reddit data extraction for lifestyle discovery.
Extracts and stores travel, food, and events data separately by location with comments.
"""

import praw
import pandas as pd
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

from ..utils.helpers import clean_text, categorize_content, generate_hash, validate_data_quality

# Load environment variables
load_dotenv('docker/.env')

logger = logging.getLogger(__name__)


@dataclass
class RedditComment:
    """Data class for Reddit comment information."""
    id: str
    author: str
    body: str
    score: int
    created_utc: float
    is_top_comment: bool


@dataclass
class LocationPost:
    """Data class for location-specific Reddit post information with comments."""
    id: str
    title: str
    text: str
    summary: str
    subreddit: str
    author: str
    score: int
    num_comments: int
    created_utc: float
    url: str
    category: str
    detected_locations: List[str]
    target_location: str
    hash: str
    relevancy_score: float
    top_comments: List[RedditComment]


class ModularRedditExtractor:
    """Extracts lifestyle content by category and location with comments."""

    def __init__(self):
        """Initialize Reddit API client and AWS S3."""
        self.reddit = None
        self.s3_client = None

        # S3 bucket name - define this first
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'lifestyle-discovery-data')

        # Top 25 most visited destinations globally
        self.top_destinations = [
            "Paris", "London", "New York", "Tokyo", "Rome", "Barcelona",
            "Amsterdam", "Prague", "Vienna", "Berlin", "Istanbul", "Dubai",
            "Bangkok", "Singapore", "Hong Kong", "Sydney", "Los Angeles",
            "Chicago", "Las Vegas", "Miami", "San Francisco", "Venice",
            "Florence", "Athens", "Lisbon"
        ]

        # Category-specific subreddits
        self.category_subreddits = {
            'travel': [
                'travel', 'solotravel', 'backpacking', 'digitalnomad', 'roadtrip',
                'travel_tips', 'budget_travel', 'europe', 'asia', 'northamerica',
                'southamerica', 'africa', 'oceania', 'worldnomads'
            ],
            'food': [
                'food', 'FoodPorn', 'recipes', 'Cooking', 'AskCulinary',
                'streetfood', 'finedining', 'FoodNYC', 'FoodLA', 'FoodLondon',
                'FoodTokyo', 'DessertPorn', 'restaurants'
            ],
            'events': [
                'concerts', 'festivals', 'Music', 'EDM', 'Jazz', 'Rock',
                'WeAreTheMusicMakers', 'eventplanning', 'aves', 'ElectronicMusic',
                'musicfestivals', 'livemusic'
            ]
        }

        # Initialize clients after setting up attributes
        self.setup_reddit_client()
        self.setup_s3_client()

    def setup_reddit_client(self):
        """Setup Reddit API client using credentials."""
        try:
            client_id = os.getenv('REDDIT_CLIENT_ID')
            client_secret = os.getenv('REDDIT_CLIENT_SECRET')
            user_agent = os.getenv('REDDIT_USER_AGENT', 'lifestyle_discovery_bot_v1.0')

            if not client_id or not client_secret:
                logger.warning("Reddit API credentials not found.")
                self.reddit = None
                return

            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )

            logger.info("Reddit API client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            self.reddit = None

    def setup_s3_client(self):
        """Setup AWS S3 client for data storage."""
        try:
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

            if not aws_access_key or not aws_secret_key:
                logger.warning("AWS credentials not found. Using local storage.")
                self.s3_client = None
                return

            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )

            # Create bucket if it doesn't exist
            self._create_bucket_if_not_exists()
            logger.info("S3 client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.s3_client = None

    def _create_bucket_if_not_exists(self):
        """Create S3 bucket if it doesn't exist."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                try:
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                    logger.info(f"Created S3 bucket: {self.bucket_name}")
                except Exception as create_error:
                    logger.error(f"Failed to create bucket: {create_error}")

    def _summarize_post(self, title: str, text: str) -> str:
        """Create a concise summary of the Reddit post."""
        if not text or len(text.strip()) < 50:
            return title

        # Simple extractive summarization - get first 2 sentences or first 200 chars
        sentences = text.split('. ')

        if len(sentences) >= 2:
            summary = '. '.join(sentences[:2]) + '.'
        else:
            summary = text[:200] + '...' if len(text) > 200 else text

        # If summary is too short, use title + beginning of text
        if len(summary) < 100:
            summary = f"{title}. {text[:150]}..." if len(text) > 150 else f"{title}. {text}"

        return summary.strip()

    def _extract_comments(self, submission) -> List[RedditComment]:
        """Extract top comments from a Reddit submission."""
        comments = []

        try:
            # Get all comments and sort by score
            submission.comments.replace_more(limit=0)  # Don't expand "more comments"
            all_comments = submission.comments.list()

            # Filter and sort comments by score
            valid_comments = [
                comment for comment in all_comments
                if hasattr(comment, 'body') and hasattr(comment, 'score')
                and len(comment.body.strip()) > 20  # Minimum comment length
                and comment.score > 1  # Minimum score
            ]

            # Sort by score and take top 3
            valid_comments.sort(key=lambda x: x.score, reverse=True)
            top_comments = valid_comments[:3]

            for i, comment in enumerate(top_comments):
                reddit_comment = RedditComment(
                    id=comment.id,
                    author=str(comment.author) if comment.author else 'deleted',
                    body=clean_text(comment.body),
                    score=comment.score,
                    created_utc=comment.created_utc,
                    is_top_comment=(i == 0)
                )
                comments.append(reddit_comment)

        except Exception as e:
            logger.warning(f"Error extracting comments: {e}")

        return comments

    def extract_location_specific_posts(
        self,
        location: str,
        category: str,
        posts_per_subreddit: int = 50
    ) -> List[LocationPost]:
        """Extract posts relevant to a specific location and category with comments."""

        location_posts = []
        subreddits = self.category_subreddits.get(category, [])

        for subreddit_name in subreddits:
            logger.info(f"Extracting {category} posts about {location} from r/{subreddit_name}")

            try:
                posts = self._search_subreddit_for_location(
                    subreddit_name, location, posts_per_subreddit
                )
                location_posts.extend(posts)

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error extracting from r/{subreddit_name}: {e}")
                continue

        # Sort by relevancy score
        location_posts.sort(key=lambda x: x.relevancy_score, reverse=True)

        logger.info(f"Found {len(location_posts)} {category} posts for {location}")
        return location_posts

    def _search_subreddit_for_location(
        self,
        subreddit_name: str,
        location: str,
        limit: int
    ) -> List[LocationPost]:
        """Search a subreddit for posts mentioning a specific location with comments."""

        if not self.reddit:
            return self._generate_mock_location_data(subreddit_name, location, limit)

        posts = []

        try:
            subreddit = self.reddit.subreddit(subreddit_name)

            # Search for location mentions
            search_terms = [location, location.lower()]

            for search_term in search_terms:
                try:
                    for submission in subreddit.search(search_term, limit=limit//2):

                        # Check if post actually mentions the location
                        full_text = f"{submission.title} {submission.selftext}".lower()
                        if location.lower() not in full_text:
                            continue

                        # Calculate relevancy score
                        relevancy = self._calculate_location_relevancy(full_text, location)
                        if relevancy < 0.3:  # Skip low relevancy posts
                            continue

                        # Clean text
                        cleaned_text = clean_text(submission.selftext or submission.title)
                        if len(cleaned_text) < 20:
                            continue

                        # Create summary
                        summary = self._summarize_post(submission.title, cleaned_text)

                        # Extract comments
                        logger.info(f"Extracting comments for post: {submission.title[:50]}...")
                        top_comments = self._extract_comments(submission)

                        post = LocationPost(
                            id=submission.id,
                            title=clean_text(submission.title),
                            text=cleaned_text,
                            summary=summary,
                            subreddit=subreddit_name,
                            author=str(submission.author) if submission.author else 'deleted',
                            score=submission.score,
                            num_comments=submission.num_comments,
                            created_utc=submission.created_utc,
                            url=submission.url,
                            category=self._determine_category(subreddit_name),
                            detected_locations=self._extract_locations(full_text),
                            target_location=location,
                            hash=generate_hash(cleaned_text),
                            relevancy_score=relevancy,
                            top_comments=top_comments
                        )

                        posts.append(post)

                        # Extra rate limiting when fetching comments
                        time.sleep(1)

                except Exception as search_error:
                    logger.warning(f"Search error in r/{subreddit_name}: {search_error}")
                    continue

                # Rate limiting between searches
                time.sleep(0.5)

        except Exception as e:
            logger.error(f"Error accessing r/{subreddit_name}: {e}")

        return posts

    def _calculate_location_relevancy(self, text: str, location: str) -> float:
        """Calculate how relevant a post is to the target location."""
        score = 0.0
        location_lower = location.lower()

        # Count mentions
        mention_count = text.count(location_lower)
        score += min(mention_count * 0.2, 0.6)

        # Check for travel-related context around location mentions
        travel_keywords = ['visit', 'trip', 'travel', 'went', 'going', 'been to', 'staying in']
        for keyword in travel_keywords:
            if keyword in text and location_lower in text:
                score += 0.2
                break

        # Check for specific experiences
        experience_keywords = ['restaurant', 'hotel', 'museum', 'food', 'concert', 'festival']
        for keyword in experience_keywords:
            if keyword in text:
                score += 0.1

        return min(score, 1.0)

    def _determine_category(self, subreddit_name: str) -> str:
        """Determine category based on subreddit name."""
        for category, subreddits in self.category_subreddits.items():
            if subreddit_name in subreddits:
                return category
        return 'general'

    def _extract_locations(self, text: str) -> List[str]:
        """Extract location mentions from text."""
        locations = []
        for destination in self.top_destinations:
            if destination.lower() in text.lower():
                locations.append(destination)
        return list(set(locations))

    def save_to_s3(self, posts: List[LocationPost], location: str, category: str):
        """Save posts with comments to S3 bucket organized by location and category."""

        # Convert posts to dict format
        posts_data = []
        for post in posts:
            # Convert comments to dict
            comments_data = []
            for comment in post.top_comments:
                comment_dict = {
                    'id': comment.id,
                    'author': comment.author,
                    'body': comment.body,
                    'score': comment.score,
                    'created_utc': comment.created_utc,
                    'timestamp': datetime.fromtimestamp(comment.created_utc).isoformat(),
                    'is_top_comment': comment.is_top_comment
                }
                comments_data.append(comment_dict)

            post_dict = {
                'id': post.id,
                'title': post.title,
                'text': post.text,
                'summary': post.summary,
                'subreddit': post.subreddit,
                'author': post.author,
                'score': post.score,
                'num_comments': post.num_comments,
                'created_utc': post.created_utc,
                'timestamp': datetime.fromtimestamp(post.created_utc).isoformat(),
                'url': post.url,
                'category': post.category,
                'detected_locations': post.detected_locations,
                'target_location': post.target_location,
                'hash': post.hash,
                'relevancy_score': post.relevancy_score,
                'top_comments': comments_data,
                'source': 'reddit'
            }
            posts_data.append(post_dict)

        # Always save locally first
        self._save_locally(posts_data, location, category)

        # S3 key structure: location/category/data.json
        s3_key = f"{location.lower().replace(' ', '_')}/{category}/reddit_posts.json"

        if self.s3_client:
            try:
                # Upload to S3
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=json.dumps(posts_data, indent=2),
                    ContentType='application/json'
                )
                logger.info(f"Uploaded {len(posts_data)} posts to S3: s3://{self.bucket_name}/{s3_key}")

            except Exception as e:
                logger.error(f"Failed to upload to S3: {e}")

    def _save_locally(self, posts_data: List[Dict], location: str, category: str):
        """Save posts locally as fallback."""
        local_dir = f"data/by_location/{location.lower().replace(' ', '_')}/{category}"
        os.makedirs(local_dir, exist_ok=True)

        filename = f"{local_dir}/reddit_posts.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(posts_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(posts_data)} posts locally: {filename}")

    def _save_extraction_summary(self, summary: Dict[str, Any]):
        """Save extraction summary to S3 and locally."""
        summary_data = {
            **summary,
            'extraction_date': datetime.now().isoformat(),
            'destinations_processed': len(self.top_destinations)
        }

        if self.s3_client:
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key='extraction_summary.json',
                    Body=json.dumps(summary_data, indent=2),
                    ContentType='application/json'
                )
                logger.info("Saved extraction summary to S3")
            except Exception as e:
                logger.error(f"Failed to save summary to S3: {e}")

        # Always save locally
        os.makedirs('data/summaries', exist_ok=True)
        with open('data/summaries/extraction_summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2)
        logger.info("Saved extraction summary locally")

    def _generate_mock_location_data(self, subreddit_name: str, location: str, limit: int) -> List[LocationPost]:
        """Generate mock data for testing when Reddit API isn't available."""
        logger.info(f"Generating mock data for {location} in r/{subreddit_name}")

        mock_posts = []
        category = self._determine_category(subreddit_name)

        # Sample templates by category
        templates = {
            'travel': [
                {
                    'title': f"Just got back from an amazing trip to {location}!",
                    'text': f"Just got back from an amazing trip to {location}! The architecture was breathtaking. Spent two weeks exploring the old town, visiting museums, and trying local cuisine. The highlights were definitely the historic district and the sunset views from the main square. Would highly recommend staying in the old town area if you visit.",
                    'comments': [
                        {'author': 'LocalGuide2024', 'body': 'Great post! I lived there for 3 years and completely agree about the old town. Try the morning markets too!', 'score': 45},
                        {'author': 'TravelBuff99', 'body': 'Thanks for sharing! How was the weather when you visited? Planning my trip for next month.', 'score': 23},
                    ]
                },
                {
                    'title': f"Solo travel in {location} - what I learned",
                    'text': f"Solo travel in {location} - here's what I learned after 2 weeks there. Safety was never an issue, people are incredibly friendly, and the public transport is excellent. Best tip: get up early to avoid crowds at major attractions. The food scene is amazing - don't miss the local markets.",
                    'comments': [
                        {'author': 'SoloWanderer', 'body': 'This is so helpful! I was nervous about going alone but your post gives me confidence.', 'score': 67},
                        {'author': 'BackpackerLife', 'body': 'The early morning tip is gold! Did the same thing and had some attractions almost to myself.', 'score': 34},
                    ]
                }
            ],
            'food': [
                {
                    'title': f"Best local restaurants in {location}",
                    'text': f"Best local restaurants in {location} - hidden gems locals actually go to. After living here for 6 months, I've discovered some incredible spots that tourists never find. The family-run place on Market Street has the most authentic cuisine, and the prices are incredibly reasonable. Don't be put off by the basic interior - the food is exceptional.",
                    'comments': [
                        {'author': 'FoodieExplorer', 'body': 'Thank you! Been looking for authentic places. Is Market Street the one near the cathedral?', 'score': 28},
                        {'author': 'LocalFoodie', 'body': 'Can confirm - that family place is incredible. The grandmother still does all the cooking!', 'score': 52},
                    ]
                }
            ],
            'events': [
                {
                    'title': f"Music scene in {location} is incredible",
                    'text': f"Music scene in {location} is incredible - here are the best venues. The underground electronic scene is thriving, and there are amazing jazz clubs in the old quarter. Don't miss the weekend festivals in the park - they're free and feature incredible local talent. The main concert hall also hosts world-class performances.",
                    'comments': [
                        {'author': 'MusicLover23', 'body': 'Love this! Which jazz club would you recommend for a first visit?', 'score': 19},
                        {'author': 'ElectroHead', 'body': 'The underground scene there is legendary. Best clubs in Europe IMO.', 'score': 41},
                    ]
                }
            ]
        }

        category_templates = templates.get(category, templates['travel'])

        for i, template in enumerate(category_templates):
            # Create mock comments
            mock_comments = []
            for j, comment_data in enumerate(template['comments']):
                mock_comment = RedditComment(
                    id=f"mock_comment_{i}_{j}",
                    author=comment_data['author'],
                    body=comment_data['body'],
                    score=comment_data['score'],
                    created_utc=time.time() - (j * 1800),  # 30 min apart
                    is_top_comment=(j == 0)
                )
                mock_comments.append(mock_comment)

            # Create summary
            summary = self._summarize_post(template['title'], template['text'])

            post = LocationPost(
                id=f"mock_{location}_{category}_{i}",
                title=template['title'],
                text=template['text'],
                summary=summary,
                subreddit=subreddit_name,
                author=f"user_{i}",
                score=150 + (i * 20),
                num_comments=len(template['comments']) + 5,
                created_utc=time.time() - (i * 3600),
                url=f"https://reddit.com/r/{subreddit_name}/posts/mock_{i}",
                category=category,
                detected_locations=[location],
                target_location=location,
                hash=generate_hash(f"{template['title']}_{i}"),
                relevancy_score=0.8 + (i * 0.05),
                top_comments=mock_comments
            )

            mock_posts.append(post)

        return mock_posts


if __name__ == "__main__":
    # Full extraction for all locations and categories
    import logging

    logging.basicConfig(level=logging.INFO)

    extractor = ModularRedditExtractor()

    # Check if clients are working
    if extractor.reddit:
        print("Reddit API client is working!")
    else:
        print("Reddit API client not working - using mock data")

    if extractor.s3_client:
        print("S3 client is working!")
    else:
        print("S3 client not available - using local storage")

    print(f"\nStarting comprehensive extraction for all {len(extractor.top_destinations)} destinations")

    extraction_summary = {
        'total_posts': 0,
        'by_location': {},
        'by_category': {'travel': 0, 'food': 0, 'events': 0}
    }

    for location in extractor.top_destinations:
        print(f"\nProcessing {location}...")

        location_summary = {'travel': 0, 'food': 0, 'events': 0}

        for category in ['travel', 'food', 'events']:
            print(f"  Extracting {category} data for {location}")

            posts = extractor.extract_location_specific_posts(
                location=location,
                category=category,
                posts_per_subreddit=8  # Reasonable number per subreddit
            )

            if posts:
                extractor.save_to_s3(posts, location, category)
                print(f"    Found {len(posts)} {category} posts for {location}")

                # Show comment extraction details
                total_comments = sum(len(post.top_comments) for post in posts)
                print(f"    Extracted {total_comments} comments total")

                location_summary[category] = len(posts)
                extraction_summary['by_category'][category] += len(posts)
                extraction_summary['total_posts'] += len(posts)
            else:
                print(f"    No {category} posts found for {location}")

            # Rate limiting between categories
            time.sleep(3)

        extraction_summary['by_location'][location] = location_summary
        total_for_location = sum(location_summary.values())
        print(f"  Total for {location}: {total_for_location} posts")

        # Longer pause between locations to respect Reddit API limits
        time.sleep(5)

    # Save extraction summary
    extractor._save_extraction_summary(extraction_summary)

    print(f"\nExtraction complete!")
    print(f"Total posts extracted: {extraction_summary['total_posts']}")
    print(f"Travel posts: {extraction_summary['by_category']['travel']}")
    print(f"Food posts: {extraction_summary['by_category']['food']}")
    print(f"Events posts: {extraction_summary['by_category']['events']}")

    print(f"\nTop 10 locations by post count:")
    location_totals = {loc: sum(cats.values()) for loc, cats in extraction_summary['by_location'].items()}
    top_locations = sorted(location_totals.items(), key=lambda x: x[1], reverse=True)[:10]
    for loc, count in top_locations:
        print(f"  {loc}: {count} posts")

    print(f"\nExtraction summary saved to S3 and locally")