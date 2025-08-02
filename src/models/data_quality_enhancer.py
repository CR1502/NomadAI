"""
Advanced data quality enhancement with ML-based sentiment analysis,
named entity recognition, and duplicate detection.
"""

import spacy
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import json
import logging
import re
from fuzzywuzzy import fuzz
from collections import Counter
import hashlib

logger = logging.getLogger(__name__)


class DataQualityEnhancer:
    """Advanced data quality enhancement using ML and NLP."""

    def __init__(self):
        """Initialize the data quality enhancer."""
        self.nlp = None
        self.load_nlp_model()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )

    def load_nlp_model(self):
        """Load spaCy NLP model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            logger.info("Please install: python -m spacy download en_core_web_sm")

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
            'hidden gem': 0.9, 'authentic': 0.7, 'local favorite': 0.8, 'worth it': 0.7,
            'unforgettable': 0.9, 'breathtaking': 1.0, 'magical': 0.8
        }

        # Travel-specific negative words with weights
        negative_words = {
            'tourist trap': -1.0, 'overrated': -0.8, 'overpriced': -0.7, 'crowded': -0.5,
            'disappointing': -0.9, 'waste of money': -1.0, 'avoid': -1.0, 'terrible': -0.9,
            'awful': -0.8, 'dirty': -0.6, 'rude': -0.6, 'scam': -1.0,
            'not worth': -0.8, 'skip': -0.7, 'boring': -0.5
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
            return {'locations': [], 'organizations': [], 'persons': []}

        doc = self.nlp(text)

        entities = {
            'locations': [],
            'organizations': [],
            'persons': [],
            'money': [],
            'dates': []
        }

        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC']:  # Geopolitical entity, Location
                entities['locations'].append(ent.text.strip())
            elif ent.label_ in ['ORG']:  # Organization
                entities['organizations'].append(ent.text.strip())
            elif ent.label_ in ['PERSON']:  # Person
                entities['persons'].append(ent.text.strip())
            elif ent.label_ in ['MONEY']:  # Money
                entities['money'].append(ent.text.strip())
            elif ent.label_ in ['DATE', 'TIME']:  # Date/Time
                entities['dates'].append(ent.text.strip())

        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities

    def detect_duplicates(self, posts: List[Dict[str, Any]], similarity_threshold: float = 0.85) -> List[
        Dict[str, Any]]:
        """Detect and handle duplicate posts using advanced similarity."""
        if len(posts) < 2:
            return posts

        # Create text representations
        texts = []
        for post in posts:
            text = f"{post.get('title', '')} {post.get('text', '')} {post.get('summary', '')}"
            texts.append(text)

        # Calculate TF-IDF similarity
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except Exception as e:
            logger.warning(f"TF-IDF similarity calculation failed: {e}")
            return posts

        # Find duplicates
        duplicates = set()
        unique_posts = []

        for i in range(len(posts)):
            if i in duplicates:
                continue

            # Check similarity with remaining posts
            current_post = posts[i].copy()
            similar_posts = [current_post]

            for j in range(i + 1, len(posts)):
                if j in duplicates:
                    continue

                # Multiple similarity checks
                tfidf_sim = similarity_matrix[i][j]
                title_sim = fuzz.ratio(posts[i].get('title', ''), posts[j].get('title', '')) / 100.0

                if tfidf_sim > similarity_threshold or title_sim > 0.9:
                    duplicates.add(j)
                    similar_posts.append(posts[j])

            # Merge similar posts (keep highest scoring one, combine insights)
            if len(similar_posts) > 1:
                best_post = max(similar_posts, key=lambda x: x.get('score', 0))
                best_post['duplicate_count'] = len(similar_posts)
                best_post['combined_score'] = sum(p.get('score', 0) for p in similar_posts)
                unique_posts.append(best_post)
            else:
                current_post['duplicate_count'] = 1
                current_post['combined_score'] = current_post.get('score', 0)
                unique_posts.append(current_post)

        logger.info(f"Deduplication: {len(posts)} -> {len(unique_posts)} posts ({len(duplicates)} duplicates removed)")
        return unique_posts

    def calculate_quality_score(self, post: Dict[str, Any], enhanced_sentiment: Dict[str, Any],
                                entities: Dict[str, List]) -> float:
        """Calculate comprehensive quality score for posts."""
        score = 0.0

        # Text quality (0-25 points)
        text_length = len(post.get('text', ''))
        if 100 <= text_length <= 1000:
            score += 25
        elif 50 <= text_length < 100 or 1000 < text_length <= 2000:
            score += 15
        elif 20 <= text_length < 50:
            score += 10

        # Reddit engagement (0-25 points)
        reddit_score = post.get('score', 0)
        if reddit_score >= 500:
            score += 25
        elif reddit_score >= 100:
            score += 20
        elif reddit_score >= 50:
            score += 15
        elif reddit_score >= 10:
            score += 10

        # Comment engagement (0-15 points)
        num_comments = post.get('num_comments', 0)
        if num_comments >= 100:
            score += 15
        elif num_comments >= 50:
            score += 12
        elif num_comments >= 20:
            score += 8
        elif num_comments >= 5:
            score += 5

        # Sentiment quality (0-20 points)
        sentiment_confidence = enhanced_sentiment.get('confidence', 0)
        score += sentiment_confidence * 20

        # Entity richness (0-10 points)
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        score += min(total_entities * 2, 10)

        # Relevancy bonus (0-5 points)
        relevancy = post.get('relevancy_score', 0)
        score += relevancy * 5

        return min(score, 100.0)

    def enhance_posts_quality(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply all quality enhancements to posts."""
        enhanced_posts = []

        logger.info(f"Enhancing quality for {len(posts)} posts...")

        for i, post in enumerate(posts):
            try:
                text = f"{post.get('title', '')} {post.get('text', '')} {post.get('summary', '')}"

                # Enhanced sentiment analysis
                sentiment = self.enhanced_sentiment_analysis(text)

                # Named entity recognition
                entities = self.extract_named_entities(text)

                # Calculate enhanced quality score
                quality_score = self.calculate_quality_score(post, sentiment, entities)

                # Update post with enhancements
                enhanced_post = post.copy()
                enhanced_post.update({
                    'enhanced_sentiment': sentiment,
                    'entities': entities,
                    'enhanced_quality_score': quality_score,
                    'processing_timestamp': pd.Timestamp.now().isoformat()
                })

                enhanced_posts.append(enhanced_post)

                # Progress tracking
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(posts)} posts")

            except Exception as e:
                logger.error(f"Error enhancing post {post.get('id', 'unknown')}: {e}")
                enhanced_posts.append(post)  # Keep original if enhancement fails

        # Apply deduplication
        unique_posts = self.detect_duplicates(enhanced_posts)

        # Sort by enhanced quality score
        unique_posts.sort(key=lambda x: x.get('enhanced_quality_score', 0), reverse=True)

        logger.info(f"✅ Quality enhancement complete: {len(unique_posts)} high-quality posts")
        return unique_posts


if __name__ == "__main__":
    # Test the data quality enhancer
    enhancer = DataQualityEnhancer()

    # Load sample data
    with open('data/processed/all_processed_posts.json', 'r') as f:
        sample_posts = json.load(f)[:100]  # Test with 100 posts

    print(f"Testing with {len(sample_posts)} posts...")

    # Enhance quality
    enhanced_posts = enhancer.enhance_posts_quality(sample_posts)

    print(f"Enhanced to {len(enhanced_posts)} high-quality posts")

    # Show sample enhancement
    if enhanced_posts:
        sample = enhanced_posts[0]
        print(f"\nSample enhancement:")
        print(f"Title: {sample['title']}")
        print(f"Enhanced sentiment: {sample['enhanced_sentiment']}")
        print(f"Entities found: {sample['entities']}")
        print(f"Quality score: {sample['enhanced_quality_score']:.1f}")