"""
Embedding model for converting text to vector representations.
Uses sentence-transformers for semantic similarity search.
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import pickle
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Creates and manages text embeddings for lifestyle content."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the embedding model."""
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.texts = None
        self.metadata = None

        self.load_model()

    def load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def create_embeddings_from_posts(self, posts_file: str) -> Dict[str, Any]:
        """Create embeddings from processed posts."""
        logger.info(f"Creating embeddings from {posts_file}")

        # Load processed posts
        with open(posts_file, 'r', encoding='utf-8') as f:
            posts = json.load(f)

        logger.info(f"Loaded {len(posts)} posts for embedding")

        # Prepare texts for embedding
        texts = []
        metadata = []

        for post in posts:
            # Combine title and cleaned text for better context
            combined_text = f"{post.get('title', '')} {post.get('cleaned_text', '')}"
            texts.append(combined_text.strip())

            # Store metadata for each post
            metadata.append({
                'id': post.get('id'),
                'title': post.get('title'),
                'category': post.get('category'),
                'locations': post.get('locations', []),
                'sentiment_label': post.get('sentiment_label'),
                'quality_score': post.get('quality_score'),
                'subreddit': post.get('subreddit'),
                'score': post.get('score'),
                'word_count': post.get('word_count')
            })

        # Create embeddings
        logger.info("Creating embeddings (this may take a few minutes)...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Store the data
        self.embeddings = embeddings
        self.texts = texts
        self.metadata = metadata

        logger.info(f"Created {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

        return {
            'embeddings': embeddings,
            'texts': texts,
            'metadata': metadata,
            'model_name': self.model_name,
            'created_at': datetime.now().isoformat()
        }

    def save_embeddings(self, data: Dict[str, Any], filename: str):
        """Save embeddings and metadata to file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Saved embeddings to {filename}")

    def load_embeddings(self, filename: str) -> Dict[str, Any]:
        """Load embeddings and metadata from file."""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            self.embeddings = data['embeddings']
            self.texts = data['texts']
            self.metadata = data['metadata']

            logger.info(f"Loaded {len(self.embeddings)} embeddings from {filename}")
            return data

        except FileNotFoundError:
            logger.error(f"Embeddings file {filename} not found")
            return {}

    def find_similar(
            self,
            query: str,
            top_k: int = 10,
            category_filter: Optional[str] = None,
            location_filter: Optional[str] = None,
            min_quality_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Find similar posts to a query."""

        if self.embeddings is None or self.texts is None:
            logger.error("No embeddings loaded. Call create_embeddings_from_posts() first.")
            return []

        # Create query embedding
        query_embedding = self.model.encode([query])

        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()

        # Get top results with metadata
        results = []
        for i, similarity in enumerate(similarities):
            result = {
                'similarity': float(similarity),
                'text': self.texts[i],
                'metadata': self.metadata[i]
            }
            results.append(result)

        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)

        # Apply filters
        filtered_results = []
        for result in results:
            metadata = result['metadata']

            # Category filter
            if category_filter and metadata.get('category') != category_filter:
                continue

            # Location filter
            if location_filter:
                locations = metadata.get('locations', [])
                if not any(location_filter.lower() in loc.lower() for loc in locations):
                    continue

            # Quality filter
            if min_quality_score and metadata.get('quality_score', 0) < min_quality_score:
                continue

            filtered_results.append(result)

            if len(filtered_results) >= top_k:
                break

        logger.info(f"Found {len(filtered_results)} similar results for query: '{query}'")
        return filtered_results

    def get_category_recommendations(self, category: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Get top recommendations for a specific category."""
        if self.metadata is None:
            return []

        # Filter by category and sort by quality score
        category_posts = []
        for i, metadata in enumerate(self.metadata):
            if metadata.get('category') == category:
                result = {
                    'text': self.texts[i],
                    'metadata': metadata,
                    'quality_score': metadata.get('quality_score', 0)
                }
                category_posts.append(result)

        # Sort by quality score
        category_posts.sort(key=lambda x: x['quality_score'], reverse=True)

        return category_posts[:top_k]

    def get_location_recommendations(self, location: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Get recommendations for a specific location."""
        if self.metadata is None:
            return []

        location_posts = []
        for i, metadata in enumerate(self.metadata):
            locations = metadata.get('locations', [])
            if any(location.lower() in loc.lower() for loc in locations):
                result = {
                    'text': self.texts[i],
                    'metadata': metadata,
                    'quality_score': metadata.get('quality_score', 0)
                }
                location_posts.append(result)

        # Sort by quality score
        location_posts.sort(key=lambda x: x['quality_score'], reverse=True)

        return location_posts[:top_k]

    def get_embeddings_stats(self) -> Dict[str, Any]:
        """Get statistics about the embeddings."""
        if self.embeddings is None:
            return {}

        return {
            'total_embeddings': len(self.embeddings),
            'embedding_dimension': self.embeddings.shape[1],
            'model_name': self.model_name,
            'categories': list(set(m.get('category') for m in self.metadata)),
            'total_locations': len(set(loc for m in self.metadata for loc in m.get('locations', []))),
            'average_quality_score': np.mean([m.get('quality_score', 0) for m in self.metadata])
        }


if __name__ == "__main__":
    # Test the embedding model
    import logging

    logging.basicConfig(level=logging.INFO)

    # Initialize model
    embedding_model = EmbeddingModel()

    # Create embeddings from processed posts
    embeddings_data = embedding_model.create_embeddings_from_posts(
        'data/processed/high_quality_posts.json'
    )

    # Save embeddings
    embedding_model.save_embeddings(
        embeddings_data,
        'models/compressed/lifestyle_embeddings.pkl'
    )

    # Test similarity search
    print("\n=== Testing Similarity Search ===")

    # Test queries
    test_queries = [
        "best restaurants in Tokyo",
        "solo travel in Europe",
        "music festivals in summer",
        "budget backpacking tips"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = embedding_model.find_similar(query, top_k=3)

        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            print(f"  {i}. [{metadata['category']}] {metadata['title'][:60]}...")
            print(f"     Similarity: {result['similarity']:.3f} | Quality: {metadata['quality_score']:.1f}")

    # Show stats
    print(f"\n=== Embeddings Statistics ===")
    stats = embedding_model.get_embeddings_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\nEmbedding model test complete!")