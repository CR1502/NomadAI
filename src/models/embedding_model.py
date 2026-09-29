"""
Embedding model for converting text to vector representations.
Uses sentence-transformers for semantic similarity search.
"""

import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone

from ..utils.content import post_comments

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Creates and manages text embeddings for lifestyle content."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", *, encoder=None):
        """Initialize the embedding model."""
        self.model_name = model_name
        self.model = encoder
        self.embeddings = None
        self.texts = None
        self.metadata = None

        if self.model is None:
            self.load_model()

    def load_model(self):
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

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

        return self.create_embeddings(posts)

    def create_embeddings(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Index request-local posts, retaining their evidence and normalized vectors."""
        logger.info(f"Loaded {len(posts)} posts for embedding")

        # Prepare texts for embedding
        texts = []
        metadata = []

        for post in posts:
            # Combine title and cleaned text for better context
            comments = ' '.join(comment['body'] for comment in post_comments(post)[:3])
            combined_text = f"{post.get('title', '')} {post.get('cleaned_text') or post.get('text', '')} {comments}"
            texts.append(combined_text.strip())

            # Store metadata for each post
            metadata.append(dict(post))

        # Create embeddings
        logger.info("Creating embeddings (this may take a few minutes)...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        ) if texts else np.empty((0, 0))
        embeddings = self._normalize_vectors(embeddings)
        if len(embeddings) != len(metadata):
            raise ValueError('The encoder returned an inconsistent number of vectors')

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
            'created_at': datetime.now(timezone.utc).isoformat()
        }

    @staticmethod
    def _normalize_vectors(vectors):
        vectors = np.asarray(vectors, dtype=float)
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError('Embeddings must be a finite two-dimensional numeric array')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)

    def save_embeddings(self, data: Dict[str, Any], filename: str):
        """Save embeddings and metadata to file."""
        path = Path(filename)
        if path.suffix != '.npz':
            raise ValueError('Use .npz archives; pickle embedding files are no longer supported')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as handle:
            np.savez_compressed(
                handle, embeddings=self._normalize_vectors(data['embeddings']),
                texts=np.asarray(data['texts'], dtype=str), metadata=np.asarray(json.dumps(data['metadata'])),
                model_name=np.asarray(data.get('model_name', self.model_name)),
                created_at=np.asarray(data.get('created_at', datetime.now(timezone.utc).isoformat())),
            )

        logger.info(f"Saved embeddings to {filename}")

    def load_embeddings(self, filename: str) -> Dict[str, Any]:
        """Load embeddings and metadata from file."""
        try:
            if Path(filename).suffix != '.npz':
                raise ValueError('Rebuild legacy pickle files as .npz archives')
            with np.load(filename, allow_pickle=False) as archive:
                data = {
                    'embeddings': self._normalize_vectors(archive['embeddings']),
                    'texts': archive['texts'].tolist(), 'metadata': json.loads(str(archive['metadata'].item())),
                    'model_name': str(archive['model_name'].item()), 'created_at': str(archive['created_at'].item()),
                }
            if data['model_name'] != self.model_name:
                raise ValueError('The saved index uses a different embedding model')
            if len(data['embeddings']) != len(data['texts']) or len(data['texts']) != len(data['metadata']):
                raise ValueError('The saved index has inconsistent metadata lengths')

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

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError('top_k must be a non-negative integer')
        if top_k == 0 or self.embeddings is None or self.texts is None or not len(self.embeddings):
            logger.error("No embeddings loaded. Call create_embeddings_from_posts() first.")
            return []

        # Create query embedding
        query_embedding = self._normalize_vectors(self.model.encode([query], convert_to_numpy=True))
        if query_embedding.shape != (1, self.embeddings.shape[1]):
            raise ValueError('Query and index embedding dimensions differ')

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
                locations = [metadata['target_location']] if metadata.get('target_location') else (
                    metadata.get('locations') or metadata.get('detected_locations') or []
                )
                if not any(location_filter.casefold() == loc.casefold() for loc in locations):
                    continue

            # Quality filter
            if min_quality_score is not None and metadata.get('quality_score', 0) < min_quality_score:
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
            locations = [metadata['target_location']] if metadata.get('target_location') else (
                metadata.get('locations') or metadata.get('detected_locations') or []
            )
            if any(location.casefold() == loc.casefold() for loc in locations):
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
            'average_quality_score': float(np.mean([m.get('quality_score', 0) for m in self.metadata])) if self.metadata else 0.0
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
        'models/compressed/lifestyle_embeddings.npz'
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
    print("\n=== Embeddings Statistics ===")
    stats = embedding_model.get_embeddings_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\nEmbedding model test complete!")
