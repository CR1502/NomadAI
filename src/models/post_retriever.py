"""Preference-driven retrieval with a no-download TF-IDF baseline."""

import math
import re
from dataclasses import asdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..data_pipeline.data_processor import DataProcessor
from ..data_pipeline.storage import deduplicate_posts
from ..utils.content import post_comments
from .trip_preferences import TripPreferences


def belongs_to_destination(post: dict, location: str) -> bool:
    target = post.get("target_location")
    if target:
        return isinstance(target, str) and target.casefold() == location.casefold()
    locations = post.get("locations") or post.get("detected_locations") or []
    if locations:
        return any(isinstance(place, str) and place.casefold() == location.casefold() for place in locations)
    return bool(
        re.search(
            r"(?<!\w)" + re.escape(location) + r"(?!\w)",
            f"{post.get('title', '')} {post.get('text', '')}",
            re.IGNORECASE,
        )
    )


class PostRetriever:
    def __init__(self, min_quality: float = 40, *, embedding_model=None):
        self.min_quality = min_quality
        self.embedding_model = embedding_model

    def retrieve(
        self,
        posts: list[dict],
        location: str,
        preferences: TripPreferences,
        *,
        top_k: int = 15,
        include_demo: bool = False,
    ) -> list[dict]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        candidates = []
        processor = None
        for post in deduplicate_posts(posts, include_demo=include_demo):
            if not belongs_to_destination(post, location):
                continue
            quality = post.get("enhanced_quality_score")
            if quality is None:
                quality = post.get("quality_score")
            if quality is None:
                processor = processor or DataProcessor()
                processed = processor.process_reddit_posts([post])
                if not processed:
                    continue
                record = asdict(processed[0])
                post = {**post, **record, "timestamp": record["timestamp"].isoformat()}
                quality = post["quality_score"]
            if (
                isinstance(quality, bool)
                or not isinstance(quality, (int, float))
                or not math.isfinite(quality)
            ):
                continue
            if quality < self.min_quality or quality > 100:
                continue
            candidates.append(
                {**post, "quality_score": quality, "target_location": post.get("target_location") or location}
            )
        if not candidates:
            return []
        query = preferences.retrieval_query(location)
        if self.embedding_model is not None:
            self.embedding_model.create_embeddings(candidates)
            results = self.embedding_model.find_similar(
                query, top_k=len(candidates), location_filter=location
            )
            scores = {result["metadata"]["id"]: result["similarity"] for result in results}
            similarities = [scores.get(post.get("id"), 0.0) for post in candidates]
            method = "semantic"
        else:
            texts = [
                f"{post.get('title', '')} {post.get('cleaned_text') or post.get('text', '')} "
                + " ".join(comment["body"] for comment in post_comments(post)[:3])
                for post in candidates
            ]
            try:
                vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10_000)
                matrix = vectorizer.fit_transform(texts)
                similarities = cosine_similarity(vectorizer.transform([query]), matrix)[0]
            except ValueError:  # Empty or stop-word-only documents still have a quality baseline.
                similarities = [0.0] * len(candidates)
            method = "tfidf"
        ranked = [
            {
                **post,
                "retrieval_similarity": float(similarity),
                "retrieval_score": 0.85 * float(similarity) + 0.15 * post["quality_score"] / 100,
                "retrieval_method": method,
            }
            for post, similarity in zip(candidates, similarities, strict=True)
        ]
        return sorted(ranked, key=lambda post: (-post["retrieval_score"], str(post.get("id", ""))))[:top_k]
