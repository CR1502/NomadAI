"""Pure helpers shared by ingestion, analysis, and presentation."""

import math
import re
from html import escape
from typing import Any
from urllib.parse import urlsplit

COST_COMPONENTS = ('accommodation', 'food', 'transport', 'attractions')
POSITIVE_TERMS = (
    'highly recommend', 'must visit', 'hidden gem', 'worth it', 'amazing', 'incredible',
    'fantastic', 'love', 'perfect', 'recommend', 'beautiful', 'wonderful', 'excellent',
    'authentic', 'favorite', 'brilliant', 'stunning', 'unforgettable', 'magical',
    'breathtaking', 'outstanding', 'great',
)
NEGATIVE_TERMS = (
    'waste of money', 'tourist trap', 'poor service', 'not worth', 'avoid', 'terrible',
    'worst', 'disappointing', 'overrated', 'skip', 'bad', 'awful', 'overpriced',
    'crowded', 'dirty', 'rude', 'scam', 'boring', 'mediocre',
)
NEGATION = re.compile(
    r"\b(?:not|no|never|cannot|hardly|don['’]?t|didn['’]?t|wouldn['’]?t|isn['’]?t|wasn['’]?t)"
    r"(?:\W+\w+){0,3}\W*$",
    re.IGNORECASE,
)


def safe_html(value: Any) -> str:
    return escape('' if value is None else str(value), quote=True)


def safe_url(value: Any) -> str:
    """Allow only absolute HTTP(S) links, escaped for an HTML attribute."""
    if not isinstance(value, str) or any(ord(char) < 32 for char in value):
        return '#'
    try:
        parsed = urlsplit(value)
    except ValueError:
        return '#'
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        return '#'
    return safe_html(value)


def post_comments(post: dict) -> list[dict]:
    """Ignore malformed optional comments instead of crashing retrieval or planning."""
    comments = post.get('top_comments')
    if not isinstance(comments, list):
        return []
    return [comment for comment in comments if isinstance(comment, dict) and isinstance(comment.get('body'), str)]


def calculate_costs(components: dict[str, int | float]) -> dict[str, int | float]:
    """Derive totals from the displayed components; do not trust a separate total."""
    costs = {name: components[name] for name in COST_COMPONENTS}
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
           for value in costs.values()):
        raise ValueError('Cost components must be finite, non-negative numbers')
    return {**costs, 'daily': sum(costs.values())}


def classify_sentiment(text: str, polarity: float = 0.0) -> dict[str, Any]:
    """Combine polarity with word-boundary cues, reversing explicitly negated cues.

    This is an interpretable baseline, not a calibrated confidence estimate.
    """
    positive = negative = 0
    for terms, direction in ((POSITIVE_TERMS, 1), (NEGATIVE_TERMS, -1)):
        pattern = r'(?<!\w)(?:' + '|'.join(re.escape(term) for term in terms) + r')(?!\w)'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            effective_direction = -direction if NEGATION.search(text[max(0, match.start() - 60):match.start()]) else direction
            if effective_direction > 0:
                positive += 1
            else:
                negative += 1
    cue_count = positive + negative
    polarity = max(-1.0, min(1.0, polarity))
    score = 0.8 * (positive - negative) / cue_count + 0.2 * polarity if cue_count else polarity
    label = 'positive' if score > 0.15 else 'negative' if score < -0.15 else 'neutral'
    return {
        'sentiment_score': score,
        'sentiment_label': label,
        'positive_indicators': positive,
        'negative_indicators': negative,
    }


def normalize_place(place: dict, details: dict | None = None) -> dict:
    """Keep absent provider values absent rather than fabricating ratings or prices."""
    details = details or {}
    price_level = details.get('price_level', place.get('price_level'))
    if isinstance(price_level, bool) or not isinstance(price_level, int) or price_level not in range(5):
        price_level = None
    rating = details.get('rating', place.get('rating'))
    if isinstance(rating, bool) or not isinstance(rating, (int, float)) or not math.isfinite(rating) or not 0 <= rating <= 5:
        rating = None
    review_count = details.get('user_ratings_total', place.get('user_ratings_total', 0))
    if isinstance(review_count, bool) or not isinstance(review_count, int) or review_count < 0:
        review_count = 0
    types = details.get('types', place.get('types', []))
    types = [value for value in types if isinstance(value, str)] if isinstance(types, (list, tuple)) else []
    return {
        'place_id': place.get('place_id'),
        'name': details.get('name') or place.get('name', 'Unnamed place'),
        'rating': rating,
        'user_ratings_total': review_count,
        'price_level': price_level,
        'price': ('Free', '$', '$$', '$$$', '$$$$')[price_level] if price_level is not None else 'Price unavailable',
        'address': details.get('formatted_address') or place.get('vicinity', 'Address not available'),
        'website': details.get('website', 'Not available'),
        'phone': details.get('formatted_phone_number', 'Not available'),
        'types': types,
    }
