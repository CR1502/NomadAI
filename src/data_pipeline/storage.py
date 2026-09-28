"""Read the extractor's actual file layout and retain source provenance."""

import json
import math
import re
from pathlib import Path

from ..utils.helpers import PROJECT_ROOT, generate_hash

CATEGORIES = ('travel', 'food', 'events')
DEFAULT_DATA_ROOT = PROJECT_ROOT / 'data' / 'by_location'


def location_slug(location: str) -> str:
    slug = location.strip().lower().replace(' ', '_')
    if not re.fullmatch(r'[a-z0-9]+(?:_[a-z0-9]+)*', slug):
        raise ValueError(f'Invalid destination: {location!r}')
    return slug


def is_demo_post(post: dict) -> bool:
    return post.get('source') == 'demo' or str(post.get('id', '')).startswith('mock_')


def deduplicate_posts(posts: list[dict], *, include_demo: bool = False) -> list[dict]:
    unique = {}
    for original in posts:
        if not isinstance(original, dict):
            raise ValueError('Post records must be JSON objects')
        if is_demo_post(original) and not include_demo:
            continue
        post = {**original, 'source': 'demo' if is_demo_post(original) else original.get('source', 'reddit')}
        if post.get('id') is not None and not isinstance(post['id'], str):
            raise ValueError('Post IDs must be strings')
        score = post.get('score', 0)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError('Post scores must be finite numbers')
        key = post.get('id') or generate_hash(f"{post.get('title', '')}\n{post.get('text', '')}")
        if key not in unique or score > unique[key].get('score', 0):
            unique[key] = post
    return list(unique.values())


def read_posts(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as handle:
        posts = json.load(handle)
    if not isinstance(posts, list) or any(not isinstance(post, dict) for post in posts):
        raise ValueError(f'{path} must contain a list of post objects')
    return posts


def validate_extraction_summary(summary: dict, *, include_demo: bool = False) -> dict:
    """Validate counts before they reach dashboard metrics."""
    if not isinstance(summary, dict) or not isinstance(summary.get('by_location'), dict):
        raise ValueError('Extraction summary must contain a by_location object')
    if summary.get('source') == 'demo' and not include_demo:
        raise ValueError('Demo summaries are not available in live mode')
    for location, counts in summary['by_location'].items():
        if not isinstance(location, str) or not isinstance(counts, dict):
            raise ValueError('Destination summaries must contain count objects')
        if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts.values()):
            raise ValueError('Summary counts must be non-negative integers')
    return summary


def load_location_posts(location: str, data_root: Path = DEFAULT_DATA_ROOT, *, include_demo: bool = False) -> list[dict]:
    posts = []
    for category in CATEGORIES:
        path = Path(data_root) / location_slug(location) / category / 'reddit_posts.json'
        if path.is_file():
            posts.extend(read_posts(path))
    return deduplicate_posts(posts, include_demo=include_demo)


def load_extracted_posts(data_root: Path = DEFAULT_DATA_ROOT, *, include_demo: bool = False) -> list[dict]:
    posts = []
    for path in sorted(Path(data_root).glob('*/*/reddit_posts.json')):
        if path.parent.name in CATEGORIES:
            posts.extend(read_posts(path))
    # Retain per-destination associations for posts retrieved for multiple cities.
    by_destination = {}
    for post in posts:
        by_destination.setdefault(post.get('target_location', ''), []).append(post)
    return [post for group in by_destination.values()
            for post in deduplicate_posts(group, include_demo=include_demo)]
