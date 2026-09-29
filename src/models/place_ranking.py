"""Explainable place ranking; discussion mentions are not endorsements."""

import re
import unicodedata

from ..data_pipeline.storage import is_demo_post
from ..utils.content import normalize_place, post_comments, safe_url
from .trip_preferences import TripPreferences

INTEREST_TYPES = {
    "food": {"restaurant", "cafe", "bakery", "meal_takeaway"},
    "art": {"art_gallery", "museum"},
    "history": {"museum", "historical_landmark", "monument"},
    "nature": {"park", "natural_feature", "botanical_garden", "zoo"},
    "shopping": {"shopping_mall", "store", "market"},
    "nightlife": {"bar", "night_club", "music_venue"},
}
INTEREST_WORDS = {
    "food": {"restaurant", "cafe", "bakery", "food", "market"},
    "art": {"art", "gallery", "museum"},
    "history": {"history", "historic", "monument", "palace", "castle"},
    "nature": {"park", "garden", "nature", "river", "forest"},
    "shopping": {"shopping", "market", "boutique", "store"},
    "nightlife": {"music", "nightlife", "club", "bar"},
}


def normalized_words(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return " ".join(
        re.findall(r"\w+", "".join(char for char in decomposed if not unicodedata.combining(char)))
    )


def discussion_mentions(place: dict, posts: list[dict]) -> list[dict]:
    """Require the full provider name, allowing only case/accent/punctuation changes."""
    name = normalized_words(place["name"])
    if len(name) < 5 or name in {"restaurant", "museum", "gallery", "market", "garden"}:
        return []
    mentions = {}
    for post in posts:
        if is_demo_post(post) or safe_url(post.get("url")) == "#":
            continue
        comments = " ".join(comment["body"] for comment in post_comments(post))
        text = normalized_words(f"{post.get('title', '')} {post.get('text', '')} {comments}")
        if f" {name} " in f" {text} ":
            mentions[post["url"]] = {
                "url": post["url"],
                "title": post.get("title", "Community discussion"),
                "sentiment_label": post.get("sentiment_label", "neutral"),
            }
    return list(mentions.values())


def rank_places(
    restaurants: list[dict],
    attractions: list[dict],
    preferences: TripPreferences,
    posts: list[dict],
    *,
    include_demo: bool = False,
) -> tuple[list[dict], int]:
    ranked, seen_ids, seen_names = [], set(), set()
    excluded = 0
    for activity_type, places in (("restaurant", restaurants), ("attraction", attractions)):
        for original in places:
            if (
                not isinstance(original, dict)
                or not isinstance(original.get("name"), str)
                or not original["name"].strip()
            ):
                continue
            if (
                original.get("source") == "demo" or str(original.get("place_id", "")).startswith("demo:")
            ) and not include_demo:
                continue
            # Normalization must preserve an already-normalized address and website.
            place = {
                **original,
                **normalize_place(
                    original,
                    {
                        **original,
                        "formatted_address": original.get("address", original.get("formatted_address")),
                        "formatted_phone_number": original.get(
                            "phone", original.get("formatted_phone_number")
                        ),
                    },
                ),
            }
            name = normalized_words(place["name"])
            identity = place.get("place_id") or name
            if identity in seen_ids or name in seen_names:
                continue
            seen_ids.add(identity)
            seen_names.add(name)
            level = place["price_level"]
            if (
                preferences.max_price_level is not None
                and level is not None
                and level > preferences.max_price_level
            ):
                excluded += 1
                continue
            types = set(place.get("types", []))
            words = set(name.split())
            matches = [
                interest
                for interest in preferences.interests
                if types & INTEREST_TYPES[interest] or words & INTEREST_WORDS[interest]
            ]
            mentions = discussion_mentions(place, posts)
            reasons = [f"Matches {interest} interest" for interest in matches]
            warnings = []
            if not reasons:
                reasons.append("Available place; no selected interest match is established")
            if mentions:
                reasons.append(
                    f"Named in {len(mentions)} retrieved discussion(s); not a verified endorsement"
                )
            if any(mention["sentiment_label"] == "negative" for mention in mentions):
                warnings.append("Mentioned in a negative discussion; read the source before deciding.")
            if level is None:
                warnings.append("Price tier unavailable; budget compatibility is unverified.")
            score = len(matches) + (place["rating"] or 0) / 50 + min(len(mentions), 3) / 100
            ranked.append(
                {
                    **place,
                    "type": activity_type,
                    "ranking_score": score,
                    "match_reasons": reasons,
                    "community_mentions": mentions,
                    "warnings": warnings,
                }
            )
    ranked.sort(key=lambda place: (-place["ranking_score"], normalized_words(place["name"])))
    return ranked, excluded
