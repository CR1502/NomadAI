from copy import deepcopy
from unittest.mock import Mock

import pytest

from src.models.ai_trip_planner import AITripPlanner
from src.models.place_ranking import discussion_mentions, rank_places
from src.models.post_retriever import PostRetriever
from src.models.trip_preferences import TripPreferences
from src.services.demo_data import demo_places
from src.services.trip_planning import TripPlanningService


@pytest.fixture
def places():
    return {
        "restaurants": [
            {"place_id": "cafe", "name": "Market Cafe", "types": ["restaurant"], "price_level": 1},
            {"place_id": "dining", "name": "Fine Dining", "types": ["restaurant"], "price_level": 4},
        ],
        "attractions": [
            {"place_id": "museum", "name": "Test Art Museum", "types": ["museum"], "price_level": 2},
            {"place_id": "park", "name": "Riverside Park", "types": ["park"], "price_level": 0},
            {"place_id": "gallery", "name": "Local Art Gallery", "types": ["art_gallery"]},
            {"place_id": "monument", "name": "Historic Monument", "types": ["historical_landmark"]},
        ],
    }


@pytest.mark.parametrize(
    "invalid",
    [
        {"duration": False},
        {"duration": 0},
        {"duration": 31},
        {"duration": 2.5},
        {"pace": "fast"},
        {"pace": []},
        {"interests": "food"},
        {"interests": ["unknown"]},
        {"interests": [1]},
        {"daily_budget": 0},
        {"daily_budget": -1},
        {"daily_budget": True},
        {"daily_budget": "50"},
        {"daily_budget": float("nan")},
        {"daily_budget": float("inf")},
        {"daily_budget": 100_001},
        {"max_price_level": True},
        {"max_price_level": -1},
        {"max_price_level": 5},
        {"max_price_level": 2.5},
        {"unexpected": "value"},
    ],
)
def test_preferences_reject_invalid_or_unsupported_values(invalid):
    with pytest.raises(ValueError):
        TripPreferences.from_dict(invalid)


def test_preferences_are_canonical_and_serializable():
    preferences = TripPreferences.from_dict({"interests": ["nature", "art", "art"], "max_price_level": 0})
    assert preferences.interests == ("art", "nature")
    assert preferences.to_dict()["interests"] == ["art", "nature"]
    assert preferences.activities_per_day == 3
    assert "garden" in preferences.retrieval_query("Paris")


def test_interests_change_place_ranking_without_mutating_provider_data(places):
    original = deepcopy(places)
    art, _ = rank_places([], places["attractions"], TripPreferences(interests=["art"]), [])
    nature, _ = rank_places([], places["attractions"], TripPreferences(interests=["nature"]), [])
    assert art[0]["name"] in {"Test Art Museum", "Local Art Gallery"}
    assert nature[0]["name"] == "Riverside Park"
    assert "Matches nature interest" in nature[0]["match_reasons"]
    assert places == original


def test_known_price_limits_exclude_expensive_places_and_preserve_unknowns(places):
    ranked, excluded = rank_places(
        places["restaurants"], places["attractions"], TripPreferences(max_price_level=0), []
    )
    assert excluded == 3
    assert {place["name"] for place in ranked} == {"Riverside Park", "Local Art Gallery", "Historic Monument"}
    assert all(place["warnings"] for place in ranked if place["price_level"] is None)


def test_mentions_require_full_names_and_valid_links_and_do_not_claim_endorsements():
    place = {"name": "Cafe Musée"}
    posts = [
        {
            "id": "match",
            "url": "https://example.com/match",
            "text": "CAFE-MUSEE was overpriced.",
            "sentiment_label": "negative",
        },
        {"id": "partial", "url": "https://example.com/partial", "text": "The cafe was good."},
        {"id": "boundary", "url": "https://example.com/boundary", "text": "Cafe Muséeology"},
        {"id": "unsafe", "url": "javascript:alert(1)", "text": "Cafe Musée"},
        {"id": "mock_match", "url": "https://example.com/demo", "text": "Cafe Musée"},
    ]
    mentions = discussion_mentions(place, posts)
    assert [mention["url"] for mention in mentions] == ["https://example.com/match"]
    ranked, _ = rank_places([place], [], TripPreferences(), posts)
    assert "not a verified endorsement" in " ".join(ranked[0]["match_reasons"])
    assert any("negative discussion" in warning for warning in ranked[0]["warnings"])


def test_mentions_can_come_from_comments():
    posts = [{"url": "https://example.com/comment", "top_comments": [{"body": "Visit Test Museum early."}]}]
    assert discussion_mentions({"name": "Test Museum"}, posts)[0]["url"] == posts[0]["url"]


@pytest.mark.parametrize(("pace", "limit"), [("relaxed", 2), ("balanced", 3), ("busy", 4)])
def test_local_plan_enforces_pace_and_does_not_repeat_places_across_categories(pace, limit, places):
    places["restaurants"].append(places["attractions"][0].copy())
    result = TripPlanningService().generate("Paris", {"duration": 3, "pace": pace}, [], places)
    activities = [activity for day in result["days"] for activity in day["activities"]]
    assert len(result["days"]) == 3
    assert all(len(day["activities"]) <= limit for day in result["days"])
    assert len({activity["place_id"] for activity in activities}) == len(activities) == 6
    assert result["ai_generated"] is False
    assert any("unfilled" in limitation for limitation in result["limitations"]) == (6 < 3 * limit)


def test_local_plan_has_budget_caveats_not_invented_estimates(places):
    result = TripPlanningService().generate(
        "Paris", {"duration": 1, "daily_budget": 20, "max_price_level": 0}, [], places
    )
    notes = " ".join(result["budget_notes"])
    assert "$20 USD" in notes and "below the illustrative" in notes
    assert "not USD amounts" in notes
    assert all(activity["price_level"] in (None, 0) for activity in result["days"][0]["activities"])


def test_empty_data_does_not_invent_places_or_call_ai(monkeypatch):
    setup = Mock()
    monkeypatch.setattr(AITripPlanner, "setup_openai", setup)
    result = TripPlanningService(api_key="configured").generate("Paris", {"duration": 2}, [], None)
    setup.assert_not_called()
    assert result["days"] == [
        {"day": 1, "title": "Day 1 in Paris", "activities": []},
        {"day": 2, "title": "Day 2 in Paris", "activities": []},
    ]
    assert result["reddit_tips"] == []


def test_demo_places_are_explicitly_isolated_and_ai_cannot_run(monkeypatch):
    setup = Mock()
    monkeypatch.setattr(AITripPlanner, "setup_openai", setup)
    places = demo_places("Paris")
    live = TripPlanningService().generate("Paris", {"duration": 1}, [], places)
    demo = TripPlanningService().generate("Paris", {"duration": 1}, [], places, demo_mode=True, use_ai=True)
    setup.assert_not_called()
    assert live["days"][0]["activities"] == []
    assert len(demo["days"][0]["activities"]) == 3
    assert all(
        activity["place_id"].startswith("demo:") and not activity["source_urls"]
        for activity in demo["days"][0]["activities"]
    )
    assert demo["demo_mode"] is True and demo["ai_generated"] is False


def test_retrieval_processes_raw_posts_and_preserves_evidence(sample_posts):
    original = deepcopy(sample_posts)
    result = TripPlanningService().generate(
        "Paris", {"duration": 1, "interests": ["food"]}, sample_posts, None
    )
    retrieved = result["retrieved_posts"]
    assert len(retrieved) == 2
    assert {post["id"] for post in retrieved} == {post["id"] for post in sample_posts}
    assert retrieved[0]["quality_score"] >= 40
    positive = next(post for post in retrieved if post["id"] == sample_posts[0]["id"])
    assert positive["url"] == sample_posts[0]["url"]
    assert positive["top_comments"] == sample_posts[0]["top_comments"]
    assert result["retrieval_method"] == "tfidf"
    assert sample_posts == original


def test_retrieval_changes_with_interests_and_filters_destinations_quality_and_demo():
    posts = [
        {
            "id": "art",
            "title": "Museum art gallery exhibition",
            "quality_score": 80,
            "target_location": "Paris",
        },
        {
            "id": "nature",
            "title": "Park garden outdoors nature hiking",
            "quality_score": 80,
            "target_location": "Paris",
        },
        {"id": "other", "title": "Museum art gallery", "quality_score": 100, "target_location": "Tokyo"},
        {"id": "low", "title": "Museum art gallery", "quality_score": 10, "target_location": "Paris"},
        {"id": "mock_art", "title": "Museum art gallery", "quality_score": 100, "target_location": "Paris"},
    ]
    retriever = PostRetriever()
    art = retriever.retrieve(posts, "Paris", TripPreferences(interests=["art"]))
    nature = retriever.retrieve(posts, "Paris", TripPreferences(interests=["nature"]))
    assert art[0]["id"] == "art" and nature[0]["id"] == "nature"
    assert {post["id"] for post in art} == {"art", "nature"}
    assert len(retriever.retrieve(posts, "Paris", TripPreferences(), top_k=1)) == 1


def test_retrieval_requires_exact_destination_metadata():
    posts = [{"id": "wrong", "title": "Paris advice", "quality_score": 100, "locations": ["Parisian"]}]
    assert PostRetriever().retrieve(posts, "Paris", TripPreferences()) == []


def test_empty_vocabulary_has_an_explicit_quality_baseline():
    posts = [{"id": "empty", "title": "the and", "quality_score": 50, "target_location": "Paris"}]
    found = PostRetriever().retrieve(posts, "Paris", TripPreferences())
    assert found[0]["retrieval_similarity"] == 0
    assert found[0]["retrieval_score"] == pytest.approx(0.075)


@pytest.mark.parametrize("comments", [None, "not a list", [None, "bad", {"body": None}]])
def test_malformed_optional_comments_and_types_do_not_crash_planning(comments):
    posts = [
        {
            "id": "test",
            "title": "Paris Test Museum",
            "text": None,
            "top_comments": comments,
            "enhanced_quality_score": None,
            "quality_score": 80,
            "target_location": "Paris",
            "url": "https://example.com/museum",
            "enhanced_sentiment": None,
        }
    ]
    places = {"attractions": [{"name": "Test Museum", "types": None}]}
    result = TripPlanningService().generate("Paris", {"duration": 1}, posts, places)
    assert len(result["retrieved_posts"]) == 1
    assert result["days"][0]["activities"][0]["source_urls"] == ["https://example.com/museum"]
    assert AITripPlanner(use_ai=False).create_reddit_context(result["retrieved_posts"], "Paris")


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_retrieval_limit_must_be_positive(top_k):
    with pytest.raises(ValueError):
        PostRetriever().retrieve([], "Paris", TripPreferences(), top_k=top_k)
