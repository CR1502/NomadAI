import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

from src.models.ai_trip_planner import AITripPlanner
from src.models.local_planner import build_local_itinerary
from src.models.place_ranking import rank_places
from src.models.trip_preferences import TripPreferences
from src.services.place_policy import (
    attribution_html,
    google_maps_link,
    hydrate_itinerary,
    model_place_catalog,
    persistable_itinerary,
)
from src.web.trip_planner import request_fingerprint


def place(**changes):
    return {
        "name": "Google Fixture Museum",
        "place_id": "google-fixture",
        "source": "google_places",
        "address": "Provider-only Address",
        "rating": 4.5,
        "price_level": 2,
        "coordinates": {"latitude": 1, "longitude": 2},
        "time_zone": "Europe/Paris",
        "regular_opening_hours": {"periods": [{"open": {"day": 1, "hour": 9}}]},
        "attributions": [{"provider": "Third Party", "provider_uri": "https://example.org"}],
        **changes,
    }


def plan():
    prefs = TripPreferences(duration=1)
    ranked, _ = rank_places([], [place()], prefs, [])
    result = build_local_itinerary("Paris", prefs, ranked)
    return {**result, "retrieved_posts": [], "retrieval_method": "tfidf", "demo_mode": False}


def test_persisted_draft_keeps_google_ids_but_no_provider_or_derived_content():
    result = plan()
    result["days"][0]["title"] = "Google Fixture Museum day"
    result["budget_notes"].append("Google Fixture Museum costs 100 dollars")
    result["reddit_tips"].append("Google Fixture Museum")
    result["provider_payload"] = place()
    before = deepcopy(result)
    saved = persistable_itinerary(result)
    encoded = json.dumps(saved)
    assert "Google Fixture Museum" not in encoded and "Provider-only Address" not in encoded
    assert "coordinates" not in encoded and "Europe/Paris" not in encoded
    assert "regular_opening_hours" not in encoded and "Third Party" not in encoded
    assert "provider_payload" not in saved
    activity = saved["days"][0]["activities"][0]
    assert activity["place_id"] == "google-fixture" and activity["time"] == "9:30 AM"
    assert saved["user_preferences"] == result["user_preferences"] and saved["export_notice"]
    assert result == before


def test_demo_and_independent_content_can_still_be_exported():
    result = plan()
    activity = result["days"][0]["activities"][0]
    activity.update(source="demo", place_id="demo:fixture", activity="Fictional Museum")
    assert persistable_itinerary(result) == result


def test_rehydration_uses_current_response_and_never_mutates_saved_draft():
    saved = persistable_itinerary(plan())
    before = deepcopy(saved)
    current = {"attractions": [place(name="Updated Museum", address="Fresh Address")]}
    displayed = hydrate_itinerary(saved, current)
    activity = displayed["days"][0]["activities"][0]
    assert activity["activity"] == "Updated Museum" and activity["address"] == "Fresh Address"
    assert activity["time_zone"] == "Europe/Paris" and activity["regular_opening_hours"]
    assert saved == before


def test_missing_current_place_details_do_not_reuse_old_content():
    saved = persistable_itinerary(plan())
    displayed = hydrate_itinerary(saved, None)
    activity = displayed["days"][0]["activities"][0]
    assert "reload" in activity["activity"] and "address" not in activity
    assert "query_place_id=google-fixture" in google_maps_link(activity)


def test_google_catalog_uses_opaque_labels_not_provider_fields():
    ranked, _ = rank_places([], [place()], TripPreferences(duration=1), [])
    catalog, lookup = model_place_catalog(ranked)
    assert catalog == [{"name": "Candidate 1", "type": "attraction", "community_mentions": []}]
    assert lookup["Candidate 1"]["name"] == "Google Fixture Museum"
    assert "google-fixture" not in json.dumps(catalog)


def test_candidate_alias_does_not_collide_with_independent_place_name():
    ranked, _ = rank_places([], [place(), {"name": "Candidate 1"}], TripPreferences(duration=1), [])
    catalog, lookup = model_place_catalog(ranked)
    assert len(lookup) == len(catalog) == 2


def test_ai_request_does_not_receive_google_content_and_alias_is_resolved_for_display():
    draft = {
        "days": [
            {
                "day": 1,
                "title": "Day 1",
                "activities": [
                    {
                        "time": "9:30 AM",
                        "activity": "Candidate 1",
                        "description": "Suggested visit.",
                        "type": "attraction",
                        "source_urls": [],
                    }
                ],
            }
        ],
        "reddit_tips": [],
        "budget_notes": [],
    }
    planner = AITripPlanner(use_ai=False)
    create = Mock(return_value=SimpleNamespace(output_text=json.dumps(draft)))
    planner.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    result = planner.generate_personalized_itinerary("Paris", [], {"duration": 1}, attractions=[place()])
    assert result["ai_generated"] is True
    request = json.dumps(create.call_args.kwargs)
    assert "Candidate 1" in request
    assert "Google Fixture Museum" not in request and "Provider-only Address" not in request
    assert "coordinates" not in request and "regular_opening_hours" not in request
    assert result["days"][0]["activities"][0]["activity"] == "Google Fixture Museum"


def test_attribution_is_present_only_for_google_and_escapes_third_party_html():
    html = attribution_html(
        place(attributions=[{"provider": "<script>bad</script>", "provider_uri": "javascript:bad"}])
    )
    assert 'translate="no"' in html and ">Google Maps</span>" in html
    assert "color: #1F1F1F" in html and "background: #fff" in html
    assert "<script>" not in html and "&lt;script&gt;" in html and "javascript:" not in html
    assert attribution_html({"source": "demo"}) == ""


def test_maps_link_uses_safe_url_or_encoded_id():
    assert (
        google_maps_link(place(google_maps_uri="https://maps.google.com/fixture"))
        == "https://maps.google.com/fixture"
    )
    assert "query_place_id=google-fixture" in google_maps_link(place(google_maps_uri="javascript:bad"))
    assert google_maps_link({"source": "demo", "place_id": "demo:id"}) is None


def test_only_fetch_timestamp_change_does_not_mark_previous_draft_stale():
    first = {"attractions": [place(fetched_at="first")]}
    second = {"attractions": [place(fetched_at="second")]}
    assert request_fingerprint("Paris", {}, [], first) == request_fingerprint("Paris", {}, [], second)
    second["attractions"][0]["name"] = "Changed Museum"
    assert request_fingerprint("Paris", {}, [], first) != request_fingerprint("Paris", {}, [], second)
