"""Keep Google content out of saved drafts/exports and model input.

Place IDs are the only Google fields retained. Display details are rehydrated
from a fresh response. This is a conservative technical guard, not legal advice.
"""

from copy import deepcopy
from urllib.parse import urlencode

from ..models.local_planner import activity_details, budget_notes
from ..models.place_ranking import rank_places
from ..models.trip_preferences import TripPreferences
from ..utils.content import safe_html, safe_url

PERSISTED_FIELDS = {
    "location",
    "user_preferences",
    "days",
    "reddit_tips",
    "budget_notes",
    "generated_at",
    "ai_generated",
    "ai_provider",
    "ai_model",
    "limitations",
    "retrieved_posts",
    "retrieval_method",
    "demo_mode",
    "generation_warning",
    "export_notice",
}


def model_place_catalog(places: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    """Google candidates get opaque aliases, not names/ratings/addresses/hours."""
    catalog, lookup = [], {}
    used_names = {place["name"] for place in places if place.get("source") != "google_places"}
    for index, place in enumerate(places, 1):
        alias = place["name"]
        if place.get("source") == "google_places":
            alias = f"Candidate {index}"
            while alias in used_names:
                alias += "_"
            catalog.append(
                {
                    "name": alias,
                    "type": place["type"],
                    "community_mentions": place["community_mentions"],
                }
            )
        else:
            catalog.append(place)
        lookup[alias] = place
        used_names.add(alias)
    return catalog, lookup


def persistable_itinerary(result: dict) -> dict:
    """Never export or session-cache Google names, details, or derived prose."""
    saved = deepcopy(result)
    google = False
    for day in saved["days"]:
        for index, activity in enumerate(day["activities"]):
            if activity.get("source") != "google_places":
                continue
            google = True
            day["activities"][index] = {
                "source": "google_places",
                "place_id": activity["place_id"],
                "time": activity["time"],
                "type": activity["type"],
                "activity": "Google Maps place (reload current details)",
                "description": "Place details are available only from a fresh Google Maps response.",
                "source_urls": activity.get("source_urls", []),
            }
    if google:
        # An accidental future provider-payload attachment must not bypass the guard.
        saved = {key: value for key, value in saved.items() if key in PERSISTED_FIELDS}
        for day in saved["days"]:
            day["title"] = f"Day {day['day']} in {saved['location']}"
        saved["reddit_tips"] = []
        saved["budget_notes"] = budget_notes(
            saved["location"], TripPreferences.from_dict(saved["user_preferences"]), 0
        )
        saved["export_notice"] = "Google Maps content omitted; place IDs retained for live lookup."
    return saved


def hydrate_itinerary(saved: dict, current_places: dict | None) -> dict:
    """Join ID references to this request's fresh provider response, never save it."""
    result = deepcopy(saved)
    preferences = TripPreferences.from_dict(result["user_preferences"])
    ranked, _ = rank_places(
        (current_places or {}).get("restaurants", []),
        (current_places or {}).get("attractions", []),
        preferences,
        result.get("retrieved_posts", []),
        include_demo=result.get("demo_mode", False),
    )
    by_id = {place["place_id"]: place for place in ranked if place.get("source") == "google_places"}
    for day in result["days"]:
        for activity in day["activities"]:
            if activity.get("source") != "google_places":
                continue
            place = by_id.get(activity["place_id"])
            if place and place["type"] == activity["type"]:
                activity.update(activity_details(place))
                activity["activity"] = place["name"]
                activity["description"] = "Suggested visit; confirm opening hours and travel times."
    return result


def attribution_html(place: dict) -> str:
    """Compact card attribution; all provider-controlled text/links are escaped."""
    if place.get("source") != "google_places":
        return ""
    providers = []
    for attribution in place.get("attributions", []):
        name = safe_html(attribution.get("provider", ""))
        uri = safe_url(attribution.get("provider_uri"))
        providers.append(
            f'<a href="{uri}" style="color: #1F1F1F; text-decoration: underline" target="_blank" rel="noopener noreferrer">{name}</a>'
            if uri != "#"
            else name
        )
    suffix = " · " + " · ".join(providers) if providers else ""
    return (
        '<div style="font-family: sans-serif; font-size: 12px; font-weight: 400; color: #1F1F1F; '
        'background: #fff; padding: 4px 6px; display: inline-block; margin-top: 10px; letter-spacing: normal">'
        f'<span translate="no" style="white-space: nowrap">Google Maps</span>{suffix}</div>'
    )


def google_maps_link(place: dict) -> str | None:
    uri = place.get("google_maps_uri")
    if safe_url(uri) != "#":
        return uri
    place_id = place.get("place_id")
    if place.get("source") == "google_places" and isinstance(place_id, str) and place_id:
        return "https://www.google.com/maps/search/?" + urlencode(
            {
                "api": "1",
                "query": "Place",
                "query_place_id": place_id,
            }
        )
    return None
