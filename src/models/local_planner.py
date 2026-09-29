"""Deterministic itinerary drafts with explicit, limited constraints."""

from datetime import datetime, timezone

from ..utils.content import calculate_costs
from ..utils.helpers import load_config
from .trip_preferences import TripPreferences

VISIT_TIMES = {
    "relaxed": ("9:30 AM", "1:00 PM"),
    "balanced": ("9:30 AM", "12:30 PM", "3:30 PM"),
    "busy": ("9:00 AM", "11:30 AM", "2:00 PM", "5:00 PM"),
}


def budget_notes(location: str, preferences: TripPreferences, excluded: int) -> list[str]:
    notes = ["Actual costs, opening hours, and travel times must be checked for your travel dates."]
    if preferences.daily_budget is not None:
        notes.append(
            f"Daily budget target: ${preferences.daily_budget:g} USD; this is not a verified cost estimate."
        )
        components = load_config().get("costs", {}).get(location)
        if components and preferences.daily_budget < calculate_costs(components)["daily"]:
            notes.append(
                "Your target is below the illustrative destination budget; check accommodation and other costs."
            )
    if preferences.max_price_level is not None:
        notes.append(
            "Known venue price tiers above your maximum were excluded. Price tiers are not USD amounts."
        )
    if excluded:
        notes.append(f"{excluded} place(s) excluded because their known price tier exceeds your maximum.")
    return notes


def activity_details(place: dict) -> dict:
    return {
        "place_id": place.get("place_id"),
        "address": place.get("address"),
        "price": place["price"],
        "price_level": place["price_level"],
        "match_reasons": place["match_reasons"],
        "community_mentions": place["community_mentions"],
        "warnings": place["warnings"],
        "source": place.get("source"),
        "coordinates": place.get("coordinates"),
        "regular_opening_hours": place.get("regular_opening_hours"),
        "current_opening_hours": place.get("current_opening_hours"),
        "time_zone": place.get("time_zone"),
        "utc_offset_minutes": place.get("utc_offset_minutes"),
        "google_maps_uri": place.get("google_maps_uri"),
        "attributions": place.get("attributions", []),
    }


def build_local_itinerary(
    location: str,
    preferences: TripPreferences,
    places: list[dict],
    excluded: int = 0,
) -> dict:
    remaining = list(places)
    days = []
    foodie = "food" in preferences.interests
    order = (
        ("restaurant", "attraction", "restaurant", "attraction")
        if foodie
        else ("attraction", "restaurant", "attraction", "attraction")
    )
    for index in range(preferences.duration):
        activities = []
        for slot, visit_time in enumerate(VISIT_TIMES[preferences.pace]):
            if not remaining:
                break
            next_index = next((i for i, place in enumerate(remaining) if place["type"] == order[slot]), 0)
            place = remaining.pop(next_index)
            activities.append(
                {
                    "time": visit_time,
                    "activity": place["name"],
                    "type": place["type"],
                    "description": "Suggested from available place data; confirm hours and availability before visiting.",
                    "source_urls": [mention["url"] for mention in place["community_mentions"]],
                    **activity_details(place),
                }
            )
        days.append({"day": index + 1, "title": f"Day {index + 1} in {location}", "activities": activities})
    requested = preferences.duration * preferences.activities_per_day
    actual = sum(len(day["activities"]) for day in days)
    limitations = [
        "Draft only: opening hours, travel times, accessibility, availability, and actual costs are unverified."
    ]
    if actual < requested:
        limitations.append(
            f"Only {actual} unique place(s) are available for {requested} activity slots; remaining slots are unfilled."
        )
    return {
        "location": location,
        "user_preferences": preferences.to_dict(),
        "days": days,
        "reddit_tips": [],
        "budget_notes": budget_notes(location, preferences, excluded),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ai_generated": False,
        "limitations": limitations,
    }
