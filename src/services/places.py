"""Bounded, uncached Google Places API (New) searches, independent of Streamlit."""

import math
import time
from datetime import datetime, timezone

import requests

from ..utils.content import normalize_place

SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
BASIC_FIELDS = (
    "id",
    "displayName",
    "formattedAddress",
    "types",
    "primaryType",
    "location",
    "googleMapsUri",
    "attributions",
    "businessStatus",
    "timeZone",
    "utcOffsetMinutes",
)
STANDARD_FIELDS = BASIC_FIELDS + (
    "rating",
    "userRatingCount",
    "priceLevel",
    "websiteUri",
    "nationalPhoneNumber",
    "regularOpeningHours",
    "currentOpeningHours",
)
PRICE_LEVELS = {
    "PRICE_LEVEL_FREE": 0,
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}


class PlacesError(RuntimeError):
    """Safe provider errors: never expose response bodies, request headers, or keys."""


def normalize_google_place(raw: dict) -> dict | None:
    """Adapt New API fields; absent or malformed values remain unknown, not zero."""
    if not isinstance(raw, dict):
        return None
    display = raw.get("displayName")
    name = display.get("text") if isinstance(display, dict) else None
    place_id = raw.get("id")
    if not isinstance(name, str) or not name.strip() or not isinstance(place_id, str) or not place_id:
        return None
    legacy = {
        "place_id": place_id,
        "name": name,
        "formatted_address": raw.get("formattedAddress"),
        "rating": raw.get("rating"),
        "user_ratings_total": raw.get("userRatingCount"),
        "price_level": PRICE_LEVELS.get(raw.get("priceLevel"))
        if isinstance(raw.get("priceLevel"), str)
        else None,
        "website": raw.get("websiteUri"),
        "formatted_phone_number": raw.get("nationalPhoneNumber"),
        "types": raw.get("types"),
    }
    for field in ("formatted_address", "website", "formatted_phone_number"):
        if not isinstance(legacy[field], str) or not legacy[field].strip():
            legacy.pop(field)
    result = normalize_place(legacy, legacy)
    coordinates = raw.get("location")
    if not isinstance(coordinates, dict) or not all(
        isinstance(coordinates.get(key), (int, float))
        and not isinstance(coordinates[key], bool)
        and math.isfinite(coordinates[key])
        and abs(coordinates[key]) <= bound
        for key, bound in (("latitude", 90), ("longitude", 180))
    ):
        coordinates = None
    time_zone = raw.get("timeZone")
    time_zone = time_zone.get("id") if isinstance(time_zone, dict) else None
    offset = raw.get("utcOffsetMinutes")
    if isinstance(offset, bool) or not isinstance(offset, int) or not -840 <= offset <= 840:
        offset = None
    attributions = raw.get("attributions")
    attributions = [
        {"provider": entry["provider"], "provider_uri": entry.get("providerUri", "")}
        for entry in (attributions if isinstance(attributions, list) else [])
        if isinstance(entry, dict) and isinstance(entry.get("provider"), str) and entry["provider"]
    ]
    return {
        **result,
        "source": "google_places",
        "coordinates": coordinates,
        "primary_type": raw.get("primaryType"),
        "business_status": raw.get("businessStatus"),
        "google_maps_uri": raw.get("googleMapsUri"),
        "attributions": attributions,
        "time_zone": time_zone if isinstance(time_zone, str) else None,
        "utc_offset_minutes": offset,
        "regular_opening_hours": raw.get("regularOpeningHours")
        if isinstance(raw.get("regularOpeningHours"), dict)
        else None,
        "current_opening_hours": raw.get("currentOpeningHours")
        if isinstance(raw.get("currentOpeningHours"), dict)
        else None,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


class PlacesClient:
    """Two categorical searches, no geocoding, per-place details, or pagination."""

    def __init__(
        self,
        api_key: str,
        *,
        detail_level: str = "standard",
        session=None,
        retries: int = 1,
        sleep=time.sleep,
    ):
        if not isinstance(api_key, str) or not api_key.strip():
            raise PlacesError("Google Places is not configured. Set GOOGLE_PLACES_API_KEY.")
        if detail_level not in ("basic", "standard"):
            raise ValueError("GOOGLE_PLACES_DETAIL_LEVEL must be basic or standard")
        if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 2:
            raise ValueError("retries must be between 0 and 2")
        self._api_key = api_key
        self.detail_level = detail_level
        self.session = session or requests.Session()
        self.retries = retries
        self.sleep = sleep
        fields = BASIC_FIELDS if detail_level == "basic" else STANDARD_FIELDS
        self.field_mask = ",".join(f"places.{field}" for field in fields)

    def search(self, destination: str, category: str, *, limit: int = 10) -> list[dict]:
        if not isinstance(destination, str) or not destination.strip() or len(destination) > 200:
            raise ValueError("A destination of 1–200 characters is required")
        if category not in ("restaurant", "tourist_attraction"):
            raise ValueError("Unsupported place category")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")
        label = "restaurants" if category == "restaurant" else "tourist attractions"
        payload = {
            "textQuery": f"{label} in {destination.strip()}",
            "includedType": category,
            "strictTypeFiltering": True,
            "pageSize": limit,
            "languageCode": "en",
        }
        for attempt in range(self.retries + 1):
            try:
                response = self.session.post(
                    SEARCH_URL,
                    json=payload,
                    headers={"X-Goog-Api-Key": self._api_key, "X-Goog-FieldMask": self.field_mask},
                    timeout=(3.05, 15),
                    allow_redirects=False,
                )
            except (requests.Timeout, requests.ConnectionError):
                if attempt < self.retries:
                    self.sleep(0.25 * 2**attempt)
                    continue
                raise PlacesError("Google Places could not be reached. Try again later.") from None
            except requests.RequestException:
                raise PlacesError("Google Places request failed. Check the provider configuration.") from None
            if response.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                self.sleep(0.25 * 2**attempt)
                continue
            if response.status_code != 200:
                messages = {
                    400: "Google Places rejected the request. Check the configured field mask and API setup.",
                    401: "Google Places authentication failed. Check your API key.",
                    403: "Google Places access denied. Enable Places API (New), billing, and appropriate key restrictions.",
                    429: "Google Places quota or rate limit reached. Try again later.",
                }
                raise PlacesError(
                    messages.get(response.status_code, "Google Places is temporarily unavailable.")
                )
            try:
                data = response.json()
            except ValueError:
                raise PlacesError("Google Places returned invalid JSON.") from None
            if not isinstance(data, dict) or "error" in data or not isinstance(data.get("places", []), list):
                raise PlacesError("Google Places returned an unexpected response.")
            places, seen = [], set()
            for raw in data.get("places", [])[:limit]:
                place = normalize_google_place(raw)
                if not place or place["place_id"] in seen:
                    continue
                seen.add(place["place_id"])
                if place["business_status"] in ("CLOSED_PERMANENTLY", "CLOSED_TEMPORARILY"):
                    continue
                places.append(place)
            return places
        raise PlacesError("Google Places is unavailable.")

    def search_destination(
        self, destination: str, *, restaurants: bool = True, attractions: bool = True
    ) -> dict:
        result = {"restaurants": [], "attractions": [], "warnings": [], "source": "google_places"}
        for enabled, key, category in (
            (restaurants, "restaurants", "restaurant"),
            (attractions, "attractions", "tourist_attraction"),
        ):
            if enabled:
                try:
                    result[key] = self.search(destination, category)
                except PlacesError as error:
                    result["warnings"].append(f"{key.title()}: {error}")
        return result
