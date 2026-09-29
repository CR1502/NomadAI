from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from src.services.places import BASIC_FIELDS, SEARCH_URL, PlacesClient, PlacesError, normalize_google_place


def response(status=200, data=None):
    return SimpleNamespace(status_code=status, json=Mock(return_value={} if data is None else data))


def raw_place(**changes):
    return {
        "id": "test-museum",
        "displayName": {"text": "Test Museum"},
        "formattedAddress": "Fixture Address",
        "priceLevel": "PRICE_LEVEL_MODERATE",
        "rating": 4.4,
        "userRatingCount": 25,
        "types": ["museum", "tourist_attraction"],
        "location": {"latitude": 48.85, "longitude": 2.35},
        "timeZone": {"id": "Europe/Paris"},
        "utcOffsetMinutes": 120,
        "businessStatus": "OPERATIONAL",
        "websiteUri": "https://example.org",
        "nationalPhoneNumber": "Fixture Phone",
        "googleMapsUri": "https://maps.google.com/fixture",
        "regularOpeningHours": {"weekdayDescriptions": ["Monday: 10:00–18:00"], "periods": []},
        "attributions": [{"provider": "Fixture Provider", "providerUri": "https://example.org/provider"}],
        **changes,
    }


def make_client(*responses, **kwargs):
    post = Mock(side_effect=list(responses))
    return PlacesClient("secret-key", session=SimpleNamespace(post=post), sleep=Mock(), **kwargs), post


def test_new_search_uses_explicit_fields_and_two_requests_not_per_place_details():
    client, post = make_client(
        response(data={"places": [raw_place()]}), response(data={"places": [raw_place()]})
    )
    result = client.search_destination("Paris")
    assert len(result["restaurants"]) == len(result["attractions"]) == 1
    assert post.call_count == 2
    for call in post.call_args_list:
        assert call.args == (SEARCH_URL,)
        request = call.kwargs
        assert request["json"]["pageSize"] == 10
        assert "Paris" in request["json"]["textQuery"]
        assert request["json"]["strictTypeFiltering"] is True
        assert request["headers"]["X-Goog-Api-Key"] == "secret-key"
        assert "*" not in request["headers"]["X-Goog-FieldMask"]
        assert "places.attributions" in request["headers"]["X-Goog-FieldMask"]
        assert "places.regularOpeningHours" in request["headers"]["X-Goog-FieldMask"]
        assert "places.reviews" not in request["headers"]["X-Goog-FieldMask"]
        assert request["allow_redirects"] is False
        assert request["timeout"] == (3.05, 15)


def test_basic_mask_excludes_enterprise_fields_and_unused_category_is_not_requested():
    client, post = make_client(response(), detail_level="basic")
    client.search_destination("Paris", attractions=False)
    assert post.call_count == 1
    assert client.field_mask == ",".join(f"places.{field}" for field in BASIC_FIELDS)
    assert "rating" not in client.field_mask and "priceLevel" not in client.field_mask


def test_normalization_retains_coordinates_hours_zone_links_and_attribution():
    place = normalize_google_place(raw_place())
    assert place["source"] == "google_places"
    assert place["name"] == "Test Museum" and place["address"] == "Fixture Address"
    assert place["price_level"] == 2 and place["price"] == "$$"
    assert place["rating"] == 4.4 and place["phone"] == "Fixture Phone"
    assert place["coordinates"] == {"latitude": 48.85, "longitude": 2.35}
    assert place["time_zone"] == "Europe/Paris" and place["utc_offset_minutes"] == 120
    assert place["regular_opening_hours"]["weekdayDescriptions"]
    assert place["current_opening_hours"] is None
    assert place["attributions"] == [
        {"provider": "Fixture Provider", "provider_uri": "https://example.org/provider"}
    ]


@pytest.mark.parametrize(
    "price, expected",
    [
        ("PRICE_LEVEL_FREE", 0),
        ("PRICE_LEVEL_INEXPENSIVE", 1),
        ("PRICE_LEVEL_MODERATE", 2),
        ("PRICE_LEVEL_EXPENSIVE", 3),
        ("PRICE_LEVEL_VERY_EXPENSIVE", 4),
        ("PRICE_LEVEL_UNSPECIFIED", None),
        (None, None),
        ([], None),
    ],
)
def test_price_enums_and_missing_prices(price, expected):
    assert normalize_google_place(raw_place(priceLevel=price))["price_level"] == expected


def test_malformed_optional_metadata_remains_unknown():
    place = normalize_google_place(
        raw_place(
            rating=True,
            userRatingCount=-1,
            location={"latitude": 999, "longitude": 2},
            timeZone=[],
            utcOffsetMinutes=True,
            regularOpeningHours=[],
            currentOpeningHours="closed",
            attributions=[None, {"provider": 42}],
            websiteUri=[],
            formattedAddress=42,
        )
    )
    assert place["rating"] is None and place["coordinates"] is None
    assert place["time_zone"] is None and place["utc_offset_minutes"] is None
    assert place["regular_opening_hours"] is None and place["current_opening_hours"] is None
    assert place["attributions"] == [] and place["website"] == "Not available"
    assert place["address"] == "Address not available"


@pytest.mark.parametrize(
    "raw", [None, [], {}, {"id": "test"}, raw_place(id=""), raw_place(displayName={"text": 4})]
)
def test_malformed_required_place_fields_are_skipped(raw):
    assert normalize_google_place(raw) is None


def test_invalid_duplicate_and_closed_places_are_skipped_and_results_are_bounded():
    client, post = make_client(
        response(
            data={
                "places": [
                    None,
                    raw_place(),
                    raw_place(),
                    raw_place(id="closed", businessStatus="CLOSED_PERMANENTLY"),
                    raw_place(id="temporary", businessStatus="CLOSED_TEMPORARILY"),
                    raw_place(id="other"),
                ]
            }
        )
    )
    result = client.search("Paris", "tourist_attraction", limit=5)
    assert [place["place_id"] for place in result] == ["test-museum"]


@pytest.mark.parametrize(
    "failure", [response(429), response(503), requests.Timeout(), requests.ConnectionError()]
)
def test_transient_errors_retry_once_then_succeed(failure):
    client, post = make_client(failure, response(data={"places": [raw_place()]}))
    assert client.search("Paris", "restaurant")
    assert post.call_count == 2 and client.sleep.call_count == 1


def test_repeated_rate_limits_are_bounded_and_safe():
    client, post = make_client(response(429), response(429))
    with pytest.raises(PlacesError, match="quota") as error:
        client.search("Paris", "restaurant")
    assert post.call_count == 2 and "secret-key" not in str(error.value)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 302])
def test_non_transient_errors_are_not_retried_and_do_not_expose_provider_body(status):
    client, post = make_client(response(status, {"error": {"message": "secret-key"}}))
    with pytest.raises(PlacesError) as error:
        client.search("Paris", "restaurant")
    assert post.call_count == 1 and "secret-key" not in str(error.value)


@pytest.mark.parametrize("data", [[], None, {"places": {}}, {"error": {"message": "secret-key"}}])
def test_malformed_response_is_rejected(data):
    client, _ = make_client(SimpleNamespace(status_code=200, json=lambda: data))
    with pytest.raises(PlacesError, match="unexpected"):
        client.search("Paris", "restaurant")


def test_invalid_json_is_rejected():
    client, _ = make_client(SimpleNamespace(status_code=200, json=Mock(side_effect=ValueError())))
    with pytest.raises(PlacesError, match="invalid JSON"):
        client.search("Paris", "restaurant")


def test_partial_category_failure_preserves_other_results():
    client, _ = make_client(response(403), response(data={"places": [raw_place()]}))
    result = client.search_destination("Paris")
    assert not result["restaurants"] and result["attractions"]
    assert len(result["warnings"]) == 1 and "access denied" in result["warnings"][0]


@pytest.mark.parametrize("key", ["", " ", None])
def test_missing_credentials_fail_before_network(key):
    with pytest.raises(PlacesError, match="not configured"):
        PlacesClient(key)


@pytest.mark.parametrize("kwargs", [{"detail_level": "all"}, {"retries": True}, {"retries": 3}])
def test_invalid_client_configuration(kwargs):
    with pytest.raises(ValueError):
        PlacesClient("secret-key", **kwargs)


@pytest.mark.parametrize(
    "destination, category, limit",
    [
        ("", "restaurant", 10),
        (None, "restaurant", 10),
        ("Paris", "hotel", 10),
        ("Paris", "restaurant", 0),
        ("Paris", "restaurant", 21),
        ("Paris", "restaurant", True),
    ],
)
def test_invalid_search_arguments_fail_before_network(destination, category, limit):
    client, post = make_client()
    with pytest.raises(ValueError):
        client.search(destination, category, limit=limit)
    post.assert_not_called()
