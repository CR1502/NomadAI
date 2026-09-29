import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from src.models.ai_trip_planner import AITripPlanner, ITINERARY_SCHEMA
from src.services.ollama import DEFAULT_MODEL, OllamaClient, OllamaError, local_base_url
from src.services.trip_planning import TripPlanningService


def payload():
    return {
        "days": [
            {
                "day": 1,
                "title": "Day 1",
                "activities": [
                    {
                        "time": "9:30 AM",
                        "activity": "Test Museum",
                        "description": "A suggested visit.",
                        "type": "attraction",
                        "source_urls": [],
                    }
                ],
            }
        ],
        "reddit_tips": [],
        "budget_notes": [],
    }


def make_client(data=None, status=200, error=None):
    response = SimpleNamespace(status_code=status, json=Mock(return_value=data))
    post = Mock(side_effect=error, return_value=response)
    return OllamaClient(session=SimpleNamespace(post=post)), post


def generate(client):
    return client.generate(model=DEFAULT_MODEL, instructions="System", prompt="User", schema=ITINERARY_SCHEMA)


def test_ollama_posts_local_schema_constrained_request_without_api_key_or_proxy():
    client, post = make_client({"message": {"content": json.dumps(payload())}, "done": True})
    assert json.loads(generate(client)) == payload()
    request = post.call_args
    assert request.args == ("http://127.0.0.1:11434/api/chat",)
    assert request.kwargs["json"]["model"] == "gemma4:12b"
    assert request.kwargs["json"]["format"] == ITINERARY_SCHEMA
    assert request.kwargs["json"]["stream"] is False
    assert request.kwargs["json"]["think"] is False
    assert request.kwargs["json"]["options"]["temperature"] == 0
    assert request.kwargs["allow_redirects"] is False
    assert request.kwargs["timeout"] == (3.05, 120)
    assert "headers" not in request.kwargs
    assert client.session.trust_env is False


@pytest.mark.parametrize(
    "url",
    [
        "https://ollama.com",
        "http://192.168.1.10:11434",
        "http://localhost.evil.test:11434",
        "http://user:secret@localhost:11434",
        "http://localhost:11434/api",
        "http://localhost/?key=secret",
        "file:///tmp/model",
        "http://127.0.0.1:bad",
        "http://127.0.0.1:70000",
        "http://127.0.0.1:0",
        None,
    ],
)
def test_remote_or_invalid_endpoints_are_rejected(url):
    with pytest.raises(ValueError, match="loopback"):
        local_base_url(url)


@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://localhost:11434/", "http://127.0.0.1:11434"),
        ("http://[::1]:11434", "http://[::1]:11434"),
        ("https://127.0.0.1", "https://127.0.0.1"),
    ],
)
def test_local_endpoints_are_normalized(url, expected):
    assert local_base_url(url) == expected


@pytest.mark.parametrize("model", ["", " ", None, "gemma:cloud", "gemma-cloud", "gemma\n"])
def test_invalid_or_cloud_model_is_rejected_before_request(model):
    client, post = make_client()
    with pytest.raises(ValueError):
        client.generate(model=model, instructions="", prompt="", schema={})
    post.assert_not_called()


@pytest.mark.parametrize(
    "failure, message",
    [
        (requests.Timeout(), "timed out"),
        (requests.ConnectionError(), "Cannot reach"),
    ],
)
def test_network_failure_is_safe_and_not_retried(failure, message):
    client, post = make_client(error=failure)
    with pytest.raises(OllamaError, match=message):
        generate(client)
    assert post.call_count == 1


@pytest.mark.parametrize(
    "status, message", [(404, "unavailable"), (500, "could not generate"), (302, "could not generate")]
)
def test_api_errors_do_not_expose_model_response(status, message):
    client, _ = make_client({"error": "private user data"}, status=status)
    with pytest.raises(OllamaError, match=message) as error:
        generate(client)
    assert "private user data" not in str(error.value)


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        {},
        {"done": False, "message": {"content": "{}"}},
        {"done": True, "message": {"content": ""}},
        {"done": True, "message": []},
    ],
)
def test_incomplete_response_is_rejected(data):
    client, _ = make_client(data)
    with pytest.raises(OllamaError, match="incomplete"):
        generate(client)


def test_truncated_output_is_rejected():
    client, _ = make_client({"done": True, "done_reason": "length", "message": {"content": "{}"}})
    with pytest.raises(OllamaError, match="output limit"):
        generate(client)


def test_invalid_json_envelope_is_rejected():
    client, post = make_client()
    post.return_value.json.side_effect = ValueError()
    with pytest.raises(OllamaError, match="invalid JSON"):
        generate(client)


def test_local_model_uses_same_full_itinerary_validation():
    planner = AITripPlanner(provider="ollama", use_ai=False)
    planner.client, post = make_client({"message": {"content": json.dumps(payload())}, "done": True})
    result = planner.generate_personalized_itinerary(
        "Paris", [], {"duration": 1}, attractions=[{"name": "Test Museum"}]
    )
    assert result["ai_generated"] is True
    assert result["ai_provider"] == "ollama" and result["ai_model"] == DEFAULT_MODEL
    assert post.call_count == 1


@pytest.mark.parametrize("failure", ["invented", "repeat", "duration", "source", "schema"])
def test_invalid_local_model_draft_falls_back(failure):
    draft = payload()
    activity = draft["days"][0]["activities"][0]
    if failure == "invented":
        activity["activity"] = "Invented Place"
    elif failure == "repeat":
        draft["days"][0]["activities"] *= 2
    elif failure == "duration":
        draft["days"] = []
    elif failure == "source":
        activity["source_urls"] = ["https://invented.example.com"]
    else:
        draft.pop("budget_notes")
    planner = AITripPlanner(provider="ollama", use_ai=False)
    planner.client, _ = make_client({"message": {"content": json.dumps(draft)}, "done": True})
    result = planner.generate_personalized_itinerary(
        "Paris", [], {"duration": 1}, attractions=[{"name": "Test Museum"}]
    )
    assert result["ai_generated"] is False and result["generation_warning"]
    assert result["days"][0]["activities"][0]["activity"] == "Test Museum"


def test_ollama_failure_does_not_initialize_or_fall_back_to_openai(monkeypatch):
    setup = Mock(side_effect=AssertionError("No cloud fallback"))
    monkeypatch.setattr(AITripPlanner, "setup_openai", setup)
    generate = Mock(side_effect=OllamaError("Cannot reach local Ollama."))
    monkeypatch.setattr(OllamaClient, "generate", generate)
    service = TripPlanningService(provider="ollama", api_key="unused-cloud-key")
    result = service.generate(
        "Paris", {"duration": 1}, [], {"attractions": [{"name": "Test Museum"}]}, use_ai=True
    )
    assert result["ai_generated"] is False and "Cannot reach" in result["generation_warning"]
    setup.assert_not_called()


def test_local_ai_is_not_called_without_opt_in_or_in_demo(monkeypatch):
    generate = Mock(side_effect=AssertionError("No automatic inference"))
    monkeypatch.setattr(OllamaClient, "generate", generate)
    service = TripPlanningService(provider="ollama")
    places = {"attractions": [{"name": "Test Museum"}]}
    service.generate("Paris", {"duration": 1}, [], places)
    service.generate("Paris", {"duration": 1}, [], places, use_ai=True, demo_mode=True)
    generate.assert_not_called()


def test_invalid_provider_is_rejected():
    with pytest.raises(ValueError, match="provider"):
        AITripPlanner(provider="unknown")
