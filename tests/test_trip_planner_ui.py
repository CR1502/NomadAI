import json
from types import SimpleNamespace
from unittest.mock import Mock

from streamlit.testing.v1 import AppTest

from src.services.trip_planning import TripPlanningService
from src.services.places import PlacesClient
from src.services.ollama import OllamaClient
from src.utils.helpers import PROJECT_ROOT
from src.web import trip_planner


def run_app():
    return AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=20).run()


def submit(app):
    next(button for button in app.button if button.label == "Generate itinerary").click().run()
    assert not app.exception
    return app


def current_draft(app, location="Paris", demo=False):
    return app.session_state["trip_drafts"][(location, demo)]["result"]


def spy_on_generation(monkeypatch):
    original = TripPlanningService.generate
    calls = []

    def generate(self, *args, **kwargs):
        calls.append({"api_key": self.api_key, "args": args, "kwargs": kwargs})
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TripPlanningService, "generate", generate)
    return calls


def test_planning_runs_only_on_submit_and_survives_unrelated_reruns(monkeypatch):
    calls = spy_on_generation(monkeypatch)
    app = run_app()
    assert not app.exception
    assert not calls
    app.slider(key="trip_duration").set_value(2)
    app.selectbox(key="trip_pace").set_value("relaxed")
    app.multiselect(key="trip_interests").set_value(["nature"])
    submit(app)
    assert len(calls) == 1
    result = current_draft(app)
    assert result["user_preferences"]["duration"] == 2
    assert result["user_preferences"]["interests"] == ["nature"]
    assert result["user_preferences"]["pace"] == "relaxed"
    assert result["ai_generated"] is False
    app.checkbox(key="show_costs").uncheck().run()
    assert not app.exception and len(calls) == 1
    assert current_draft(app) == result


def test_previous_draft_is_marked_stale_when_preferences_change():
    app = submit(run_app())
    original = current_draft(app)
    app.slider(key="trip_duration").set_value(5).run()
    assert not app.exception
    assert any("previous draft" in warning.value for warning in app.warning)
    assert current_draft(app) == original


def test_drafts_are_scoped_to_destination_and_demo_mode():
    app = submit(run_app())
    app.selectbox(key="destination").select("Tokyo").run()
    assert not app.exception
    assert not any("Paris —" in heading.value for heading in app.subheader)
    app.selectbox(key="destination").select("Paris").run()
    assert any("Paris — 3-day draft" in heading.value for heading in app.subheader)
    app.checkbox(key="demo_mode").check().run()
    assert not any("Paris —" in heading.value for heading in app.subheader)
    submit(app)
    assert current_draft(app, demo=True)["demo_mode"] is True


def test_demo_creates_unique_fictional_places_without_api_or_download(monkeypatch):
    encoder = Mock(side_effect=AssertionError("No model should be downloaded"))
    monkeypatch.setattr(trip_planner, "get_semantic_encoder", encoder)
    app = run_app()
    app.checkbox(key="demo_mode").check().run()
    assert app.checkbox(key="trip_use_ai").disabled
    assert not app.checkbox(key="trip_use_ai").value
    submit(app)
    result = current_draft(app, demo=True)
    activities = [activity for day in result["days"] for activity in day["activities"]]
    assert len(activities) == len({activity["place_id"] for activity in activities}) == 9
    assert all(
        activity["place_id"].startswith("demo:") and not activity["source_urls"] for activity in activities
    )
    assert any("fictional demo places" in warning.value for warning in app.warning)
    encoder.assert_not_called()


def test_price_and_budget_preferences_reach_the_local_planner():
    app = run_app()
    app.checkbox(key="demo_mode").check().run()
    app.selectbox(key="trip_price_limit").set_value(0)
    app.number_input(key="trip_budget_Paris").set_value(20.0)
    submit(app)
    result = current_draft(app, demo=True)
    activities = [activity for day in result["days"] for activity in day["activities"]]
    assert len(activities) == 2
    assert all(activity["price_level"] == 0 for activity in activities)
    assert "$20 USD" in " ".join(result["budget_notes"])
    assert any("unfilled" in limitation for limitation in result["limitations"])


def test_planner_retrieves_local_posts_when_insights_are_hidden(sample_posts, tmp_path):
    path = tmp_path / "data" / "by_location" / "paris" / "travel" / "reddit_posts.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(sample_posts))
    app = run_app()
    app.checkbox(key="show_reddit").uncheck().run()
    submit(app)
    assert len(current_draft(app)["retrieved_posts"]) == 2


def test_ai_is_opt_in_and_streamlit_secret_is_passed_to_the_service(monkeypatch):
    calls = spy_on_generation(monkeypatch)
    monkeypatch.setattr(
        trip_planner, "find_spec", lambda name: SimpleNamespace() if name == "openai" else None
    )
    app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=20)
    app.secrets["OPENAI_API_KEY"] = "test-secret-not-for-network"
    app.run()
    assert not app.exception
    assert not calls
    assert not app.checkbox(key="trip_use_ai").disabled
    assert app.checkbox(key="trip_use_ai").value is False
    app.selectbox(key="trip_ai_provider").select("openai")
    app.checkbox(key="trip_use_ai").check()
    submit(app)
    assert calls[0]["kwargs"]["use_ai"] is True
    assert calls[0]["api_key"] == "test-secret-not-for-network"
    assert "test-secret-not-for-network" not in json.dumps(current_draft(app))


def test_failed_semantic_model_load_falls_back_to_tfidf(monkeypatch):
    monkeypatch.setattr(
        trip_planner, "find_spec", lambda name: SimpleNamespace() if name == "sentence_transformers" else None
    )
    load = Mock(side_effect=RuntimeError("Offline"))
    monkeypatch.setattr(trip_planner, "get_semantic_encoder", load)
    app = run_app()
    assert not load.called
    app.checkbox(key="trip_semantic").check()
    submit(app)
    assert load.call_count == 1
    assert current_draft(app)["retrieval_method"] == "tfidf"
    assert any("Semantic retrieval is unavailable" in warning.value for warning in app.warning)


def test_fresh_source_data_marks_the_existing_draft_stale(sample_posts, tmp_path):
    app = submit(run_app())
    path = tmp_path / "data" / "by_location" / "paris" / "travel" / "reddit_posts.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(sample_posts))
    import streamlit as st

    st.cache_data.clear()
    app.run()
    assert not app.exception
    assert any("source data changed" in warning.value for warning in app.warning)


def test_untrusted_place_names_are_rendered_as_text_not_raw_html(monkeypatch):
    malicious_name = "<script>alert(1)</script> Museum"
    monkeypatch.setattr(
        PlacesClient,
        "search_destination",
        lambda *args, **kwargs: {
            "restaurants": [],
            "attractions": [
                {
                    "name": malicious_name,
                    "place_id": "museum",
                    "source": "google_places",
                }
            ],
            "warnings": [],
        },
    )
    app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=20)
    app.secrets["GOOGLE_PLACES_API_KEY"] = "test"
    app.run()
    submit(app)
    assert any(malicious_name in item.value for item in app.text)
    assert not any("<script>" in item.value for item in app.markdown)
    assert malicious_name not in json.dumps(current_draft(app))


def live_places():
    return {
        "restaurants": [],
        "attractions": [
            {
                "name": "Provider Museum",
                "place_id": "provider-museum",
                "source": "google_places",
                "address": "Provider-only Address",
                "price_level": 1,
                "rating": None,
                "types": ["museum"],
                "attributions": [{"provider": "Third Party", "provider_uri": "https://example.org"}],
            }
        ],
        "warnings": [],
    }


def test_ollama_is_default_opt_in_and_results_are_redacted_in_session(monkeypatch):
    places = Mock(side_effect=lambda *args, **kwargs: live_places())
    monkeypatch.setattr(PlacesClient, "search_destination", places)
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
    generate = Mock(return_value=json.dumps(draft))
    monkeypatch.setattr(OllamaClient, "generate", generate)
    app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=20)
    app.secrets["GOOGLE_PLACES_API_KEY"] = "fixture-key"
    app.run()
    assert app.selectbox(key="trip_ai_provider").value == "ollama"
    assert app.checkbox(key="trip_use_ai").value is False
    generate.assert_not_called()
    app.slider(key="trip_duration").set_value(1)
    app.checkbox(key="trip_use_ai").check()
    submit(app)
    assert generate.call_count == 1
    assert generate.call_args.kwargs["model"] == "gemma4:12b"
    assert "Provider-only Address" not in generate.call_args.kwargs["prompt"]
    assert "Provider Museum" not in generate.call_args.kwargs["prompt"]
    saved = current_draft(app)
    assert saved["ai_generated"] is True and saved["ai_provider"] == "ollama"
    assert "Provider Museum" not in json.dumps(saved) and "Provider-only Address" not in json.dumps(saved)
    assert any("Provider Museum" in item.value for item in app.text)
    app.checkbox(key="show_costs").uncheck().run()
    assert not app.exception and generate.call_count == 1
    assert places.call_count == 3  # fresh page response, generation submit, unrelated rerun
    assert any("Provider Museum" in item.value for item in app.text)
    assert current_draft(app) == saved
    assert app.get("download_button")[0].proto.ignore_rerun is True


def test_backend_switch_updates_availability_without_running_generation(monkeypatch):
    generate = Mock(side_effect=AssertionError("No automatic inference"))
    monkeypatch.setattr(OllamaClient, "generate", generate)
    app = run_app()
    assert not app.checkbox(key="trip_use_ai").disabled
    app.selectbox(key="trip_ai_provider").select("openai").run()
    assert app.checkbox(key="trip_use_ai").disabled
    app.selectbox(key="trip_ai_provider").select("ollama").run()
    assert not app.checkbox(key="trip_use_ai").disabled
    generate.assert_not_called()


def test_old_unversioned_session_drafts_are_not_rendered_or_exported():
    app = run_app()
    app.session_state["trip_drafts"] = {
        ("Paris", False): {
            "fingerprint": "legacy",
            "result": {"days": [{"activities": [{"activity": "Old Provider Content"}]}]},
        },
    }
    app.run()
    assert not app.exception
    assert not app.session_state["trip_drafts"]
    assert not app.get("download_button")


def test_live_provider_errors_warn_without_crashing_the_app(monkeypatch):
    monkeypatch.setattr(
        PlacesClient,
        "search_destination",
        lambda *args, **kwargs: {
            "restaurants": [],
            "attractions": [],
            "warnings": ["Restaurants: Google Places quota reached."],
        },
    )
    app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=20)
    app.secrets["GOOGLE_PLACES_API_KEY"] = "fixture-key"
    app.run()
    assert not app.exception
    assert any("quota" in warning.value for warning in app.warning)


def test_public_policy_links_are_required_for_live_sources_and_unsafe_links_are_rejected(monkeypatch):
    monkeypatch.setattr(PlacesClient, "search_destination", lambda *args, **kwargs: live_places())
    app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=20)
    app.secrets["GOOGLE_PLACES_API_KEY"] = "fixture-key"
    app.secrets["TERMS_OF_USE_URL"] = "javascript:bad"
    app.secrets["PRIVACY_POLICY_URL"] = "https://example.org/privacy"
    app.run()
    assert any("Before public deployment" in warning.value for warning in app.warning)
    app.secrets["TERMS_OF_USE_URL"] = "https://example.org/terms"
    app.run()
    assert not app.exception
    assert not any("Before public deployment" in warning.value for warning in app.warning)
    links = app.get("link_button")
    assert any(link.label == "Terms of Use" for link in links)
    assert any(link.label == "Privacy Policy" for link in links)
