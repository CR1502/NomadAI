"""Explicit-action itinerary UI; widget reruns never trigger generation."""

import hashlib
import json
from importlib.util import find_spec

import streamlit as st

from ..data_pipeline.storage import is_demo_post, location_slug
from ..models.embedding_model import EmbeddingModel
from ..models.trip_preferences import INTERESTS, PACE_LIMITS, TripPreferences
from ..services.trip_planning import TripPlanningService
from ..services.ollama import DEFAULT_BASE_URL, DEFAULT_MODEL
from ..services.place_policy import (
    attribution_html,
    google_maps_link,
    hydrate_itinerary,
    persistable_itinerary,
)
from ..utils.content import calculate_costs, safe_url
from ..utils.helpers import load_config


@st.cache_resource
def get_semantic_encoder():
    """Cache only the encoder; each request owns its index and metadata."""
    return EmbeddingModel().model


def request_fingerprint(location: str, preferences: dict, posts: list, places: dict | None, **options) -> str:
    # Fetch timestamps change on every live request, not the underlying places.
    if places:
        places = {
            key: [
                {field: value for field, value in place.items() if field != "fetched_at"}
                for place in places.get(key, [])
            ]
            for key in ("restaurants", "attractions")
        }
    payload = json.dumps([location, preferences, posts, places, options], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def render_trip_planner(
    location: str,
    posts: list[dict],
    places: dict | None,
    *,
    demo_mode: bool,
    api_key: str = "",
    model_name: str | None = None,
    provider: str = "ollama",
    ollama_model: str = DEFAULT_MODEL,
    ollama_base_url: str = DEFAULT_BASE_URL,
):
    st.subheader("Plan your trip")
    st.caption("Generate a preference-based draft from available places and ranked community discussions.")
    # Outside the form so backend availability updates immediately, without generation.
    provider = st.selectbox(
        "AI backend",
        ["ollama", "openai"],
        index=1 if provider == "openai" else 0,
        format_func=lambda value: "Local model (Ollama)" if value == "ollama" else "OpenAI (cloud)",
        key="trip_ai_provider",
        disabled=demo_mode,
    )
    selected_model = ollama_model if provider == "ollama" else model_name
    ai_available = not demo_mode and (provider == "ollama" or bool(api_key and find_spec("openai")))
    semantic_available = find_spec("sentence_transformers") is not None
    components = load_config().get("costs", {}).get(location)
    initial_budget = float(calculate_costs(components)["daily"]) if components else 0.0
    with st.form("trip_preferences"):
        left, right = st.columns(2)
        with left:
            duration = st.slider("Trip duration (days)", 1, 30, 3, key="trip_duration")
            interests = st.multiselect(
                "Interests",
                INTERESTS,
                default=["food", "history"],
                format_func=str.title,
                key="trip_interests",
            )
            pace = st.selectbox("Pace", list(PACE_LIMITS), index=1, format_func=str.title, key="trip_pace")
        with right:
            daily_budget = st.number_input(
                "Daily budget target in USD (0 = unspecified)",
                min_value=0.0,
                max_value=100_000.0,
                value=initial_budget,
                step=10.0,
                key=f"trip_budget_{location}",
            )
            price_limit = st.selectbox(
                "Maximum venue price tier",
                [None, 0, 1, 2, 3, 4],
                key="trip_price_limit",
                format_func=lambda value: (
                    "Any / unspecified" if value is None else ("Free", "$", "$$", "$$$", "$$$$")[value]
                ),
            )
            st.caption(
                "Budget is a target, not verified spending. Unknown venue prices remain explicitly unverified."
            )
        use_ai = st.checkbox(
            "Use AI for this draft", value=False, key="trip_use_ai", disabled=not ai_available
        )
        st.caption(
            "AI is disabled in demo mode. The no-AI planner remains available."
            if demo_mode
            else (
                f"Local model: {ollama_model}. Requires a running local Ollama server and an installed model; no OpenAI key is needed."
                if provider == "ollama"
                else (
                    "Preferences and selected community evidence are sent to OpenAI; API charges may apply."
                    if ai_available
                    else "OpenAI requires the ai extra and a configured key. The no-AI planner remains available."
                )
            )
        )
        st.caption(
            "Google Maps details are not sent to either model; candidates use opaque labels. No cloud fallback runs automatically."
        )
        semantic = st.checkbox(
            "Use semantic retrieval", value=False, key="trip_semantic", disabled=not semantic_available
        )
        st.caption(
            "Optional semantic search downloads model weights on first generation. Default search uses local TF-IDF."
        )
        submitted = st.form_submit_button("Generate itinerary")

    preferences = TripPreferences(
        duration=duration,
        interests=interests,
        pace=pace,
        daily_budget=daily_budget or None,
        max_price_level=price_limit,
    ).to_dict()
    use_ai = bool(use_ai and ai_available)
    semantic = bool(semantic and semantic_available)
    fingerprint = request_fingerprint(
        location,
        preferences,
        posts,
        places,
        demo=demo_mode,
        ai=use_ai,
        semantic=semantic,
        model=selected_model,
        provider=provider,
        local_endpoint=ollama_base_url if provider == "ollama" else None,
    )
    drafts = st.session_state.setdefault("trip_drafts", {})
    # A hot reload from Phase 2 may leave drafts without provider provenance.
    # Drop those unversioned in-memory drafts before rendering or exporting them.
    for key in list(drafts):
        if drafts[key].get("policy_version") != 1:
            del drafts[key]
    draft_key = (location, demo_mode)
    result = None
    if submitted:
        with st.spinner("Retrieving evidence and building your draft..."):
            embedding_model = None
            if semantic:
                try:
                    embedding_model = EmbeddingModel(encoder=get_semantic_encoder())
                except Exception:
                    st.warning("Semantic retrieval is unavailable; using local TF-IDF search instead.")
            service = TripPlanningService(
                api_key=api_key,
                model_name=selected_model,
                embedding_model=embedding_model,
                provider=provider,
                ollama_base_url=ollama_base_url,
            )
            try:
                result = service.generate(
                    location, preferences, posts, places, use_ai=use_ai, demo_mode=demo_mode
                )
            except (ValueError, OSError) as error:
                st.error(f"Could not build the itinerary: {error}")
                return
            drafts[draft_key] = {
                "fingerprint": fingerprint,
                "result": persistable_itinerary(result),
                "policy_version": 1,
            }
    stored = drafts.get(draft_key)
    if stored is None:
        st.info(
            "Choose your preferences and click Generate itinerary. No AI request or model download runs automatically."
        )
        return
    if stored["fingerprint"] != fingerprint:
        st.warning(
            "Preferences or source data changed. This is your previous draft; generate again to update it."
        )
    result = result if result is not None else hydrate_itinerary(stored["result"], places)
    plan_preferences = result["user_preferences"]
    st.subheader(f"{location} — {plan_preferences['duration']}-day draft")
    st.caption(
        f"{('AI-assisted (' + result.get('ai_model', 'configured model') + ')') if result['ai_generated'] else 'No-AI preference-based'} planning · "
        f"{plan_preferences['pace'].title()} pace · {result['retrieval_method']} retrieval"
    )
    if result["demo_mode"]:
        st.warning("This itinerary uses fictional demo places, not real businesses or traveler evidence.")
    if result.get("generation_warning"):
        st.warning(result["generation_warning"])
    for limitation in result["limitations"]:
        st.warning(limitation)
    for day in result["days"]:
        with st.expander(day["title"], expanded=day["day"] == 1):
            if not day["activities"]:
                st.info(
                    "No additional unique places are available. Load place data or adjust your price-tier limit."
                )
            for activity in day["activities"]:
                if activity.get("source") == "google_places":
                    with st.container(border=True):
                        st.text(f"{activity['time']} · {activity['activity']}")
                        st.caption("Current place details from Google Maps; not stored in the draft.")
                        if activity.get("address"):
                            st.text(activity["address"])
                        st.caption(f"Venue price tier: {activity.get('price', 'Price unavailable')}")
                        st.markdown(attribution_html(activity), unsafe_allow_html=True)
                        maps_link = google_maps_link(activity)
                        if maps_link:
                            st.link_button("View place on Google Maps", maps_link)
                else:
                    st.text(f"{activity['time']} · {activity['activity']}")
                    st.caption(f"Venue price tier: {activity.get('price', 'Price unavailable')}")
                st.text(activity["description"])
                for reason in activity.get("match_reasons", []):
                    st.caption(reason)
                for warning in activity.get("warnings", []):
                    st.warning(warning)
                for index, mention in enumerate(activity.get("community_mentions", []), 1):
                    if safe_url(mention["url"]) != "#":
                        st.link_button(f"Discussion {index} (not a verified endorsement)", mention["url"])
    with st.expander("Budget assumptions and limitations"):
        for note in result["budget_notes"]:
            st.text(note)
    with st.expander("Retrieved community evidence"):
        if not result["retrieved_posts"]:
            st.info("No qualifying community evidence is available for this destination.")
        for post in result["retrieved_posts"]:
            st.text(post.get("title", "Untitled"))
            st.text(str(post.get("summary") or post.get("text") or "")[:500])
            st.caption(
                f"Quality score: {post['quality_score']:.0f}/100 · sentiment: {post.get('sentiment_label', 'neutral')}"
            )
            if is_demo_post(post):
                st.caption("Fictional demo discussion; not used as itinerary evidence.")
            elif safe_url(post.get("url")) != "#":
                st.link_button("Read community discussion", post["url"])
    if stored["result"].get("export_notice"):
        st.caption(stored["result"]["export_notice"])
    st.download_button(
        "Download itinerary JSON",
        json.dumps(stored["result"], indent=2, ensure_ascii=False),
        file_name=f"nomadai_{location_slug(location)}_itinerary.json",
        mime="application/json",
        key="trip_download",
        on_click="ignore",
    )
