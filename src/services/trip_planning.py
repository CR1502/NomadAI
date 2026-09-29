"""Coordinate processing, retrieval, and generation without UI dependencies."""

from ..models.ai_trip_planner import AITripPlanner
from ..models.post_retriever import PostRetriever
from ..models.trip_preferences import TripPreferences
from ..utils.helpers import load_config


class TripPlanningService:
    def __init__(
        self,
        *,
        api_key: str = "",
        model_name: str | None = None,
        embedding_model=None,
        provider: str = "openai",
        ollama_base_url: str | None = None,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.provider = provider
        self.ollama_base_url = ollama_base_url

    def generate(
        self,
        location: str,
        preferences: dict,
        posts: list[dict],
        places: dict | None,
        *,
        use_ai: bool = False,
        demo_mode: bool = False,
    ) -> dict:
        preferences = TripPreferences.from_dict(preferences)
        evidence = PostRetriever(
            min_quality=load_config()["ml"]["quality_threshold"],
            embedding_model=self.embedding_model,
        ).retrieve(posts, location, preferences, include_demo=demo_mode)
        planner = AITripPlanner(
            api_key=self.api_key,
            model_name=self.model_name,
            use_ai=use_ai and not demo_mode,
            provider=self.provider,
            ollama_base_url=self.ollama_base_url,
        )
        result = planner.generate_personalized_itinerary(
            location,
            evidence,
            preferences.to_dict(),
            (places or {}).get("restaurants", []),
            (places or {}).get("attractions", []),
            demo_mode=demo_mode,
        )
        if use_ai and not demo_mode and planner.client is None:
            result["generation_warning"] = (
                "AI is not configured or its optional package is unavailable; using local planning."
            )
        elif use_ai and not demo_mode and not result["ai_generated"] and not result.get("generation_warning"):
            result["generation_warning"] = (
                "No eligible places are available; model generation was skipped to avoid inventing venues."
            )
        return {
            **result,
            "retrieved_posts": evidence,
            "retrieval_method": "semantic" if self.embedding_model else "tfidf",
            "demo_mode": demo_mode,
        }
