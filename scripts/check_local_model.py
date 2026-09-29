"""Manual local-model smoke test with synthetic input; never calls Google/OpenAI."""

import argparse
import json
import os

from src.models.ai_trip_planner import AITripPlanner
from src.services.ollama import DEFAULT_BASE_URL, DEFAULT_MODEL


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL))
    args = parser.parse_args()
    planner = AITripPlanner(
        provider="ollama",
        model_name=args.model,
        ollama_base_url=args.base_url,
        api_key="",
    )
    result = planner.generate_personalized_itinerary(
        "Paris",
        [],
        {"duration": 1, "pace": "relaxed", "interests": ["food", "history"]},
        restaurants=[
            {
                "name": "Fixture Cafe",
                "place_id": "fixture-cafe",
                "source": "fixture",
                "types": ["restaurant"],
                "price_level": 1,
            }
        ],
        attractions=[
            {
                "name": "Fixture Museum",
                "place_id": "fixture-museum",
                "source": "fixture",
                "types": ["museum"],
                "price_level": 1,
            }
        ],
    )
    activities = [activity for day in result["days"] for activity in day["activities"]]
    passed = result["ai_generated"] and len(activities) == 2
    print(
        json.dumps(
            {
                "passed": bool(passed),
                "model": args.model,
                "ai_generated": result["ai_generated"],
                "activities": len(activities),
                "warning": result.get("generation_warning"),
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
