"""Explicitly fictional provider-shaped examples; never used in live mode."""

from ..data_pipeline.storage import location_slug
from ..utils.content import normalize_place


def demo_places(location: str) -> dict[str, list[dict]]:
    templates = {
        "restaurants": [
            ("Demo Market Cafe", ["restaurant", "cafe"], 1),
            ("Demo Neighborhood Bakery", ["bakery"], 1),
            ("Demo Evening Restaurant", ["restaurant"], 3),
        ],
        "attractions": [
            ("Demo Art Museum", ["museum"], 2),
            ("Demo History Palace", ["historical_landmark"], 2),
            ("Demo Riverside Park", ["park"], 0),
            ("Demo Botanical Garden", ["botanical_garden"], 1),
            ("Demo Shopping Arcade", ["shopping_mall"], 0),
            ("Demo Music Venue", ["music_venue"], 2),
        ],
    }
    return {
        category: [
            {
                **normalize_place(
                    {
                        "name": name,
                        "place_id": f"demo:{location_slug(location)}:{category}:{index}",
                        "price_level": price,
                        "vicinity": f"Fictional example in {location}",
                    }
                ),
                "types": types,
                "source": "demo",
            }
            for index, (name, types, price) in enumerate(places)
        ]
        for category, places in templates.items()
    }
