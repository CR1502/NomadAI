"""Validated preferences shared by local planning, retrieval, and AI drafts."""

import math
from dataclasses import asdict, dataclass

INTERESTS = ("food", "art", "history", "nature", "shopping", "nightlife")
PACE_LIMITS = {"relaxed": 2, "balanced": 3, "busy": 4}


@dataclass(frozen=True)
class TripPreferences:
    duration: int = 3
    interests: tuple[str, ...] = ()
    pace: str = "balanced"
    daily_budget: float | None = None
    max_price_level: int | None = None

    def __post_init__(self):
        if (
            isinstance(self.duration, bool)
            or not isinstance(self.duration, int)
            or not 1 <= self.duration <= 30
        ):
            raise ValueError("Trip duration must be an integer from 1 to 30 days")
        if not isinstance(self.pace, str) or self.pace not in PACE_LIMITS:
            raise ValueError("Pace must be relaxed, balanced, or busy")
        if not isinstance(self.interests, (list, tuple)) or any(
            not isinstance(interest, str) or interest not in INTERESTS for interest in self.interests
        ):
            raise ValueError("Interests must be selected from the supported interests")
        object.__setattr__(self, "interests", tuple(sorted(set(self.interests))))
        if self.daily_budget is not None and (
            isinstance(self.daily_budget, bool)
            or not isinstance(self.daily_budget, (int, float))
            or not math.isfinite(self.daily_budget)
            or not 0 < self.daily_budget <= 100_000
        ):
            raise ValueError("Daily budget must be a finite, positive USD amount up to 100,000")
        if self.max_price_level is not None and (
            isinstance(self.max_price_level, bool)
            or not isinstance(self.max_price_level, int)
            or self.max_price_level not in range(5)
        ):
            raise ValueError("Maximum price level must be from 0 to 4, or unspecified")

    @classmethod
    def from_dict(cls, values: dict) -> "TripPreferences":
        if not isinstance(values, dict):
            raise ValueError("Trip preferences must be an object")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unsupported preferences: {', '.join(sorted(unknown))}")
        return cls(**values)

    @property
    def activities_per_day(self) -> int:
        return PACE_LIMITS[self.pace]

    def to_dict(self) -> dict:
        return {**asdict(self), "interests": list(self.interests)}

    def retrieval_query(self, location: str) -> str:
        vocabulary = {
            "food": "restaurant cafe bakery cuisine market dining",
            "art": "art museum gallery exhibition",
            "history": "history historic monument architecture museum",
            "nature": "nature park garden hiking outdoors",
            "shopping": "shopping market store boutiques",
            "nightlife": "nightlife music bar club evening",
        }
        return " ".join([location, *(vocabulary[interest] for interest in self.interests)])
