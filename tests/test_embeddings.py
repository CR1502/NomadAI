import json

import numpy as np
import pytest

from src.models.embedding_model import EmbeddingModel
from src.models.post_retriever import PostRetriever
from src.models.trip_preferences import TripPreferences


class FakeEncoder:
    def encode(self, texts, **kwargs):
        return np.asarray([[2, 0] if "museum" in text.lower() else [0, 10] for text in texts], dtype=float)


def records():
    return [
        {
            "id": "art",
            "title": "Museum",
            "text": "Art collections",
            "quality_score": 80,
            "target_location": "Paris",
            "url": "https://example.com/art",
            "top_comments": [{"body": "Visit early"}],
        },
        {
            "id": "park",
            "title": "Park",
            "text": "Outdoor gardens",
            "quality_score": 80,
            "target_location": "Paris",
        },
        {"id": "wrong", "title": "Museum", "quality_score": 100, "target_location": "Parisian"},
    ]


def test_cosine_search_normalizes_vectors_and_keeps_source_metadata():
    model = EmbeddingModel(encoder=FakeEncoder())
    model.create_embeddings(records())
    found = model.find_similar("museum art", location_filter="PARIS")
    assert [result["metadata"]["id"] for result in found] == ["art", "park"]
    assert found[0]["similarity"] == pytest.approx(1)
    assert found[1]["similarity"] == pytest.approx(0)
    assert found[0]["metadata"]["url"] == "https://example.com/art"
    assert found[0]["metadata"]["top_comments"][0]["body"] == "Visit early"
    assert len(model.get_location_recommendations("Paris")) == 2
    assert model.find_similar("museum", top_k=0) == []


def test_semantic_retrieval_is_optional_and_has_a_request_local_index():
    model = EmbeddingModel(encoder=FakeEncoder())
    found = PostRetriever(embedding_model=model).retrieve(
        records(), "Paris", TripPreferences(interests=["art"])
    )
    assert found[0]["id"] == "art"
    assert all(post["retrieval_method"] == "semantic" for post in found)
    assert len(model.metadata) == 2


def test_embedding_archives_are_non_pickle_and_round_trip(tmp_path):
    model = EmbeddingModel(encoder=FakeEncoder())
    data = model.create_embeddings(records())
    archive = tmp_path / "nested" / "index.npz"
    model.save_embeddings(data, archive)
    loaded = EmbeddingModel(encoder=FakeEncoder()).load_embeddings(archive)
    assert loaded["metadata"] == data["metadata"]
    assert loaded["texts"] == data["texts"]
    assert loaded["model_name"] == model.model_name
    with np.load(archive, allow_pickle=False) as saved:
        assert json.loads(str(saved["metadata"].item()))[0]["url"] == "https://example.com/art"
    assert np.allclose(loaded["embeddings"], data["embeddings"])


def test_legacy_pickle_files_are_never_loaded(tmp_path):
    model = EmbeddingModel(encoder=FakeEncoder())
    with pytest.raises(ValueError, match="legacy pickle"):
        model.load_embeddings(tmp_path / "legacy.pkl")
    with pytest.raises(ValueError, match="pickle"):
        model.save_embeddings(model.create_embeddings(records()), tmp_path / "index.pkl")


def test_model_mismatch_is_rejected(tmp_path):
    model = EmbeddingModel(encoder=FakeEncoder())
    archive = tmp_path / "index.npz"
    model.save_embeddings(model.create_embeddings(records()), archive)
    with pytest.raises(ValueError, match="different embedding model"):
        EmbeddingModel(model_name="different", encoder=FakeEncoder()).load_embeddings(archive)


@pytest.mark.parametrize("invalid", [[[float("nan"), 1]], [[float("inf"), 1]], [1, 2]])
def test_bad_embeddings_are_rejected(invalid):
    with pytest.raises(ValueError):
        EmbeddingModel._normalize_vectors(invalid)


def test_empty_index_and_zero_vectors_are_handled_without_nan():
    model = EmbeddingModel(encoder=FakeEncoder())
    assert model.create_embeddings([])["embeddings"].shape == (0, 0)
    assert model.find_similar("museum") == []
    assert np.array_equal(model._normalize_vectors([[0, 0]]), [[0, 0]])
