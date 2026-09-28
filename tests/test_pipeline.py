import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.data_pipeline.data_processor import DataProcessor, main as process_main
from src.data_pipeline.reddit_extractor import ModularRedditExtractor
from src.data_pipeline.storage import deduplicate_posts, load_extracted_posts, load_location_posts, location_slug, validate_extraction_summary
from src.models.ai_trip_planner import AITripPlanner
from src.models.data_quality_enhancer import DataQualityEnhancer


@pytest.mark.parametrize(('text', 'expected'), [
    ('The service was excellent and the food was delicious.', []),
    ('I enjoyed visiting Paris and loved the museum.', ['Paris']),
    ('A trip to NEW YORK and hong kong.', ['Hong Kong', 'New York']),
    ('A parish church and a Berliner pastry.', []),
])
def test_location_extraction_does_not_invent_places(text, expected):
    assert DataProcessor().extract_locations_advanced(text) == expected


def test_processing_preserves_evidence_and_planner_consumes_it(sample_posts, tmp_path):
    processor = DataProcessor()
    posts = processor.process_reddit_posts(sample_posts)
    output = tmp_path / 'processed.json'
    processor.save_processed_data(posts, output)
    saved = json.loads(output.read_text())
    for original, processed in zip(sample_posts, saved, strict=True):
        for key in ('url', 'author', 'target_location', 'top_comments', 'num_comments', 'relevancy_score'):
            assert processed[key] == original[key]
        assert processed['timestamp'].endswith('+00:00')
    context = AITripPlanner().create_reddit_context(saved, 'Paris')
    assert sample_posts[0]['url'] in context
    assert sample_posts[0]['top_comments'][0]['body'] in context


def test_extraction_to_processing_handoff_and_demo_isolation(tmp_path):
    extractor = ModularRedditExtractor(demo_mode=True, data_root=tmp_path / 'data')
    posts = extractor.extract_location_specific_posts('Paris', 'travel', posts_per_subreddit=2)
    assert len(posts) == 2
    extractor.save_to_s3(posts, 'Paris', 'travel')
    assert not (tmp_path / 'data' / 'by_location').exists()
    root = tmp_path / 'data' / 'demo' / 'by_location'
    assert load_location_posts('Paris', root) == []
    loaded = load_extracted_posts(root, include_demo=True)
    assert len(loaded) == 2
    assert {post['source'] for post in loaded} == {'demo'}
    processed = DataProcessor().process_reddit_posts(loaded)
    assert len(processed) == 2
    assert all(post.source == 'demo' for post in processed)


def test_missing_credentials_do_not_generate_sample_posts():
    extractor = ModularRedditExtractor()
    assert extractor._search_subreddit_for_location('travel', 'Paris', 8) == []


def test_live_storage_rejects_demo_posts(tmp_path):
    demo = ModularRedditExtractor(demo_mode=True, data_root=tmp_path)
    posts = demo._generate_mock_location_data('travel', 'Paris', 2)
    live = ModularRedditExtractor(data_root=tmp_path)
    with pytest.raises(ValueError, match='Demo posts'):
        live.save_to_s3(posts, 'Paris', 'travel')
    assert not (tmp_path / 'by_location').exists()


def test_duplicate_submissions_are_collected_once_and_link_to_threads(monkeypatch):
    extractor = ModularRedditExtractor()
    submission = SimpleNamespace(
        id='example', title='A trip to Paris', selftext='Visit Paris museums and local food markets.',
        author='tester', score=10, num_comments=2, created_utc=1767261600,
        permalink='/r/travel/comments/example', url='https://external.example.com',
    )
    search = Mock(return_value=[submission, submission])
    extractor.reddit = SimpleNamespace(subreddit=lambda name: SimpleNamespace(search=search))
    monkeypatch.setattr(extractor, '_extract_comments', lambda post: [])
    monkeypatch.setattr('src.data_pipeline.reddit_extractor.time.sleep', lambda seconds: None)
    posts = extractor._search_subreddit_for_location('travel', 'Paris', 8)
    assert len(posts) == 1
    assert posts[0].url == 'https://www.reddit.com/r/travel/comments/example'
    assert search.call_count == 1


def test_legacy_mock_records_are_excluded_and_repeated_ids_do_not_inflate_counts(sample_posts):
    posts = sample_posts + [sample_posts[0], {**sample_posts[0], 'id': 'mock_Paris_food_0'}]
    assert len(deduplicate_posts(posts)) == 2
    assert len(deduplicate_posts(posts, include_demo=True)) == 3


@pytest.mark.parametrize('invalid', [{'id': []}, {'id': 'test', 'score': None}, {'id': 'test', 'score': '100'}])
def test_malformed_post_identity_and_scores_are_rejected(invalid):
    with pytest.raises(ValueError):
        deduplicate_posts([invalid])


@pytest.mark.parametrize('location', ['../Paris', '../../etc', 'Paris/food'])
def test_location_paths_cannot_escape_storage(location):
    with pytest.raises(ValueError):
        location_slug(location)


def test_local_loader_validates_payload_shape(tmp_path):
    path = tmp_path / 'paris' / 'travel' / 'reddit_posts.json'
    path.parent.mkdir(parents=True)
    path.write_text('{"posts": []}')
    with pytest.raises(ValueError, match='list of post objects'):
        load_location_posts('Paris', tmp_path)


@pytest.mark.parametrize('malformed', [
    [], {'by_location': []}, {'by_location': {'Paris': None}}, {'by_location': {'Paris': {'travel': '2'}}},
])
def test_invalid_summary_counts_are_rejected(malformed):
    with pytest.raises(ValueError):
        validate_extraction_summary(malformed)


def test_demo_summaries_are_not_available_in_live_mode():
    summary = {'source': 'demo', 'by_location': {'Paris': {'travel': 2}}}
    with pytest.raises(ValueError, match='Demo summaries'):
        validate_extraction_summary(summary)
    assert validate_extraction_summary(summary, include_demo=True) == summary


def test_processor_cli_accepts_extractor_layout(sample_posts, tmp_path, monkeypatch):
    path = tmp_path / 'data' / 'by_location' / 'paris' / 'travel' / 'reddit_posts.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(sample_posts))
    monkeypatch.setattr('src.data_pipeline.data_processor.PROJECT_ROOT', tmp_path)
    process_main([])
    saved = json.loads((tmp_path / 'data' / 'processed' / 'all_processed_posts.json').read_text())
    assert len(saved) == 2
    assert saved[0]['url'] == sample_posts[0]['url']


def test_identical_titles_do_not_merge_unrelated_places():
    enhancer = DataQualityEnhancer.__new__(DataQualityEnhancer)
    from sklearn.feature_extraction.text import TfidfVectorizer
    enhancer.tfidf_vectorizer = TfidfVectorizer()
    posts = [
        {'id': 'paris', 'title': 'Weekend recommendations', 'text': 'Paris bakeries and museums.', 'target_location': 'Paris', 'score': 100},
        {'id': 'tokyo', 'title': 'Weekend recommendations', 'text': 'Tokyo temples and ramen.', 'target_location': 'Tokyo', 'score': 80},
    ]
    assert len(enhancer.detect_duplicates(posts)) == 2
    assert 'duplicate_count' not in posts[0]


def test_enhanced_sentiment_strength_is_not_reported_as_confidence():
    enhancer = DataQualityEnhancer.__new__(DataQualityEnhancer)
    sentiment = enhancer.enhanced_sentiment_analysis('I highly recommend the food in Paris.')
    assert sentiment['sentiment_strength'] == abs(sentiment['sentiment_score'])
    assert 'confidence' not in sentiment


def test_repeated_post_engagement_is_not_summed():
    enhancer = DataQualityEnhancer.__new__(DataQualityEnhancer)
    from sklearn.feature_extraction.text import TfidfVectorizer
    enhancer.tfidf_vectorizer = TfidfVectorizer()
    post = {'id': 'same', 'title': 'Paris food', 'text': 'Paris market food recommendation', 'target_location': 'Paris', 'score': 100}
    unique = enhancer.detect_duplicates([post, post.copy()])
    assert len(unique) == 1
    assert unique[0]['combined_score'] == 100
