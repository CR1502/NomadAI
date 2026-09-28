import pytest

from src.utils.content import calculate_costs, classify_sentiment, normalize_place, safe_html, safe_url
from src.utils.helpers import load_config


@pytest.mark.parametrize(('text', 'expected'), [
    ('I do not recommend this restaurant. It was not worth it.', 'negative'),
    ("I don't recommend this restaurant.", 'negative'),
    ('Amazing and beautiful but a terrible overpriced tourist trap. Avoid it.', 'negative'),
    ('I highly recommend this excellent place.', 'positive'),
    ('The food was not bad.', 'positive'),
    ('The train leaves at noon.', 'neutral'),
    ('A badge on the lovable wall.', 'neutral'),
    ('Great food but terrible service.', 'neutral'),
])
def test_sentiment_respects_negation_boundaries_and_mixed_reviews(text, expected):
    assert classify_sentiment(text)['sentiment_label'] == expected


@pytest.mark.parametrize('polarity', [-1.0, 0.0, 1.0])
def test_negative_cues_are_not_overridden_by_positive_keyword_count(polarity):
    text = 'Amazing and beautiful but a terrible overpriced tourist trap. Avoid it.'
    assert classify_sentiment(text, polarity=polarity)['sentiment_label'] != 'positive'


def test_budget_totals_are_derived_and_settings_are_canonical(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = load_config()
    assert len(config['destinations']) == 25
    assert set(config['costs']).issubset(config['destinations'])
    for components in config['costs'].values():
        costs = calculate_costs({**components, 'daily': 999})
        assert costs['daily'] == sum(components.values())
    assert calculate_costs(config['costs']['Istanbul'])['daily'] == 85


@pytest.mark.parametrize('invalid', [-1, True, '100', float('nan'), float('inf')])
def test_invalid_cost_components_are_rejected(invalid):
    with pytest.raises(ValueError):
        calculate_costs({'accommodation': invalid, 'food': 10, 'transport': 5, 'attractions': 5})


def test_unknown_place_values_stay_unknown():
    place = normalize_place({'place_id': 'test', 'name': 'Test restaurant'})
    assert place['rating'] is None
    assert place['price_level'] is None
    assert place['price'] == 'Price unavailable'
    assert normalize_place({'name': 'Free place', 'price_level': 0})['price'] == 'Free'
    assert normalize_place({'name': 'Place', 'price_level': None})['price_level'] is None


def test_place_details_override_search_results():
    place = normalize_place({'name': 'Old name', 'rating': 4.0, 'price_level': 2}, {'name': 'New name', 'rating': 4.5, 'price_level': 3})
    assert (place['name'], place['rating'], place['price']) == ('New name', 4.5, '$$$')


@pytest.mark.parametrize('invalid', [True, '4.5', -1, 6, float('nan'), float('inf')])
def test_invalid_place_ratings_do_not_break_rendering(invalid):
    place = normalize_place({'name': 'Test place', 'rating': invalid, 'user_ratings_total': None})
    assert place['rating'] is None
    assert place['user_ratings_total'] == 0


@pytest.mark.parametrize('url', ['javascript:alert(1)', '//example.com', 'data:text/html,test', None, 'https://example.com\n'])
def test_unsafe_links_are_rejected(url):
    assert safe_url(url) == '#'


def test_external_text_and_url_attributes_are_escaped():
    assert safe_html('<b>test</b>') == '&lt;b&gt;test&lt;/b&gt;'
    assert safe_url('https://example.com/?q="test"&x=1') == 'https://example.com/?q=&quot;test&quot;&amp;x=1'
