import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from jsonschema import ValidationError

from src.models.ai_trip_planner import AITripPlanner


def draft_payload(source_urls=None):
    return {
        'days': [{'day': 1, 'title': 'Paris', 'activities': [{
            'time': '9:30 AM', 'activity': 'Test Museum', 'description': 'A suggested visit.',
            'type': 'attraction', 'source_urls': source_urls or [],
        }]}],
        'reddit_tips': [], 'budget_notes': [],
    }


def test_structured_output_parser_validates_schema_and_requested_days():
    planner = AITripPlanner()
    result = planner._parse_ai_response(json.dumps(draft_payload()), 'Paris', {'duration': 1})
    assert result['days'][0]['activities'][0]['activity'] == 'Test Museum'
    assert result['ai_generated'] is True
    with pytest.raises(ValueError, match='duration'):
        planner._parse_ai_response(json.dumps(draft_payload()), 'Paris', {'duration': 2})
    with pytest.raises(ValidationError):
        planner._parse_ai_response('{"days": []}', 'Paris', {'duration': 1})


def test_current_api_uses_structured_output_and_validates_sources(sample_posts):
    planner = AITripPlanner()
    create = Mock(return_value=SimpleNamespace(output_text=json.dumps(draft_payload([sample_posts[0]['url']]))))
    planner.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    posts = [{**post, 'quality_score': 80} for post in sample_posts]
    result = planner.generate_personalized_itinerary('Paris', posts, {'duration': 1}, attractions=[{'name': 'Test Museum'}])
    assert result['ai_generated'] is True
    request = create.call_args.kwargs
    assert request['text']['format']['strict'] is True
    assert request['store'] is False
    assert request['model'] == planner.model_name


@pytest.mark.parametrize('payload', [
    draft_payload(['https://invented.example.com']),
    {**draft_payload(), 'days': []},
    {**draft_payload(), 'days': [{'day': 1, 'title': 'Day', 'activities': [{
        'time': '9:30 AM', 'activity': 'Invented Museum', 'description': '', 'type': 'attraction', 'source_urls': [],
    }]}]},
])
def test_invalid_ai_drafts_fall_back_without_claiming_success(payload):
    planner = AITripPlanner()
    planner.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(output_text=json.dumps(payload))))
    result = planner.generate_personalized_itinerary('Paris', [], {'duration': 1}, attractions=[{'name': 'Test Museum'}])
    assert result['ai_generated'] is False
    assert result['generation_warning']
    assert result['days'][0]['activities'][0]['activity'] == 'Test Museum'


@pytest.mark.parametrize('excluded', [
    {'target_location': 'Tokyo', 'quality_score': 90},
    {'target_location': 'Paris', 'quality_score': 10},
    {'target_location': 'Paris', 'quality_score': 90, 'source': 'demo'},
])
def test_ai_cannot_cite_posts_excluded_from_its_evidence(excluded):
    planner = AITripPlanner()
    url = 'https://www.reddit.com/r/travel/comments/excluded'
    planner.client = SimpleNamespace(responses=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(output_text=json.dumps(draft_payload([url]))),
    ))
    result = planner.generate_personalized_itinerary(
        'Paris', [{'id': 'excluded', 'url': url, **excluded}], {'duration': 1}, attractions=[{'name': 'Test Museum'}],
    )
    assert result['ai_generated'] is False
    assert result['generation_warning']


def test_ai_drafts_cannot_repeat_a_place():
    planner = AITripPlanner()
    payload = draft_payload()
    payload['days'][0]['activities'] *= 2
    planner.client = SimpleNamespace(responses=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(output_text=json.dumps(payload)),
    ))
    result = planner.generate_personalized_itinerary('Paris', [], {'duration': 1}, attractions=[{'name': 'Test Museum'}])
    assert result['ai_generated'] is False
    assert len(result['days'][0]['activities']) == 1


@pytest.mark.parametrize('duration', [0, -1, 31, True, 2.5, 'three'])
def test_invalid_trip_durations_are_rejected(duration):
    with pytest.raises(ValueError, match='duration'):
        AITripPlanner().generate_personalized_itinerary('Paris', [], {'duration': duration})


def test_local_plan_does_not_repeat_places_or_invent_community_endorsements():
    result = AITripPlanner().generate_personalized_itinerary(
        'Paris', [], {'duration': 3}, attractions=[{'place_id': 'museum', 'name': 'Museum'}] * 2,
    )
    activities = [activity for day in result['days'] for activity in day['activities']]
    assert len(activities) == 1
    assert result['reddit_tips'] == []
    assert result['limitations']


def test_context_excludes_demo_and_other_destinations_and_ranks_quality():
    posts = [
        {'id': 'low', 'title': 'Lower quality', 'quality_score': 45, 'target_location': 'Paris'},
        {'id': 'high', 'title': 'Higher quality', 'enhanced_quality_score': 90, 'target_location': 'Paris'},
        {'id': 'other', 'title': 'Other city', 'quality_score': 90, 'target_location': 'Tokyo'},
        {'id': 'mock_Paris', 'title': 'Invented sample', 'quality_score': 100, 'source': 'reddit'},
    ]
    context = AITripPlanner().create_reddit_context(posts, 'Paris')
    assert context.index('Higher quality') < context.index('Lower quality')
    assert 'Other city' not in context
    assert 'Invented sample' not in context
