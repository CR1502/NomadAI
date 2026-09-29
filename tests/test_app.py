import io
import json
import tomllib
from types import SimpleNamespace
from unittest.mock import Mock

import boto3
import praw
import pytest
from botocore.exceptions import ClientError
from streamlit.testing.v1 import AppTest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

from src.utils.helpers import PROJECT_ROOT


def run_app():
    return AppTest.from_file(str(PROJECT_ROOT / 'streamlit_app.py'), default_timeout=20)


def write_local_posts(tmp_path, posts, demo=False):
    root = tmp_path / 'data' / 'demo' if demo else tmp_path / 'data'
    path = root / 'by_location' / 'paris' / 'travel' / 'reddit_posts.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(posts))


def test_app_starts_without_credentials_and_has_all_destinations():
    app = run_app().run()
    assert not app.exception
    assert len(app.selectbox(key='destination').options) == 25
    assert 'Hong Kong' in app.selectbox(key='destination').options
    app.selectbox(key='destination').select('Hong Kong').run()
    assert not app.exception
    assert any('No budget estimate' in info.value for info in app.info)


def test_streamlit_config_keeps_cross_origin_and_xsrf_protection_enabled():
    config = tomllib.loads((PROJECT_ROOT / '.streamlit' / 'config.toml').read_text())
    assert config['server']['enableCORS'] is True
    assert config['server']['enableXsrfProtection'] is True


def test_lockfile_tracks_dynamic_requirements_and_build_cache_inputs():
    pyproject = tomllib.loads((PROJECT_ROOT / 'pyproject.toml').read_text())
    assert {'file': 'requirements.txt'} in pyproject['tool']['uv']['cache-keys']
    requirements = [
        Requirement(line) for line in (PROJECT_ROOT / 'requirements.txt').read_text().splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    ]
    expected = {canonicalize_name(requirement.name): requirement.specifier for requirement in requirements}
    lockfile = tomllib.loads((PROJECT_ROOT / 'uv.lock').read_text())
    package = next(package for package in lockfile['package'] if package['name'] == 'nomadai')
    actual = {
        canonicalize_name(dependency['name']): SpecifierSet(dependency.get('specifier', ''))
        for dependency in package['metadata']['requires-dist'] if 'marker' not in dependency
    }
    assert actual == expected


def test_local_community_data_is_visible_without_reddit_credentials(sample_posts, tmp_path):
    write_local_posts(tmp_path, sample_posts)
    app = run_app().run()
    assert not app.exception
    content = '\n'.join(item.value for item in app.markdown)
    assert sample_posts[0]['title'] in content
    assert sample_posts[1]['title'] in content
    assert 'Stored Reddit Data (2 posts)' in content


def test_stored_s3_data_does_not_require_reddit_credentials(sample_posts, monkeypatch):
    def get_object(**kwargs):
        if kwargs['Key'].endswith('/travel/reddit_posts.json'):
            return {'Body': io.BytesIO(json.dumps(sample_posts).encode())}
        raise ClientError({'Error': {'Code': 'NoSuchKey'}}, 'GetObject')

    monkeypatch.setattr(boto3, 'client', lambda *args, **kwargs: SimpleNamespace(get_object=get_object))
    app = run_app()
    app.secrets['S3_BUCKET_NAME'] = 'test-bucket'
    app.run()
    assert not app.exception
    assert any(sample_posts[0]['title'] in item.value for item in app.markdown)


@pytest.mark.parametrize('malformed', [[None], [{'id': [], 'score': 10}], [{'id': 'test', 'score': None}]])
def test_invalid_s3_records_warn_and_fall_back_to_local_data(malformed, sample_posts, tmp_path, monkeypatch):
    write_local_posts(tmp_path, sample_posts)
    monkeypatch.setattr(boto3, 'client', lambda *args, **kwargs: SimpleNamespace(
        get_object=lambda **kwargs: {'Body': io.BytesIO(json.dumps(malformed).encode())},
    ))
    app = run_app()
    app.secrets['S3_BUCKET_NAME'] = 'test-bucket'
    app.run()
    assert not app.exception
    assert any('Could not read stored' in warning.value for warning in app.warning)
    assert any(sample_posts[0]['title'] in item.value for item in app.markdown)


def test_demo_requires_explicit_selection_and_is_labeled(sample_posts, tmp_path):
    demo_posts = [{**post, 'source': 'demo', 'id': f"mock_{post['id']}"} for post in sample_posts]
    write_local_posts(tmp_path, demo_posts, demo=True)
    app = run_app().run()
    assert not any(sample_posts[0]['title'] in item.value for item in app.markdown)
    app.checkbox(key='demo_mode').check().run()
    assert not app.exception
    assert any('fictional samples' in item.value for item in app.warning)
    assert any('Demo Data (2 posts)' in item.value for item in app.markdown)


def test_refresh_runs_on_request_and_survives_unrelated_reruns(monkeypatch):
    submission = SimpleNamespace(
        id='fresh_test', title='An amazing trip to Paris', selftext='Visit Paris for excellent local food.',
        comments=SimpleNamespace(replace_more=lambda **kwargs: None, list=lambda: []),
        author='tester', score=10, num_comments=0, created_utc=1767261600,
        permalink='/r/travel/comments/fresh_test',
    )
    search = Mock(return_value=[submission])
    monkeypatch.setattr(praw, 'Reddit', lambda **kwargs: SimpleNamespace(subreddit=lambda name: SimpleNamespace(search=search)))
    monkeypatch.setattr('time.sleep', lambda seconds: None)
    app = run_app()
    app.secrets.update({'REDDIT_CLIENT_ID': 'test', 'REDDIT_CLIENT_SECRET': 'test'})
    app.run()
    assert search.call_count == 0
    app.radio(key='reddit_data_source').set_value('Extract Fresh Data').run()
    assert search.call_count == 0
    app.button(key='fetch_reddit').click().run()
    assert not app.exception
    assert search.call_count == 6
    assert any('Fresh Reddit snapshot (1 posts)' in item.value for item in app.markdown)
    app.checkbox(key='show_costs').uncheck().run()
    assert not app.exception
    assert search.call_count == 6
    app.selectbox(key='destination').select('Tokyo').run()
    assert not app.exception
    assert search.call_count == 6


def test_empty_data_collection_button_actually_collects(monkeypatch):
    search = Mock(return_value=[])
    monkeypatch.setattr(praw, 'Reddit', lambda **kwargs: SimpleNamespace(subreddit=lambda name: SimpleNamespace(search=search)))
    monkeypatch.setattr('time.sleep', lambda seconds: None)
    app = run_app()
    app.secrets.update({'REDDIT_CLIENT_ID': 'test', 'REDDIT_CLIENT_SECRET': 'test'})
    app.run()
    app.button(key='collect_empty_reddit').click().run()
    assert not app.exception
    assert search.call_count == 6


def test_external_post_markup_and_unsafe_links_do_not_enter_html(sample_posts, tmp_path):
    post = {
        **sample_posts[0], 'title': '<img src=x onerror=alert(1)> Amazing Paris',
        'text': '<b>Highly recommend Paris</b>', 'url': 'javascript:alert(1)',
        'top_comments': [{'body': '<script>alert(1)</script>', 'author': '<b>author</b>', 'score': 10}],
    }
    write_local_posts(tmp_path, [post])
    app = run_app().run()
    assert not app.exception
    content = '\n'.join(item.value for item in app.markdown)
    assert '<img src=x' not in content
    assert '<script>' not in content
    assert 'javascript:alert' not in content
    assert '&lt;img src=x' in content
