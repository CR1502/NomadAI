"""Regression checks use synthetic data and never contact external providers."""

import json
from pathlib import Path

import pytest
import streamlit as st
import requests
import dotenv
from botocore.httpsession import URLLib3Session


@pytest.fixture(autouse=True)
def isolate_providers(monkeypatch, tmp_path):
    for key in (
        'REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT',
        'GOOGLE_PLACES_API_KEY', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY',
        'AWS_SESSION_TOKEN', 'S3_BUCKET_NAME', 'OPENAI_API_KEY', 'OPENAI_MODEL',
        'NOMADAI_AI_PROVIDER', 'OLLAMA_MODEL', 'OLLAMA_BASE_URL',
        'GOOGLE_PLACES_DETAIL_LEVEL', 'PRIVACY_POLICY_URL', 'TERMS_OF_USE_URL',
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('NOMADAI_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('AWS_EC2_METADATA_DISABLED', 'true')
    monkeypatch.setattr(dotenv, 'load_dotenv', lambda *args, **kwargs: False)

    def reject_network(*args, **kwargs):
        raise AssertionError('Regression tests must not contact external APIs')

    monkeypatch.setattr(requests.sessions.Session, 'request', reject_network)
    monkeypatch.setattr(URLLib3Session, 'send', reject_network)
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture
def sample_posts():
    return json.loads((Path(__file__).parent / 'fixtures' / 'reddit_posts.json').read_text())
