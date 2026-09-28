# Nomad AI

A Streamlit travel discovery dashboard that combines stored Reddit discussions with Google Places results. The project also contains processing, embedding, and optional AI itinerary modules.

## What works today

- Explore 25 configured destinations.
- Read community posts from local files or S3 without Reddit credentials.
- Collect fresh Reddit posts explicitly, with results retained for the current session.
- Classify positive, negative, and neutral/mixed discussions with shared sentiment logic.
- Display place data without inventing missing ratings or prices.
- Show illustrative USD budgets calculated from their component costs.
- Generate clearly labeled local demo posts without API credentials.
- Process extracted files while retaining comments, URLs, destination metadata, and UTC timestamps.

The optional AI planner produces structured itinerary drafts through OpenAI, with a basic local fallback. It and the embedding module are **not yet connected to the dashboard**. Opening hours, travel times, and itinerary feasibility are not verified. Airflow and PostgreSQL are not implemented.

Budget estimates are illustrative, not current price quotes. Hong Kong is selectable but has no budget estimate yet.

## Setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/). The committed lockfile fixes dependency versions.

```bash
uv sync --locked
uv run streamlit run streamlit_app.py
```

Alternatively:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

`python main.py` also launches Streamlit using the active environment.

## Try the demo without API keys

```bash
uv run nomadai-extract --demo --location Paris
uv run nomadai-process --demo
uv run streamlit run streamlit_app.py
```

Select Paris and enable **Use demo data** in the sidebar. Demo records are fictional and are saved under `data/demo/`. They never overwrite live data, and normal readers exclude both labeled demo posts and legacy `mock_` records.

## Configure live sources

Copy `.env.example` to `.env` and fill in only the providers you need. Existing `docker/.env` configurations also work. Streamlit Cloud can use its secrets settings. Never commit credentials.

- **Reddit:** approved API access plus `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and a descriptive user agent. Follow Reddit's [Responsible Builder Policy](https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy).
- **Google Places:** `GOOGLE_PLACES_API_KEY`, with the required APIs and billing enabled. This integration currently uses the legacy Google Maps Python client. Review Google's [storage and attribution policies](https://developers.google.com/maps/documentation/places/web-service/policies) before deployment; a cache timeout alone does not establish permission to retain provider content.
- **S3:** `S3_BUCKET_NAME` and AWS credentials for the collector. The dashboard also supports the standard AWS credential chain and IAM roles.
- **OpenAI:** optional, for the standalone planner. Install with `uv sync --extra ai`; set `OPENAI_API_KEY` and optionally `OPENAI_MODEL` (default: `gpt-4.1-mini`). API charges may apply.

Collect and process real data:

```bash
uv run nomadai-extract --location Paris
uv run nomadai-process
```

Repeat `--location` to select multiple destinations; omit it to collect every configured destination. The collector fails clearly when live credentials are absent. In the dashboard, select fresh data and click **Extract Fresh Reddit Data** to collect; unrelated widget changes reuse the session's previous result.

## Data layout

```text
data/
  by_location/<destination>/<category>/reddit_posts.json
  processed/
    all_processed_posts.json
    high_quality_posts.json
    analytics_summary.json
  summaries/extraction_summary.json
  demo/
    by_location/<destination>/<category>/reddit_posts.json
    processed/
    summaries/extraction_summary.json
```

Categories are travel, food, and events. S3 uses the same destination/category key layout. The app tries S3 first, then local files. `NOMADAI_DATA_DIR` can override the dashboard's local data directory.

Legacy combined JSON files can still be processed with `uv run nomadai-process --input path/to/posts.json`.

## Optional NLP and embeddings

```bash
uv sync --extra ml
uv run python -m spacy download en_core_web_sm
```

The embedding model downloads its sentence-transformers weights when initialized. The base dashboard and processor do not download models or NLTK corpora at startup. Named entity recognition falls back to empty entities when the optional spaCy model is unavailable.

## Development checks

```bash
uv sync --locked
uv run ruff check .
uv run pytest
```

Regression tests cover stored-data loading without credentials, refresh actions, demo isolation, pipeline handoffs, preserved evidence, sentiment negation, destination matching, budget totals, unknown provider values, HTML escaping, and structured AI output validation. Tests use synthetic fixtures and mocked providers; no live API requests are made.

GitHub Actions runs the same lint and test checks on pushes and pull requests.

## Next milestones

Connect the planner and retrieval modules to user preferences in the dashboard, add evidence-based place matching, validate scheduling constraints, and migrate the Places integration. Measure recommendation relevance and itinerary quality before expanding infrastructure.

