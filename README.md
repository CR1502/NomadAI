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
- Build itineraries from duration, interests, pace, a daily budget target, and a maximum venue price tier.
- Retrieve relevant community discussions using local TF-IDF or opt-in semantic search.
- Show explainable place matches, linked discussion mentions, missing-data warnings, and downloadable itinerary JSON.

The dashboard connects the processing, retrieval, and itinerary modules. Local planning requires no OpenAI key or model download. Opt-in OpenAI generation produces validated structured drafts, falling back to local planning when generation fails. Opening hours, travel times, actual costs, accessibility, and itinerary feasibility remain unverified. Airflow and PostgreSQL are not implemented.

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

Select Paris and enable **Use demo data** in the sidebar. Demo discussions are saved under `data/demo/`, and the dashboard supplies explicitly fictional example places. Demo mode never requests Google Places or OpenAI data. It never overwrites live data, and normal readers exclude both labeled demo posts and legacy `mock_` records.

## Plan a trip

Keep **Show Trip Planner** enabled. Choose your preferences and click **Generate itinerary**. Generation only runs on submission, not on unrelated widget changes. Drafts are retained for the current session, scoped to destination and demo/live mode; changed preferences or source data mark the previous draft as stale. Download the draft as JSON when needed.

The local planner ranks places by selected interests, avoids duplicate visits, and limits activities to 2/3/4 per day for relaxed/balanced/busy pace. A known venue price tier above your maximum is excluded; unknown prices remain marked as unverified. Your USD budget is a target, not an enforced spending guarantee. Without enough available places, activity slots stay empty rather than being invented.

Raw posts are processed before retrieval, retaining their provenance. Destination and quality filters exclude unrelated, low-quality, or live-mode demo records. Full provider-name matching links places to retrieved discussions, including comment mentions. A mention is **not a verified endorsement**: overall discussion sentiment is not necessarily sentiment about that particular place. Full-name matching intentionally misses nicknames and ambiguous partial matches.

AI generation is off by default. With the `ai` extra installed and a configured key, enable **Use OpenAI for this draft** before submitting. Preferences, available place data, and selected community evidence are sent to OpenAI; charges may apply. Responses are schema-validated, with additional checks for duration, pace, chronological order, non-repetition, eligible places, and place-specific source links. Times are proposed slots, not verified reservations or routing.

## Configure live sources

Copy `.env.example` to `.env` and fill in only the providers you need. Existing `docker/.env` configurations also work. Streamlit Cloud can use its secrets settings. Never commit credentials.

- **Reddit:** approved API access plus `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and a descriptive user agent. Follow Reddit's [Responsible Builder Policy](https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy).
- **Google Places:** `GOOGLE_PLACES_API_KEY`, with the required APIs and billing enabled. This integration currently uses the legacy Google Maps Python client. Review Google's [storage and attribution policies](https://developers.google.com/maps/documentation/places/web-service/policies) before deployment; a cache timeout alone does not establish permission to retain provider content.
- **S3:** `S3_BUCKET_NAME` and AWS credentials for the collector. The dashboard also supports the standard AWS credential chain and IAM roles.
- **OpenAI:** optional, for opt-in dashboard generation and standalone planning. Install with `uv sync --locked --extra ai` and launch with `uv run --extra ai streamlit run streamlit_app.py`; set `OPENAI_API_KEY` and optionally `OPENAI_MODEL` (default: `gpt-4.1-mini`). Environment variables and Streamlit secrets are supported. API charges may apply.

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
uv sync --locked --extra ml
uv run --extra ml streamlit run streamlit_app.py
```

Enable **Use semantic retrieval** in the planner to use the embedding module. Sentence-transformers weights download only when you explicitly generate a semantic-search draft for the first time. The encoder is shared, but each request owns its index and metadata. If model loading fails, the dashboard warns and uses TF-IDF instead. Cosine search uses normalized vectors and retains source URLs and comments.

The base dashboard and processor do not download models or NLTK corpora at startup. Named entity recognition is separate from semantic retrieval and falls back to empty entities when its optional spaCy model is unavailable. To enable it, run `uv run --extra ml python -m spacy download en_core_web_sm`. Persisted embedding indexes now use non-pickle `.npz` archives; rebuild legacy `.pkl` indexes rather than loading them.

## Development checks

```bash
uv sync --locked
uv run ruff check .
uv run pytest
```

Regression tests cover stored-data loading without credentials, refresh actions, demo isolation, pipeline handoffs, preserved evidence, sentiment negation, destination matching, budget totals, unknown provider values, HTML escaping, preference validation, local and semantic retrieval, safe index persistence, place matching, itinerary constraints, explicit-action generation, session isolation, stale drafts, and structured AI output validation. Tests use synthetic fixtures, fake encoders, and mocked providers; no live API requests or model downloads are made.

GitHub Actions runs the same lint and test checks on pushes and pull requests.

## Next milestones

Migrate the Places integration and address provider attribution/storage requirements before deployment. Then add geographic routing and verified opening-hour constraints, improve ambiguous place matching, and measure recommendation relevance and itinerary quality against evaluation fixtures before expanding infrastructure.
