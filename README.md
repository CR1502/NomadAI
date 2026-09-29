# Nomad AI

A Streamlit travel discovery dashboard that combines stored Reddit discussions with Google Places results. The project also contains processing, embedding, and optional AI itinerary modules.

## What works today

- Explore 25 configured destinations.
- Read community posts from local files or S3 without Reddit credentials.
- Collect fresh Reddit posts explicitly, with results retained for the current session.
- Classify positive, negative, and neutral/mixed discussions with shared sentiment logic.
- Search Google Places API (New) with bounded requests, explicit fields, attribution, and honest missing values.
- Show illustrative USD budgets calculated from their component costs.
- Generate clearly labeled local demo posts without API credentials.
- Process extracted files while retaining comments, URLs, destination metadata, and UTC timestamps.
- Build itineraries from duration, interests, pace, a daily budget target, and a maximum venue price tier.
- Retrieve relevant community discussions using local TF-IDF or opt-in semantic search.
- Show explainable place matches, linked discussion mentions, missing-data warnings, and provider-safe itinerary exports.
- Optionally generate validated drafts with a local Ollama model, including `gemma4:12b`.

The dashboard connects the processing, retrieval, and itinerary modules. The no-AI planner requires no model or API key. Opt-in Ollama or OpenAI generation produces validated structured drafts, falling back to the no-AI planner when generation fails. Provider hours, coordinates, and time zones are preserved during live requests for future scheduling, but travel-date opening hours, routing, costs, accessibility, and itinerary feasibility are not yet enforced. Airflow and PostgreSQL are not implemented.

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

Select Paris and enable **Use demo data** in the sidebar. Demo discussions are saved under `data/demo/`, and the dashboard supplies explicitly fictional example places. Demo mode never requests Google Places or AI generation, including local Ollama. It never overwrites live data, and normal readers exclude both labeled demo posts and legacy `mock_` records.

## Plan a trip

Keep **Show Trip Planner** enabled. Choose your preferences and click **Generate itinerary**. Generation only runs on submission, not on unrelated widget changes. Drafts are retained for the current session, scoped to destination and demo/live mode; changed preferences or source data mark the previous draft as stale. Google-backed activities retain only place IDs and proposed slots, not provider names/details or generated descriptions; current details are joined from a fresh provider response for display. JSON downloads use that same redacted draft. Independently sourced Reddit evidence, preferences, and fictional demo content remain exportable. Unversioned drafts from older app sessions are cleared on upgrade.

The local planner ranks places by selected interests, avoids duplicate visits, and limits activities to 2/3/4 per day for relaxed/balanced/busy pace. A known venue price tier above your maximum is excluded; unknown prices remain marked as unverified. Your USD budget is a target, not an enforced spending guarantee. Without enough available places, activity slots stay empty rather than being invented.

Raw posts are processed before retrieval, retaining their provenance. Destination and quality filters exclude unrelated, low-quality, or live-mode demo records. Full provider-name matching links places to retrieved discussions, including comment mentions. A mention is **not a verified endorsement**: overall discussion sentiment is not necessarily sentiment about that particular place. Full-name matching intentionally misses nicknames and ambiguous partial matches.

AI generation is off by default. Select **Local model (Ollama)** or **OpenAI (cloud)**, then enable **Use AI for this draft** before submitting. Both backends use a JSON schema plus checks for duration, pace, chronological order, non-repetition, eligible places, and place-specific citations. Google candidates are replaced with opaque labels in model prompts; Google names, addresses, ratings, coordinates, hours, and IDs are not sent to either model. Independently collected Reddit text may still name places. Times are proposed slots, not verified reservations or routing. Failure never automatically switches to a cloud provider.

### Use your local Gemma model

Ollama is the default dashboard backend; no extra Python package or OpenAI key is required. The default model is your installed `gemma4:12b`.

```bash
ollama list
# Only if the local server is not already running:
ollama serve
```

In `.env` or Streamlit secrets, optionally override:

```dotenv
NOMADAI_AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=gemma4:12b
```

Use the exact installed model tag from `ollama list`. The app does not pull models automatically. Only loopback endpoints are accepted, redirects/proxies are disabled for model requests, and cloud-suffixed Ollama model tags are rejected. This is local to the machine running Streamlit: a cloud-hosted dashboard cannot reach Ollama on your laptop. Disable Ollama cloud features for a strictly local deployment and do not expose an unauthenticated Ollama server publicly. See the official [Ollama API](https://docs.ollama.com/api/chat).

Run a manual smoke check using synthetic places, with no Google/OpenAI calls:

```bash
uv run python scripts/check_local_model.py
```

The base app does not otherwise contact Ollama until an AI-enabled draft with eligible places is submitted. Generation has a bounded timeout and falls back to the no-AI planner when the local server/model is unavailable or its output is invalid.

## Configure live sources

Copy `.env.example` to `.env` and fill in only the providers you need. Existing `docker/.env` configurations also work. Streamlit Cloud can use its secrets settings. Never commit credentials.

- **Reddit:** approved API access plus `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and a descriptive user agent. Follow Reddit's [Responsible Builder Policy](https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy).
- **Google Places:** `GOOGLE_PLACES_API_KEY`, with **Places API (New)** and billing enabled. Restrict the server key to the required API and appropriate server restrictions. The app uses [Text Search (New)](https://developers.google.com/maps/documentation/places/web-service/text-search), not the legacy Python client; Geocoding API is not needed. See the provider guardrails below.
- **S3:** `S3_BUCKET_NAME` and AWS credentials for the collector. The dashboard also supports the standard AWS credential chain and IAM roles.
- **OpenAI:** optional, for opt-in dashboard generation and standalone planning. Install with `uv sync --locked --extra ai` and launch with `uv run --extra ai streamlit run streamlit_app.py`; set `OPENAI_API_KEY` and optionally `OPENAI_MODEL` (default: `gpt-4.1-mini`). Select the cloud backend explicitly, or set `NOMADAI_AI_PROVIDER=openai` for the dashboard default. Uses the [Responses API's structured output](https://developers.openai.com/api/docs/guides/structured-outputs) with `store=False`; preferences and selected community evidence leave your machine, and API charges may apply.

### Places request, billing, and policy guardrails

- At most two search operations per live page rerun: one per enabled category, up to 10 results each. Hidden categories are not requested unless the planner needs them. There is no geocoding, per-place details fan-out, pagination, or wildcard mask. Each operation retries at most once for transient failures; quota/authentication/timeout errors use safe messages and one category's failure does not discard the other's results.
- `GOOGLE_PLACES_DETAIL_LEVEL=standard` preserves ratings, prices, hours, and contact details. Those fields trigger the **Text Search Enterprise** SKU. `basic` requests **Pro** fields only and leaves those extra values unavailable. Fewer requests do not necessarily mean lower charges: review the current field-based billing, configure quotas/budget alerts, and choose the mask deliberately. Photos, reviews, and AI summaries are not requested.
- No Google results are application-cached, written to disk/S3, embedded, or included in model prompts. Search responses live only for the current request. Reruns fetch fresh content, so sidebar/backend changes can trigger new billable searches; fetch timestamps alone do not make a saved draft stale. JSON downloads do not themselves trigger a rerun (requires Streamlit 1.44+, now the minimum). A cached transport contains no result data. Google-backed saved/exported activities contain IDs and proposed schedule slots, not provider details. If a place is absent from the current search, its saved ID remains with a Maps link and an explicit missing-details placeholder.
- Compact result cards display **Google Maps** text attribution and supplied third-party attribution links, separated from community/AI explanations. Links and provider-controlled strings are escaped. Do not combine this data with a non-Google map, use it to train/evaluate models, or reuse it outside permitted terms. These safeguards are not a blanket legal-compliance guarantee; consult Google's current [Places policies](https://developers.google.com/maps/documentation/places/web-service/policies), [Terms](https://cloud.google.com/maps-platform/terms), and region-specific requirements before public release.
- Public deployment still requires your own publicly accessible Terms of Use and Privacy Policy pages incorporating Google's required terms/policies. Configure `TERMS_OF_USE_URL` and `PRIVACY_POLICY_URL`; the app shows a deployment warning when either is missing. Supplying URLs alone does not prove their contents comply.
- Results are category searches mentioning the destination, not a verified city boundary or an 8 km geofence. Known temporarily/permanently closed businesses are excluded. Current/regular opening-hour metadata is a provider snapshot, not verified availability on your travel dates; unknown values remain unknown. Geographic routing and date-aware hours are Phase 4.

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

Regression tests cover stored-data loading, demo isolation, pipeline handoffs, preserved evidence, retrieval/ranking, itinerary constraints, explicit-action generation, stale/session-isolated drafts, new Places payloads and masks, bounded provider retries, malformed responses, metadata/attribution preservation, export redaction, local-only endpoint enforcement, no cloud fallback, and both AI backends' structured validation. Tests use synthetic fixtures, fake encoders, and mocked providers; no live API requests, local model inference, or model downloads are made. The manual smoke script is separate from the regression suite.

GitHub Actions runs the same lint and test checks on pushes and pull requests.

## Next milestones

Next: geographic routing and date-aware opening-hour constraints (Phase 4). Then improve ambiguous place matching and measure recommendation relevance/itinerary quality against independently licensed or synthetic evaluation fixtures before expanding infrastructure. Complete the deployment policy checklist before making live Google Places data publicly available.
