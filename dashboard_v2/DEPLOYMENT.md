# Dashboard v2 — Streamlit Community Cloud deployment

This is a one-time setup (~30 min) to make the dashboard accessible at a public URL for user testing.

## 0. Prerequisites

- A GitHub account.
- The project pushed to a GitHub repo (public if you're on free tier, or pay $20/mo for one private-repo Streamlit Cloud workspace).
- The repo must include:
  - `dashboard_v2/app.py` (entrypoint)
  - `dashboard_v2/requirements.txt` (dashboard deps; already created)
  - `dashboard_v2/.streamlit/config.toml` (already created)
  - At least one pipeline output directory under `results/competency_v2_*/` so the dashboard has data to render.

## 1. Push to GitHub

```powershell
# from the project root
git add dashboard_v2/ curriculum_coverage.py user_ratings.py translator.py competency_*.py clustering/ llm_client_router.py kkni.py PYTHON_ENV.md
git add results/competency_v2_pipeline_e2e_v1/competencies.json results/competency_v2_pipeline_e2e_v1/batch_reasonings.json results/competency_v2_pipeline_e2e_v1/clusters.json results/competency_v2_pipeline_e2e_v1/pipeline_report.json
git commit -m "Ship v2 pipeline + Streamlit dashboard"
git push origin main
```

## 2. Sign in to Streamlit Community Cloud

1. Go to https://streamlit.io/cloud
2. Sign in with GitHub.
3. Authorize Streamlit to read your repos.

## 3. Create a new app

1. Click **"New app"**.
2. **Repository**: your repo (`<user>/<repo>`).
3. **Branch**: `main`.
4. **Main file path**: `dashboard_v2/app.py`.
5. **Python version**: 3.11 or 3.12 (Streamlit Cloud's defaults; the app is 3.13-compatible too but the runtime may not be).
6. Click **"Advanced settings"** → **"Secrets"** and paste (in TOML format):

   ```toml
   # Same content as your local api_keys/* files
   OPENROUTER_API_KEY = "sk-or-..."
   JATEVO_API_KEY     = "sk-clb-..."
   HF_HOME            = "/tmp/hf_cache"          # avoid the read-only system path
   RATINGS_HASH_SALT  = "<generate a random 32-char string here>"
   ```

   `llm_client_router.py` already reads from `OPENROUTER_API_KEY` / `JATEVO_API_KEY` env vars before falling back to `api_keys/*.txt` files — so Streamlit Cloud's secrets injection works without code changes.

7. Click **Deploy**.

## 4. After deploy

- First boot takes 5–10 min (Streamlit Cloud builds the requirements environment).
- The public URL is shown at the top of the app page once it's live (something like `https://<your-slug>.streamlit.app/`).
- Share that URL with SMK / university contacts for user testing.

## Resource limits to keep in mind

Streamlit Cloud free tier:
- **1 GB memory** — should fit, but with SBERT loaded the headroom is tight. If you OOM, switch to `paraphrase-MiniLM-L3-v2` (much smaller) in `curriculum_coverage.py`.
- **No persistent disk** for writes — `feedback_store/` ratings will reset on each app restart. For production user testing, either:
  - Switch the rating store to a small SQLite-on-GitHub commit pattern, OR
  - Use a hosted CSV (Google Sheets, Notion DB, etc.) for the rating store.
- **Sleep after inactivity** — apps wake on first request (~10 s cold start).

## Troubleshooting

- **Build fails on `hdbscan`**: comment it out of `dashboard_v2/requirements.txt` — the dashboard doesn't import it directly (only the pipeline does).
- **OOM on first request**: see "Resource limits" above.
- **Translator fails silently**: confirm `OPENROUTER_API_KEY` / `JATEVO_API_KEY` secrets are set in Streamlit Cloud.
- **Curriculum upload accepts PDF but returns 0 sections**: pypdf can fail on scanned-image PDFs. Use the "paste text directly" textarea as fallback.

## Privacy note

`feedback_store/public_users.csv` stores **email hashes** (SHA-256 with a project-local salt), never plaintext emails. Treat the file as PII-adjacent — keep the GitHub repo private if you store real user ratings there, or store ratings in a separate location outside the repo.
