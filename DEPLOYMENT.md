# Deployment Guide

## 1. Local (already covered in README.md)

```bash
pip install -r requirements.txt
python data/generate_dataset.py --rows 12000
python ml/train_models.py
streamlit run app.py
```

## 2. Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. Go to share.streamlit.io → New app → point at `app.py`.
3. **Important:** `ml/saved/*.pkl` and `data/resume_dataset.csv` must be committed to the repo
   (or regenerated on startup) — Streamlit Cloud doesn't persist a build step that runs
   `train_models.py` for you. Either:
   - Commit the generated dataset + trained `.pkl` files directly, **or**
   - Add a one-time startup check in `app.py` that calls
     `data/generate_dataset.py` + `ml/train_models.py` if `ml/saved/` is empty
     (adds ~30s to cold start, acceptable for a demo).
4. In the app's "Secrets" panel, add your API key the same way you would in `.env`:
   ```toml
   AI_PROVIDER = "openai"
   OPENAI_API_KEY = "sk-..."
   ```
   `config.py` reads from `os.environ`, and Streamlit Cloud injects secrets as environment
   variables automatically.
5. SQLite's `database/resume_platform.db` is **ephemeral** on Streamlit Cloud (resets on redeploy/
   restart) — fine for a demo, not for production persistence. See the Postgres section below if
   you need durability.

## 3. Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python data/generate_dataset.py --rows 12000 && python ml/train_models.py
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

```bash
docker build -t resume-platform .
docker run -p 8501:8501 -e AI_PROVIDER=openai -e OPENAI_API_KEY=sk-... resume-platform
```

## 4. Migrating from SQLite to PostgreSQL

`database.py` is plain parameterized SQL (no ORM), and every `CREATE TABLE` statement is valid
Postgres except for `AUTOINCREMENT`. To migrate:

1. In `SCHEMA_SQL`, replace `INTEGER PRIMARY KEY AUTOINCREMENT` with `SERIAL PRIMARY KEY`.
2. Replace the `get_connection()` context manager's `sqlite3.connect(DB_PATH)` with:
   ```python
   import psycopg2
   conn = psycopg2.connect(os.environ["DATABASE_URL"])
   ```
3. SQLite's `ON CONFLICT(model_name) DO UPDATE SET ...` syntax (used in `upsert_model_metadata`)
   is already Postgres-compatible — no change needed there.
4. Everything else (the CRUD functions) works unchanged since they only use standard parameterized
   `?`-style placeholders — swap to `%s` placeholders for psycopg2, or use `psycopg2.extras` with
   named params if you prefer.

## 5. Environment variables reference

| Variable | Required | Default | Notes |
|---|---|---|---|
| `AI_PROVIDER` | No | `openai` | `openai`, `gemini`, `anthropic`, or `none` |
| `OPENAI_API_KEY` | If using OpenAI | — | |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | |
| `GEMINI_API_KEY` | If using Gemini | — | |
| `GEMINI_MODEL` | No | `gemini-1.5-flash` | |
| `ANTHROPIC_API_KEY` | If using Claude | — | |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-6` | |

If no key is set for the selected provider, every AI feature degrades to its offline
template/rule-based fallback automatically — no crash, no missing feature, just lower-quality
generated text until a key is added.
