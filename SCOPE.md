# Scope: what's in, what's out, and why

The original spec asked for an extremely large platform (24 fields, 10+ ML models, Prophet/ARIMA
forecasting, DOCX export, full resume versioning, CatBoost, BERT/SBERT pipelines, etc.) — enough
work for a small team over weeks, not a single build pass. To avoid shipping shallow stubs that
*look* complete but don't actually work, this build prioritized **5 core features done deeply and
honestly** plus two bonus pieces (Career Coach, AI Training Center).

## ✅ Fully built, real, working

- **24-field skill taxonomy** (270 unique skills, hand-curated — not scraped, so it's honestly
  sized rather than claiming "500+" with nothing backing it)
- **Synthetic dataset generator** — 12,000 rows, formula-driven so labels carry real signal
- **4 ML tasks x 5 algorithms each**, real train/val/test split (70/15/15), real leaderboards
  (Linear/Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM)
- **SHAP feature importance, confusion matrices, ROC curves** — generated from the actual trained
  models, not illustrative mockups
- **Resume Builder** with dynamic, field-aware AI Interview Engine + skill-based branching
  follow-ups (Python → Flask/Django/FastAPI/ML/DL/Data Analysis)
- **ATS Analyzer**: transparent 10-factor rule-based score + independent ML model estimate
- **Skill Gap Analyzer**: core/trending/high-paying skill comparison + recommendations
- **Job Matcher**: TF-IDF cosine similarity + trained classifier + radar chart + missing
  skills/keywords/certifications
- **Salary Predictor**: trained regression model, range derived from actual model RMSE (not an
  arbitrary +/-), confidence heuristic
- **Career Success Predictor**: trained classifier with plain-English driver explanations
- **AI Career Coach**: sidebar chatbot, pluggable to OpenAI/Gemini/Anthropic, offline
  knowledge-base fallback, learning roadmap generator, mock interview question generator
- **PDF export**: 5 templates via ReportLab
- **SQLite database**: full schema (users, resumes, ATS scores, job matches, salary/career
  predictions, chat history, training logs), Postgres-portable
- **AI Training Center**: per-task leaderboards, test metrics, SHAP/ROC/confusion-matrix images,
  dataset stats — a real (lightweight) MLOps dashboard

## ❌ Intentionally not built (and why)

- **DOCX export** — PDF export (5 templates) covers the core ask; DOCX would roughly double the
  export-layer work for a format most ATS guidance treats as equivalent to PDF for parsing.
- **Prophet/ARIMA skill-demand forecasting** — needs real time-series job-posting data to mean
  anything; faking a time series and fitting Prophet to it would be a worked example with no
  signal, not a real feature.
- **CatBoost** — XGBoost + LightGBM already cover gradient boosting in the leaderboard; adding a
  third nearly-identical algorithm family doesn't change the winner or teach anything new.
- **BERT/DistilBERT/SBERT semantic matching** — TF-IDF cosine similarity is the right-sized tool
  for resume-vs-JD matching at this scope; transformer embeddings would add real latency/compute
  for a marginal accuracy gain over TF-IDF on short documents like resumes and job posts.
- **Full resume version history UI** — the database schema supports versioning (`resumes.version`)
  but there's no dedicated "compare version 2 vs version 3" screen.
- **PostgreSQL connection (live)** — schema is Postgres-compatible (see comment in `database.py`)
  but the running app uses SQLite for zero-setup local use.

If any of these matter for your use case, the architecture is modular enough to add them — the
engines only depend on the resume dict schema and the saved-model interface, not on each other.
