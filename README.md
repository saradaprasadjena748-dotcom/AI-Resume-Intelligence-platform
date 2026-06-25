# AI Resume Intelligence Platform

*"Create ATS-Optimized Resumes Powered by Artificial Intelligence"*

A working AI-powered resume platform: **Resume Builder → ATS Analyzer → Skill Gap Analyzer →
Job Matcher → Salary Predictor**, backed by **real machine learning models trained on a real
(synthetic) dataset** — not mocked numbers. An AI Career Coach chatbot and an enterprise
"AI Training Center" MLOps dashboard round it out.

This was built as a **deep MVP**: five core features done properly (real data, real trained
models, real metrics, working UI) rather than a wide shallow scaffold of every feature in the
original spec. See [`SCOPE.md`](SCOPE.md) for exactly what's in vs. out.

---

## ⚠️ Honesty notes (read this first)

- **The dataset is synthetic.** `data/generate_dataset.py` generates 12,000 rows from hand-designed
  formulas + noise (more experience + more relevant skills + better formatting → higher ATS score,
  etc.), not scraped real resumes. The *relationships* are realistic; the *absolute numbers*
  (especially salary) are illustrative, not a market survey. Swap in a real dataset by pointing
  `ml/train_models.py` at a CSV with the same column schema.
- **All model metrics are real.** Every number in the AI Training Center (R², F1, ROC-AUC, MAE...)
  came out of an actual `python ml/train_models.py` run on a 70/15/15 train/val/test split — nothing
  is hardcoded.
- **AI content generation (resume polishing, career coach) needs your own API key.** Without one,
  the platform runs fully offline using rule-based/template fallbacks (clearly labeled in the UI as
  "offline mode"). Add a key in `.env` and it becomes live LLM-generated content with zero code changes.

---

## What's actually in here

| Feature | Status |
|---|---|
| Resume Builder (AI Interview Engine, 24 fields, dynamic skill follow-ups) | ✅ Working |
| ATS Analyzer (10-factor rule-based score + trained ML model + SHAP) | ✅ Working |
| Skill Gap Analyzer (core/trending/high-paying skills, 24-field taxonomy) | ✅ Working |
| Job Matcher (TF-IDF similarity + trained classifier + radar chart) | ✅ Working |
| Salary Predictor (trained regression model + range + confidence) | ✅ Working |
| AI Career Coach (sidebar chatbot, roadmap/mock-interview quick actions) | ✅ Working |
| AI Training Center (leaderboards, confusion matrices, ROC, SHAP plots) | ✅ Working |
| PDF export (5 templates: Modern/Executive/Corporate/Minimal/ATS-Optimized) | ✅ Working |
| SQLite database (users, resumes, scores, matches, chat history) | ✅ Working |
| Career Success Predictor | ✅ Working (bonus, beyond the original 5) |

See `SCOPE.md` for what was intentionally **not** built (DOCX export, Prophet/ARIMA skill
forecasting, full resume-versioning UI, CatBoost, etc.) and why.

---

## Project structure

```
resume_intelligence/
├── app.py                      # Main entry: Resume Builder + Executive Dashboard
├── config.py                   # Settings, API key loading
├── database.py                 # SQLite schema + CRUD helpers
├── requirements.txt
├── .env.example                # Copy to .env and add your API key
├── data/
│   ├── skills_taxonomy.py      # 24 fields x core/trending/high-paying skills + certs
│   ├── generate_dataset.py     # Synthetic dataset generator
│   └── resume_dataset.csv      # Generated dataset (12,000 rows)
├── ml/
│   ├── train_models.py         # Trains + evaluates 4 tasks x 5 algorithms, saves best
│   └── saved/                  # .pkl models, leaderboards, SHAP/ROC/confusion-matrix plots
├── engines/
│   ├── ai_client.py            # Pluggable OpenAI/Gemini/Anthropic client + offline fallback
│   ├── ats_engine.py           # Rule-based + ML ATS scoring
│   ├── resume_generator.py     # AI Interview Engine + resume schema + AI bullet polishing
│   ├── skill_gap_analyzer.py
│   ├── job_matcher.py
│   ├── salary_predictor.py
│   ├── career_success_predictor.py
│   └── career_coach.py         # Chatbot + roadmap + mock interview question generator
├── utils/
│   ├── pdf_generator.py        # ReportLab, 5 templates
│   ├── styling.py               # Shared Plotly charts + glassmorphism UI helpers
│   └── sidebar_chat.py          # Persistent sidebar AI coach widget
├── pages/                       # Streamlit multipage: ATS / Skill Gap / Job Match / Salary / Training Center
├── assets/style.css              # Dark glassmorphism design system
└── tests/test_pages.py           # Streamlit AppTest smoke tests for every page
```

---

## Setup

```bash
cd resume_intelligence
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 1. Generate the dataset (12,000 synthetic rows)
python data/generate_dataset.py --rows 12000

# 2. Train all models (4 tasks x 5 algorithms each, ~30 seconds)
python ml/train_models.py

# 3. (Optional) add your AI API key
cp .env.example .env
# edit .env: set AI_PROVIDER and the matching API key

# 4. Initialize the database (auto-runs on first app launch too)
python database.py

# 5. Run the app
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Adding your AI API key

Edit `.env`:
```
AI_PROVIDER=openai          # or gemini, or anthropic
OPENAI_API_KEY=sk-...
```
No code changes needed. If the key is missing or a call fails for any reason, every AI feature
falls back to its offline template/rule-based version automatically — the app never breaks.

## Retraining on real data

If you have a real resume/salary dataset, give it the same columns as `data/resume_dataset.csv`
(see `data/generate_dataset.py` for the exact schema) and point `ml/train_models.py`'s `DATA_PATH`
at it. Nothing else in the platform needs to change — engines and UI only depend on column names.

## Running tests

```bash
python -m pytest tests/test_pages.py -v
```
Uses Streamlit's `AppTest` framework to load every page with a sample resume in session state and
assert it runs without throwing.

---

## Deployment

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for Streamlit Community Cloud, Docker, and PostgreSQL
migration notes.
