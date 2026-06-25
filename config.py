"""
Central configuration.
Reads API keys from environment variables (recommended) so secrets never
get hardcoded or committed. Copy .env.example to .env and fill in your key,
or export the variable in your shell before running streamlit.
"""

import os

# ---- AI Provider Configuration ----
# Set AI_PROVIDER to "openai", "gemini", "anthropic", or "none".
# "none" runs the platform fully offline with template-based generation.
AI_PROVIDER = os.environ.get("AI_PROVIDER", "openai").lower()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "resume_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "ml", "saved")
DATABASE_DIR = os.path.join(BASE_DIR, "database")

# ---- App ----
APP_NAME = "AI Resume Intelligence Platform"
APP_TAGLINE = "Create ATS-Optimized Resumes Powered by Artificial Intelligence"
PRIMARY_COLOR = "#6C5CE7"
SECONDARY_COLOR = "#00CEC9"
DARK_BG = "#0F1117"


def ai_provider_configured() -> bool:
    """Returns True if a usable API key is present for the selected provider."""
    if AI_PROVIDER == "openai":
        return bool(OPENAI_API_KEY)
    if AI_PROVIDER == "gemini":
        return bool(GEMINI_API_KEY)
    if AI_PROVIDER == "anthropic":
        return bool(ANTHROPIC_API_KEY)
    return False
