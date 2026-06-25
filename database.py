"""
Database Layer (SQLite, PostgreSQL-ready)
==========================================
Uses plain SQL via sqlite3 with parameterized queries -- no ORM magic,
so it's easy to port: every CREATE TABLE statement below is valid
PostgreSQL too (the only SQLite-specific bit is AUTOINCREMENT, which
maps to SERIAL/IDENTITY in Postgres -- see the comment at the bottom
of get_schema_sql() for the swap).
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database", "resume_platform.db")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    field TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS resumes (
    resume_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    version INTEGER DEFAULT 1,
    field TEXT,
    resume_type TEXT,
    data_json TEXT,
    created_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);

CREATE TABLE IF NOT EXISTS ats_scores (
    score_id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_id INTEGER,
    ats_score REAL,
    breakdown_json TEXT,
    created_at TEXT,
    FOREIGN KEY (resume_id) REFERENCES resumes (resume_id)
);

CREATE TABLE IF NOT EXISTS skills (
    skill_id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_id INTEGER,
    skill_name TEXT,
    skill_category TEXT,
    FOREIGN KEY (resume_id) REFERENCES resumes (resume_id)
);

CREATE TABLE IF NOT EXISTS job_matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_id INTEGER,
    job_title TEXT,
    match_pct REAL,
    missing_skills_json TEXT,
    created_at TEXT,
    FOREIGN KEY (resume_id) REFERENCES resumes (resume_id)
);

CREATE TABLE IF NOT EXISTS salary_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_id INTEGER,
    predicted_salary REAL,
    salary_low REAL,
    salary_high REAL,
    confidence REAL,
    created_at TEXT,
    FOREIGN KEY (resume_id) REFERENCES resumes (resume_id)
);

CREATE TABLE IF NOT EXISTS career_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_id INTEGER,
    success_probability REAL,
    created_at TEXT,
    FOREIGN KEY (resume_id) REFERENCES resumes (resume_id)
);

CREATE TABLE IF NOT EXISTS chat_history (
    chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    role TEXT,
    message TEXT,
    created_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);

CREATE TABLE IF NOT EXISTS model_metadata (
    model_name TEXT PRIMARY KEY,
    algorithm TEXT,
    metrics_json TEXT,
    trained_at TEXT
);

CREATE TABLE IF NOT EXISTS training_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT,
    n_train INTEGER,
    n_val INTEGER,
    n_test INTEGER,
    best_model TEXT,
    test_metrics_json TEXT,
    created_at TEXT
);
"""
# --- Postgres port note ---
# Replace "INTEGER PRIMARY KEY AUTOINCREMENT" with "SERIAL PRIMARY KEY"
# and point sqlite3.connect() at psycopg2.connect(DATABASE_URL) instead.


def get_schema_sql():
    return SCHEMA_SQL


@contextmanager
def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)


def now():
    return datetime.utcnow().isoformat()


# ---------------- Users ----------------
def upsert_user(name, email, field):
    with get_connection() as conn:
        cur = conn.execute("SELECT user_id FROM users WHERE email = ?", (email,))
        row = cur.fetchone()
        if row:
            conn.execute("UPDATE users SET name=?, field=? WHERE user_id=?", (name, field, row["user_id"]))
            return row["user_id"]
        cur = conn.execute(
            "INSERT INTO users (name, email, field, created_at) VALUES (?, ?, ?, ?)",
            (name, email, field, now()),
        )
        return cur.lastrowid


# ---------------- Resumes ----------------
def save_resume(user_id, field, resume_type, data: dict):
    with get_connection() as conn:
        cur = conn.execute("SELECT MAX(version) as v FROM resumes WHERE user_id = ?", (user_id,))
        v = cur.fetchone()["v"]
        version = (v or 0) + 1
        cur = conn.execute(
            "INSERT INTO resumes (user_id, version, field, resume_type, data_json, created_at) VALUES (?,?,?,?,?,?)",
            (user_id, version, field, resume_type, json.dumps(data), now()),
        )
        return cur.lastrowid


def get_resume_history(user_id):
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM resumes WHERE user_id = ? ORDER BY version DESC", (user_id,)
        )
        return [dict(r) for r in cur.fetchall()]


# ---------------- ATS Scores ----------------
def save_ats_score(resume_id, ats_score, breakdown: dict):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO ats_scores (resume_id, ats_score, breakdown_json, created_at) VALUES (?,?,?,?)",
            (resume_id, ats_score, json.dumps(breakdown), now()),
        )


# ---------------- Job Matches ----------------
def save_job_match(resume_id, job_title, match_pct, missing_skills: list):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO job_matches (resume_id, job_title, match_pct, missing_skills_json, created_at) VALUES (?,?,?,?,?)",
            (resume_id, job_title, match_pct, json.dumps(missing_skills), now()),
        )


# ---------------- Salary Predictions ----------------
def save_salary_prediction(resume_id, predicted_salary, salary_low, salary_high, confidence):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO salary_predictions
               (resume_id, predicted_salary, salary_low, salary_high, confidence, created_at)
               VALUES (?,?,?,?,?,?)""",
            (resume_id, predicted_salary, salary_low, salary_high, confidence, now()),
        )


# ---------------- Career Predictions ----------------
def save_career_prediction(resume_id, success_probability):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO career_predictions (resume_id, success_probability, created_at) VALUES (?,?,?)",
            (resume_id, success_probability, now()),
        )


# ---------------- Chat History ----------------
def save_chat_message(user_id, role, message):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_history (user_id, role, message, created_at) VALUES (?,?,?,?)",
            (user_id, role, message, now()),
        )


def get_chat_history(user_id, limit=50):
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM chat_history WHERE user_id = ? ORDER BY chat_id DESC LIMIT ?",
            (user_id, limit),
        )
        return [dict(r) for r in cur.fetchall()][::-1]


# ---------------- Training Logs / Model Metadata ----------------
def log_training_run(task_name, n_train, n_val, n_test, best_model, test_metrics: dict):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO training_logs
               (task_name, n_train, n_val, n_test, best_model, test_metrics_json, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (task_name, n_train, n_val, n_test, best_model, json.dumps(test_metrics), now()),
        )


def upsert_model_metadata(model_name, algorithm, metrics: dict):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO model_metadata (model_name, algorithm, metrics_json, trained_at)
               VALUES (?,?,?,?)
               ON CONFLICT(model_name) DO UPDATE SET
                 algorithm=excluded.algorithm,
                 metrics_json=excluded.metrics_json,
                 trained_at=excluded.trained_at""",
            (model_name, algorithm, json.dumps(metrics), now()),
        )


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
