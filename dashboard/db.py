from __future__ import annotations

import hashlib
import secrets
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DB_PATH = Path(__file__).resolve().parent / "dashboard.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    return f"{salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    if "$" not in stored:
        return False
    salt, digest = stored.split("$", 1)
    cand = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    return secrets.compare_digest(cand, digest)


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS schools (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            school_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            vocational_field TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(school_id, name),
            FOREIGN KEY (school_id) REFERENCES schools(id)
        );

        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'school', 'public')),
            school_id INTEGER,
            display_name TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (school_id) REFERENCES schools(id)
        );

        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            department_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            message TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            completed_at TEXT,
            config_snapshot TEXT,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        );

        CREATE TABLE IF NOT EXISTS job_uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            department_id INTEGER NOT NULL,
            run_id INTEGER,
            file_path TEXT NOT NULL,
            row_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (department_id) REFERENCES departments(id),
            FOREIGN KEY (run_id) REFERENCES runs(id)
        );

        CREATE TABLE IF NOT EXISTS curriculum_uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            department_id INTEGER NOT NULL,
            run_id INTEGER,
            file_path TEXT NOT NULL,
            format TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (department_id) REFERENCES departments(id),
            FOREIGN KEY (run_id) REFERENCES runs(id)
        );

        -- Phase 2: snapshot of a canonical run published to the public surface.
        -- The public dashboard always reads from the latest is_active row.
        CREATE TABLE IF NOT EXISTS published_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version_label TEXT NOT NULL,
            results_dir TEXT NOT NULL,
            spektrum_code TEXT,
            vocational_field TEXT,
            notes TEXT,
            n_competencies INTEGER DEFAULT 0,
            n_skills INTEGER DEFAULT 0,
            published_at TEXT DEFAULT CURRENT_TIMESTAMP,
            published_by INTEGER,
            is_active INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (published_by) REFERENCES users(id)
        );

        -- Phase 5: saved curriculum-coverage analyses for logged-in public users.
        CREATE TABLE IF NOT EXISTS coverage_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            curriculum_blob TEXT NOT NULL,
            report_blob TEXT NOT NULL,
            published_run_id INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (published_run_id) REFERENCES published_runs(id)
        );
        """
    )

    # Migration: Add Spektrum fields (Kepmen 244/M/2024) - backward compatible
    cur.execute("PRAGMA table_info(departments)")
    cols = {row[1] for row in cur.fetchall()}
    if "spektrum_code" not in cols:
        cur.execute("ALTER TABLE departments ADD COLUMN spektrum_code TEXT")

    # Migration: Add display_name to users (Phase 5 — public-user friendly label)
    cur.execute("PRAGMA table_info(users)")
    user_cols = {row[1] for row in cur.fetchall()}
    if "display_name" not in user_cols:
        cur.execute("ALTER TABLE users ADD COLUMN display_name TEXT")

    # Migration: Relax users.role CHECK to allow 'public' on existing databases.
    # SQLite cannot ALTER a CHECK constraint, so we rebuild the table only when
    # the existing constraint forbids 'public'.
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='users'")
    row = cur.fetchone()
    table_sql = row[0] if row else ""
    if table_sql and "'public'" not in table_sql:
        cur.executescript(
            """
            CREATE TABLE users_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'school', 'public')),
                school_id INTEGER,
                display_name TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (school_id) REFERENCES schools(id)
            );
            INSERT INTO users_new(id, email, password_hash, role, school_id, display_name, created_at)
                SELECT id, email, password_hash, role, school_id,
                       COALESCE(display_name, NULL), created_at
                FROM users;
            DROP TABLE users;
            ALTER TABLE users_new RENAME TO users;
            """
        )
    conn.commit()

    # Seed admin user once.
    cur.execute("SELECT id FROM users WHERE role='admin' LIMIT 1")
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO users(email, password_hash, role) VALUES(?, ?, 'admin')",
            ("admin@local", hash_password("admin123")),
        )
    conn.commit()
    conn.close()


def q_all(sql: str, params: Iterable[Any] = ()) -> List[sqlite3.Row]:
    conn = get_conn()
    rows = conn.execute(sql, tuple(params)).fetchall()
    conn.close()
    return rows


def q_one(sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
    conn = get_conn()
    row = conn.execute(sql, tuple(params)).fetchone()
    conn.close()
    return row


def exec_sql(sql: str, params: Iterable[Any] = ()) -> int:
    conn = get_conn()
    cur = conn.execute(sql, tuple(params))
    conn.commit()
    last_id = cur.lastrowid
    conn.close()
    return int(last_id)

