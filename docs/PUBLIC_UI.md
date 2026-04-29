# Public Surface — Architecture

> Companion to [PIPELINE.md](../PIPELINE.md) and [README.md](../README.md).
> Describes how the FastAPI app exposes the competency recommender to the public
> at `/`, while keeping admin tools at `/dashboard/admin/*`.

---

## Audiences

| Audience | Surface | Auth |
|----------|---------|------|
| Public (students, teachers, anyone) | `/`, `/competencies`, `/competencies/{id}`, `/coverage`, `/about` | None |
| Light account holder | `/me/coverage`, `/coverage` (with save) | email + password (`role='public'`) |
| Admin | `/dashboard/admin/*` (schools, users, runs, **publish**) | session login (`role='admin'`) |
| Legacy school user | `/dashboard/school/*` | session login (`role='school'`) — retained but no new signups |

---

## Data flow

```
[ scrape jobs ]              admin only
       ↓
[ pipeline (run.bat) ]       admin only — writes results/
       ↓
[ /dashboard/admin/publish ] admin clicks "Publish"
       ↓
[ published_runs row, is_active=1 ]
       ↓
[ public API (api_public.py) ] reads from active row's results_dir
       ↓
[ public UI (templates/public/) ]
```

The public surface never reads `results/` while it is being rewritten — only
the *published* snapshot. Admin must explicitly publish a run for it to go live.

---

## URL map

### Public (anonymous)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Landing page with stage chips and stats |
| GET | `/competencies` | Browse competencies, filter by stage / KKNI level / domain / search |
| GET | `/competencies/{id}` | Competency detail incl. soft skills, related skills, education-demand chart |
| GET | `/coverage` | Curriculum coverage upload form |
| POST | `/coverage` | Curriculum coverage analysis (server-rendered HTML) |
| GET | `/about` | Methodology, KKNI explainer, curriculum format |

### Public (logged-in `role='public'`)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/me/coverage` | List of my saved analyses |
| GET | `/me/coverage/{id}` | View one saved analysis |

### JSON API (read-only, anonymous)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/stages` | Stage→KKNI map for filter chips |
| GET | `/api/published` | Active published run metadata |
| GET | `/api/competencies` | Filtered list (paginated) |
| GET | `/api/competencies/{id}` | Single competency |
| GET | `/api/skills/top` | Top demanded hard skills (filterable by stage / KKNI level) |
| POST | `/api/coverage/analyze` | Upload curriculum, get JSON report |
| GET | `/api/me/coverage` | Logged-in user's saved analyses |
| GET | `/api/me/coverage/{id}` | One saved analysis |

### Auth

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/dashboard/login` | Login form (single page for all roles) |
| POST | `/dashboard/login` | Submit; redirect by role (admin → `/dashboard/admin/schools`, public → `/`, school → `/`) |
| GET | `/signup` | Public signup form |
| POST | `/signup` | Create `role='public'` account, log in, redirect to `/coverage` |
| GET | `/dashboard/logout` | Clear session |

### Admin

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/dashboard/admin/schools` | Schools / departments management |
| GET | `/dashboard/admin/users` | User management |
| GET | `/dashboard/admin/runs` | All pipeline runs |
| GET | `/dashboard/admin/publish` | Publish-a-run page (Phase 2) |
| POST | `/dashboard/admin/publish` | Snapshot `results/` → `published_runs`, set `is_active = 1` |

---

## Database additions (Phase 2 + 5)

```sql
CREATE TABLE published_runs (
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
  is_active INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE coverage_analyses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  curriculum_blob TEXT NOT NULL,    -- JSON
  report_blob TEXT NOT NULL,        -- JSON
  published_run_id INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- users.role CHECK relaxed to allow 'public'
-- users.display_name added (nullable)
```

`db.init_db()` runs an idempotent migration on startup that adds the
`display_name` column and rebuilds the `users` table to relax the CHECK
constraint when the existing schema was written before the reframe.

---

## Domain lockdown

`LOCKED_BIDANG_CODES = {"4"}` in `dashboard/app.py` filters
`spektrum_keahlian.json` so the admin only sees Bidang 4 (Teknologi
Informasi) when creating departments. The public surface is implicitly
locked because it serves whatever the active published run was generated
for. Older Kepmen 130/2017 split RPL and Pengembangan Gim into separate
program codes; admin can still set those by hand via `vocational_field`.

---

## Static assets

Two static directories are mounted:

- `dashboard/static_public/style.css` → `/static/style.css` — public surface CSS
- `dashboard/static/style.css` → `/dashboard/static/style.css` — admin CSS

The admin / login pages link both stylesheets so transitions feel cohesive.

---

## Caching

`api_public._CACHE` keyed by `results_dir` holds the parsed competency JSON +
skill education demand DataFrames. The cache is invalidated on every
successful `POST /dashboard/admin/publish`.

For multi-worker deployments (uvicorn workers > 1) the cache is per-process,
which is fine — the next request will repopulate from disk. For very large
runs you can switch this to an explicit memcached / Redis layer.
