"""
pipeline_orchestrator.py

Department-scoped runner that keeps the default pipeline untouched.

- Writes run outputs to: data/schools/{school_id}/departments/{department_id}/results
- Falls back to default DATA/results when uploads are missing.
- Uses subprocess calls with explicit CLI args where supported.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import config


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PREP = PROJECT_ROOT / "DATA" / "preprocessing" / "data_prepared"
DEFAULT_RESULTS = PROJECT_ROOT / "results"


@dataclass
class DepartmentPaths:
    school_id: int
    department_id: int
    base: Path
    uploads: Path
    preprocessing: Path
    results: Path
    feedback_store: Path


def department_paths(school_id: int, department_id: int) -> DepartmentPaths:
    base = (
        PROJECT_ROOT
        / "data"
        / "schools"
        / str(school_id)
        / "departments"
        / str(department_id)
    )
    return DepartmentPaths(
        school_id=school_id,
        department_id=department_id,
        base=base,
        uploads=base / "uploads",
        preprocessing=base / "preprocessing",
        results=base / "results",
        feedback_store=base / "feedback_store",
    )


def ensure_department_dirs(paths: DepartmentPaths) -> None:
    for p in [paths.base, paths.uploads, paths.preprocessing, paths.results, paths.feedback_store]:
        p.mkdir(parents=True, exist_ok=True)


def latest_upload(paths: DepartmentPaths, prefix: str) -> Optional[Path]:
    candidates = sorted(paths.uploads.glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _run_cmd(cmd: List[str], env: Optional[dict] = None) -> None:
    completed = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env or os.environ.copy(),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


def _write_run_log(paths: DepartmentPaths, payload: dict) -> None:
    logs_dir = paths.base / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "latest_run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_generate_competencies_cmd(
    results_dir: Path,
    curriculum_file: Optional[Path],
    vocational_field: Optional[str],
    comprehensive: bool = False,
    feedback_dir: Optional[Path] = None,
) -> List[str]:
    """Build generate_competencies.py command with optional --curriculum, --domain, --comprehensive."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "generate_competencies.py"),
        "--output_dir",
        str(results_dir),
    ]
    if comprehensive:
        cmd.append("--comprehensive")
    if feedback_dir and feedback_dir.exists():
        cmd.extend(["--feedback_dir", str(feedback_dir)])
    if curriculum_file and curriculum_file.exists():
        cmd.extend(["--curriculum", str(curriculum_file)])
    if vocational_field and str(vocational_field).strip():
        cmd.extend(["--domain", str(vocational_field).strip()])
    return cmd


def _build_future_weight_mapping_cmd(
    results_dir: Path,
    future_domains_file: Path,
    input_type: str = "knowledge",
    spektrum_code: Optional[str] = None,
) -> List[str]:
    """Build future_weight_mapping.py command with optional --spektrum-code."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "future_weight_mapping.py"),
        "--output_dir",
        str(results_dir),
        "--future_domains_file",
        str(future_domains_file),
        "--input_type",
        input_type,
    ]
    if spektrum_code and str(spektrum_code).strip():
        cmd.extend(["--spektrum-code", str(spektrum_code).strip()])
    return cmd


def _build_evaluate_cmd(script: str, results_dir: Path, spektrum_code: Optional[str] = None) -> List[str]:
    """Build evaluate_extraction.py or evaluate_future_mapping.py command with optional --spektrum-code."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / script),
        "--output_dir",
        str(results_dir),
    ]
    if spektrum_code and str(spektrum_code).strip():
        cmd.extend(["--spektrum-code", str(spektrum_code).strip()])
    return cmd


def _build_log_run_metadata_cmd(
    results_dir: Path,
    spektrum_code: Optional[str] = None,
    future_domains_file: Optional[Path] = None,
    spektrum_mapping_file: Optional[Path] = None,
) -> List[str]:
    """Build log_run_metadata.py command with optional provenance args."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "log_run_metadata.py"),
        "--output_dir",
        str(results_dir),
        "--seed",
        str(config.RANDOM_SEED),
    ]
    if spektrum_code and str(spektrum_code).strip():
        cmd.extend(["--spektrum-code", str(spektrum_code).strip()])
    if future_domains_file:
        cmd.extend(["--future-domains-file", str(future_domains_file)])
    if spektrum_mapping_file and spektrum_mapping_file.exists():
        cmd.extend(["--spektrum-mapping-file", str(spektrum_mapping_file)])
    return cmd


def _build_evaluate_extraction_cmd(results_dir: Path, spektrum_code: Optional[str] = None) -> List[str]:
    """Build evaluate_extraction.py command with optional --spektrum-code."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "evaluate_extraction.py"),
        "--output_dir",
        str(results_dir),
    ]
    if spektrum_code and str(spektrum_code).strip():
        cmd.extend(["--spektrum-code", str(spektrum_code).strip()])
    return cmd


def _build_evaluate_future_mapping_cmd(results_dir: Path, spektrum_code: Optional[str] = None) -> List[str]:
    """Build evaluate_future_mapping.py command with optional --spektrum-code."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "evaluate_future_mapping.py"),
        "--output_dir",
        str(results_dir),
    ]
    if spektrum_code and str(spektrum_code).strip():
        cmd.extend(["--spektrum-code", str(spektrum_code).strip()])
    return cmd


# Item 8: Checkpoint/resume — maps a step script name to the output file it produces.
# If resume=True and the marker file exists, the step is skipped.
_STEP_CHECKPOINTS: dict = {
    "pipeline.py":                    "advanced_skills.csv",
    "verify_skills.py":               "verified_skills.csv",
    "enrich_with_dates.py":           "advanced_skills_with_dates.csv",
    "skill_time_trend_analysis.py":   "skill_time_trends.csv",
    "generate_competencies.py":       "competency_proposals.json",
    "recommendations.py":             "recommendations.csv",
    "evaluate_extraction.py":         "extraction_evaluation_report.json",
    "plot_scientific_analysis.py":    "figures/scientific_calibration_curve.png",
}


def _checkpoint_done(step: List[str], results_dir: Path) -> bool:
    """Return True if this step's output marker already exists."""
    for token in step:
        script = Path(token).name
        marker = _STEP_CHECKPOINTS.get(script)
        if marker:
            return (results_dir / marker).exists()
    return False


def run_department_pipeline(
    school_id: int,
    department_id: int,
    sample_size: int = 1000,
    vocational_field: Optional[str] = None,
    spektrum_code: Optional[str] = None,
    resume: bool = False,
) -> dict:
    """
    Run a department-scoped Phase-1 style pipeline.

    If jobs upload is missing, return fallback metadata and do not run scripts.
    """
    paths = department_paths(school_id, department_id)
    ensure_department_dirs(paths)

    jobs_csv = latest_upload(paths, "jobs")
    curriculum_file = latest_upload(paths, "curriculum")

    if jobs_csv is None:
        payload = {
            "status": "fallback_default_results",
            "reason": "No jobs upload found for department",
            "default_results_dir": str(DEFAULT_RESULTS),
            "default_data_dir": str(DEFAULT_DATA_PREP),
        }
        _write_run_log(paths, payload)
        return payload

    # 1) Preprocess uploaded raw jobs into department preprocessing dir.
    #    Use data_prepared subdir to match default pipeline structure (DATA/preprocessing/data_prepared).
    preprocess_out = paths.preprocessing / "data_prepared"
    preprocess_out.mkdir(parents=True, exist_ok=True)
    preprocess_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "preprocess_jobs_pipeline.py"),
        "--input",
        str(jobs_csv),
        "--output_dir",
        str(preprocess_out.resolve()),  # resolve() to avoid Windows path issues
        "--dedupe",
    ]
    completed = subprocess.run(
        preprocess_cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    preprocess_out_log = f"\nPreprocess STDOUT:\n{completed.stdout}\nPreprocess STDERR:\n{completed.stderr}"

    if completed.returncode != 0:
        raise RuntimeError(
            f"Preprocessing failed: {' '.join(preprocess_cmd)}{preprocess_out_log}"
        )

    input_csv = preprocess_out / "jobs_sentences.csv"
    jobs_meta = preprocess_out / "jobs_metadata.csv"
    if not input_csv.exists():
        listing = list(preprocess_out.iterdir()) if preprocess_out.exists() else []
        raise FileNotFoundError(
            f"Expected preprocessing output not found: {input_csv}\n"
            f"Files in output dir: {[p.name for p in listing]}{preprocess_out_log}"
        )

    # 2) Execute core Phase-1 scripts with output_dir override.
    #    This keeps global run.bat unchanged and isolated.
    steps = [
        [
            sys.executable,
            str(PROJECT_ROOT / "log_run_metadata.py"),
            "--output_dir",
            str(paths.results),
            "--seed",
            str(config.RANDOM_SEED),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "pipeline.py"),
            "--input_csv",
            str(input_csv),
            "--output_dir",
            str(paths.results),
            "--sample_size",
            str(sample_size),
        ],
        [sys.executable, str(PROJECT_ROOT / "plot_generator.py"), "--output_dir", str(paths.results)],
        [sys.executable, str(PROJECT_ROOT / "verify_skills.py"), "--output_dir", str(paths.results)],
        _build_future_weight_mapping_cmd(
            paths.results,
            PROJECT_ROOT / "future_domains.csv",
            input_type="knowledge",
            spektrum_code=spektrum_code,
        ),
        _build_future_weight_mapping_cmd(
            paths.results,
            PROJECT_ROOT / "future_domains.csv",
            input_type="skills",
            spektrum_code=spektrum_code,
        ),
        [
            sys.executable,
            str(PROJECT_ROOT / "enrich_with_dates.py"),
            "--output_dir",
            str(paths.results),
            "--jobs_meta",
            str(jobs_meta),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "skill_time_trend_analysis.py"),
            "--output_dir",
            str(paths.results),
            "--only_hard",
            "--stability",
        ],
        _build_generate_competencies_cmd(paths.results, curriculum_file, vocational_field),
        [
            sys.executable,
            str(PROJECT_ROOT / "recommendations.py"),
            "--output_dir",
            str(paths.results),
            "--ablation",
            "--sensitivity",
        ],
        [sys.executable, str(PROJECT_ROOT / "export_for_review.py"), "--output_dir", str(paths.results)],
        [
            sys.executable,
            str(PROJECT_ROOT / "export_gold_set.py"),
            "--output_dir",
            str(paths.results),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "merge_gold_labels.py"),
            "--labels_dir",
            str(PROJECT_ROOT / "DATA" / "labels"),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "export_competencies_for_review.py"),
            "--output_dir",
            str(paths.results),
        ],
        [sys.executable, str(PROJECT_ROOT / "evaluate_extraction.py"), "--output_dir", str(paths.results)],
        [
            sys.executable,
            str(PROJECT_ROOT / "plot_scientific_analysis.py"),
            "--output_dir",
            str(paths.results),
        ],
    ]

    # Optional curriculum upload is currently tracked for metadata; integration can be added later.
    for step in steps:
        # Item 8: Skip completed steps when resume=True
        if resume and _checkpoint_done(step, paths.results):
            print(f"[RESUME] Skipping (checkpoint exists): {Path(step[1]).name}")
            continue
        # merge_gold_labels may fail if no gold labels exist; run non-fatal
        if "merge_gold_labels.py" in str(step):
            try:
                _run_cmd(step)
            except RuntimeError:
                pass
        else:
            _run_cmd(step)

    payload = {
        "status": "completed",
        "phase": "phase1",
        "results_dir": str(paths.results),
        "preprocessing_dir": str(paths.preprocessing),
        "jobs_upload": str(jobs_csv),
        "curriculum_upload": str(curriculum_file) if curriculum_file else None,
    }
    _write_run_log(paths, payload)
    return payload


def run_department_phase2(
    school_id: int,
    department_id: int,
    vocational_field: Optional[str] = None,
    spektrum_code: Optional[str] = None,
    resume: bool = False,
) -> dict:
    """
    Run Phase 2 (post-review) pipeline for a department.

    Prerequisites:
        - Phase 1 completed (run_department_pipeline)
        - Review done via dashboard/review UI
        - Feedback saved in department feedback_store/

    Steps: import_feedback → apply_feedback → validate_parameters → verify_skills
           → generate_competencies --comprehensive → export_competencies_for_review
           → evaluate_competency_generation → skill_time_trend_analysis
           → recommendations → evaluate_extraction → evaluate_future_mapping
           → log_run_metadata → plot_generator
    """
    paths = department_paths(school_id, department_id)
    ensure_department_dirs(paths)

    if not (paths.results / "expert_review_skills.csv").exists():
        payload = {
            "status": "error",
            "reason": "Phase 1 not completed. Run Phase 1 first and export for review.",
        }
        _write_run_log(paths, payload)
        return payload

    curriculum_file = latest_upload(paths, "curriculum")

    steps = [
        [
            sys.executable,
            str(PROJECT_ROOT / "import_feedback.py"),
            "--output_dir",
            str(paths.results),
            "--feedback_dir",
            str(paths.feedback_store),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "apply_feedback.py"),
            "--output_dir",
            str(paths.results),
            "--feedback_dir",
            str(paths.feedback_store),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "validate_parameters.py"),
            "--output_dir",
            str(paths.results),
            "--feedback_dir",
            str(paths.feedback_store),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "verify_skills.py"),
            "--output_dir",
            str(paths.results),
            "--feedback_dir",
            str(paths.feedback_store),
        ],
        _build_generate_competencies_cmd(
            paths.results,
            curriculum_file,
            vocational_field,
            comprehensive=True,
            feedback_dir=paths.feedback_store,
        ),
        [
            sys.executable,
            str(PROJECT_ROOT / "export_competencies_for_review.py"),
            "--output_dir",
            str(paths.results),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "evaluate_competency_generation.py"),
            "--output_dir",
            str(paths.results),
            "--feedback_dir",
            str(paths.feedback_store),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "skill_time_trend_analysis.py"),
            "--only_hard",
            "--stability",
            "--output_dir",
            str(paths.results),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "recommendations.py"),
            "--ablation",
            "--sensitivity",
            "--evaluate",
            "--output_dir",
            str(paths.results),
        ],
        _build_evaluate_extraction_cmd(paths.results, spektrum_code),
        _build_evaluate_future_mapping_cmd(paths.results, spektrum_code),
        _build_log_run_metadata_cmd(
            paths.results,
            spektrum_code=spektrum_code,
            future_domains_file=PROJECT_ROOT / "future_domains.csv",
            spektrum_mapping_file=_resolve_spektrum_mapping_path(),
        ),
        [
            sys.executable,
            str(PROJECT_ROOT / "plot_generator.py"),
            "--output_dir",
            str(paths.results),
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "plot_scientific_analysis.py"),
            "--output_dir",
            str(paths.results),
        ],
    ]

    try:
        for step in steps:
            _run_cmd(step)
    except RuntimeError as e:
        payload = {"status": "error", "reason": str(e)}
        _write_run_log(paths, payload)
        return payload

    payload = {
        "status": "completed",
        "phase": "phase2",
        "results_dir": str(paths.results),
        "feedback_store": str(paths.feedback_store),
    }
    _write_run_log(paths, payload)
    return payload


def run_load_more_samples(school_id: int, department_id: int) -> dict:
    """
    Export additional skills, knowledge, and competencies for review,
    excluding items already in the existing expert_review_*.csv files.
    Appends new samples to those files.
    """
    paths = department_paths(school_id, department_id)
    ensure_department_dirs(paths)

    results = paths.results
    if not (results / "expert_review_skills.csv").exists():
        return {"status": "error", "reason": "No existing review samples. Run Phase 1 first."}

    steps = [
        [
            sys.executable,
            str(PROJECT_ROOT / "export_for_review.py"),
            "--output_dir",
            str(results),
            "--append",
            "--max_skills",
            "200",
            "--max_knowledge",
            "100",
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "export_competencies_for_review.py"),
            "--output_dir",
            str(results),
            "--append",
            "--max_competencies",
            "50",
        ],
    ]

    try:
        for step in steps:
            _run_cmd(step)
    except RuntimeError as e:
        return {"status": "error", "reason": str(e)}

    return {
        "status": "completed",
        "results_dir": str(results),
    }

