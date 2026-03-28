"""
log_run_metadata.py

Records metadata for the current pipeline run to enable reproducibility.

Outputs:
    results/run_metadata.json

Captured:
    - timestamp
    - input dataset path + row count + SHA256 hash
    - config parameters (sample size, model names, thresholds)
    - model versions (SBERT, JobBERT, LLM)
    - random seed
    - git hash (if available)
    - Python version
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import config


def file_hash(path: Path, algo: str = "sha256") -> str:
    """Compute hex digest of a file."""
    h = hashlib.new(algo)
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def row_count(path: Path) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f) - 1  # minus header
    except Exception:
        return -1


def git_hash() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(config.PROJECT_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="Log run metadata for reproducibility."
    )
    parser.add_argument("--output_dir", type=str, default=str(config.OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spektrum-code", type=str, default=None,
                        help="Spektrum Keahlian code (e.g. 4.1.1) for artifact provenance")
    parser.add_argument("--future-domains-file", type=str, default=None,
                        help="Path to future_domains.csv used for reproducibility")
    parser.add_argument("--spektrum-mapping-file", type=str, default=None,
                        help="Path to spektrum_mapping.csv used for domain filtering")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_csv = Path(config.PIPELINE_INPUT_CSV)

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "project_root": str(config.PROJECT_ROOT),
        "output_dir": str(out_dir),
        "random_seed": args.seed,
        "pipeline_sample_seed": args.seed,
        "git_hash": git_hash(),
        "input_dataset": {
            "path": str(input_csv),
            "exists": input_csv.exists(),
            "rows": row_count(input_csv) if input_csv.exists() else -1,
            "sha256": file_hash(input_csv) if input_csv.exists() else "",
        },
        "models": {
            "jobbert": str(config.JOBBERT_MODEL_NAME),
            "jobbert_checkpoint": str(config.MULTITASK_MODEL_DIR),
            "sbert": "all-MiniLM-L6-v2",
            "llm": "",
            "llm_base_url": "",
            "llm_temperature": None,
        },
        "config_parameters": {
            "output_dir": str(config.OUTPUT_DIR),
        },
        "spektrum_code": args.spektrum_code if args.spektrum_code and str(args.spektrum_code).strip() else None,
        "future_domains_file": args.future_domains_file if args.future_domains_file else None,
        "spektrum_mapping_file": args.spektrum_mapping_file if args.spektrum_mapping_file else None,
    }

    try:
        from pipeline import AdvancedPipelineConfig
        metadata["config_parameters"].update({
            "sample_size": AdvancedPipelineConfig.SAMPLE_SIZE,
            "embedding_model": AdvancedPipelineConfig.EMBEDDING_MODEL,
            "similarity_threshold": AdvancedPipelineConfig.SEMANTIC_AGREEMENT_THRESHOLD,
        })
        metadata["models"]["sbert"] = AdvancedPipelineConfig.EMBEDDING_MODEL
        metadata["models"]["llm"] = getattr(AdvancedPipelineConfig, "LLM_MODEL", "")
        metadata["models"]["llm_base_url"] = getattr(AdvancedPipelineConfig, "OPENAI_BASE_URL", "")
        metadata["models"]["llm_temperature"] = 0
    except Exception:
        pass

    out_path = out_dir / "run_metadata.json"
    out_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Saved run metadata to {out_path}")


if __name__ == "__main__":
    main()
