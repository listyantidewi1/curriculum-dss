"""
skill_llm_offline.py — offline batch backend for the Skill-LLM Layer-1 extractor.

Skill-LLM is an 8B-parameter LoRA fine-tune of LLaMA 3.1 Instruct (see
baseline_versions/skill_llm/). At 4-bit quantization it needs ~10-12 GB
VRAM for inference; consumer 4 GB GPUs cannot run it locally. The
production deployment story (per docs/EXTRACTOR_DECISION.md) is therefore
offline batch inference: a Kaggle notebook (or HF Inference Endpoint
post-deadline) processes all sentences once, writes results to a JSONL
file, and pipeline.py loads that file at runtime instead of running the
model live.

This module is the local-side half of that workflow. The Kaggle-side
script is at baseline_versions/skill_llm/kaggle/run_inference_on_kaggle.py.

Activate by setting AdvancedPipelineConfig.EXTRACTION_MODE = "skill_llm_offline"
in pipeline.py.

File format (one JSON object per line, UTF-8, produced by the Kaggle script):
    {
      "sentence_id":   "job123_0001",
      "sentence_text": "Strong Python skills required",
      "SKILL":         [{"skill_span": "implementing CI/CD", "context": "..."}],
      "KNOWLEDGE":     [{"skill_span": "Python",            "context": "..."}],
      "model":         "skill_llm_8b_lora_v1",
      "extracted_at":  "2026-05-12T03:00:00Z"
    }

The SKILL/KNOWLEDGE arrays may be empty when the sentence has no extractions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


# Skill-LLM does not emit per-span confidence (the underlying LLM produces
# JSON, not logit-aligned token scores). We assign a constant calibrated to
# the model's measured precision on SkillSpan: skill precision 0.583,
# knowledge precision 0.660 (see baseline_versions/skill_llm/outputs/trained/
# metrics_test.txt). 0.90 is a conservative single-number proxy used for
# downstream confidence weighting; the verb-preservation and grounding gates
# remain the real quality controls.
SKILL_LLM_CONFIDENCE = 0.90


class SkillLLMOfflineExtractor:
    """Loads pre-computed Skill-LLM extractions from a JSONL file.

    The extractor returns (skills, knowledge) tuples in the same shape as
    ModelManager.extract_with_bert: lists of `{"text": str, "confidence": float}`
    dicts. This is intentional — the pipeline's downstream code treats this
    extractor as a drop-in replacement for the BERT-path Layer 1.

    Lookup order on extract():
      1. by sentence_id (canonical, set by Phase 1.1 provenance)
      2. by exact sentence_text (fallback for legacy inputs without sentence_id)
      3. (None, None) — returns empty lists; pipeline proceeds without Layer 1
         output for this sentence
    """

    def __init__(self, extractions_path: Path):
        self.extractions_path = Path(extractions_path)
        self._by_sentence_id: Dict[str, dict] = {}
        self._by_sentence_text: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.extractions_path.exists():
            raise FileNotFoundError(
                f"Skill-LLM extractions not found at {self.extractions_path}.\n"
                f"To produce this file:\n"
                f"  1. Run preprocess_jobs_pipeline.py to make jobs_sentences.csv\n"
                f"  2. Run scripts/export_sentences_for_skill_llm.py to make a Kaggle-ready JSONL\n"
                f"  3. Upload that JSONL to Kaggle, run baseline_versions/skill_llm/kaggle/"
                f"run_inference_on_kaggle.py via Save Version\n"
                f"  4. Download the output JSONL and place it at {self.extractions_path}\n"
                f"See baseline_versions/skill_llm/INTEGRATION.md for the full workflow."
            )

        n_loaded = 0
        n_dupes = 0
        with open(self.extractions_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSONL at {self.extractions_path}:{line_no}: {exc}"
                    ) from exc

                sid = (rec.get("sentence_id") or "").strip()
                stext = (rec.get("sentence_text") or "").strip()
                if sid:
                    if sid in self._by_sentence_id:
                        n_dupes += 1
                    self._by_sentence_id[sid] = rec
                if stext:
                    # Only set if not already present; sentence_id takes priority
                    self._by_sentence_text.setdefault(stext, rec)
                n_loaded += 1

        print(
            f"[SkillLLMOfflineExtractor] loaded {n_loaded} records "
            f"({len(self._by_sentence_id)} unique sentence_ids, "
            f"{len(self._by_sentence_text)} unique sentence_texts, "
            f"{n_dupes} duplicate sentence_ids) from {self.extractions_path}"
        )

    def extract(
        self,
        sentence_text: str,
        sentence_id: str = "",
    ) -> Tuple[List[Dict], List[Dict]]:
        """Return (skills, knowledge) for the given sentence.

        Each item is `{"text": str, "confidence": float}` matching
        ModelManager.extract_with_bert's output shape.
        """
        rec = None
        if sentence_id and sentence_id in self._by_sentence_id:
            rec = self._by_sentence_id[sentence_id]
        else:
            stext_norm = (sentence_text or "").strip()
            if stext_norm and stext_norm in self._by_sentence_text:
                rec = self._by_sentence_text[stext_norm]

        if rec is None:
            return [], []

        def _items_to_dicts(arr) -> List[Dict]:
            out: List[Dict] = []
            for item in (arr or []):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("skill_span", "")).strip()
                if not text:
                    continue
                out.append({"text": text, "confidence": SKILL_LLM_CONFIDENCE})
            return out

        return _items_to_dicts(rec.get("SKILL")), _items_to_dicts(rec.get("KNOWLEDGE"))

    def coverage(
        self,
        expected_sentence_ids: List[str],
    ) -> Dict[str, float]:
        """Report what fraction of expected sentences have a record in this file.

        Useful for diagnosing 'why is my Layer 1 output sparse?': if coverage
        is < 1.0, the Kaggle batch run did not process every sentence (likely
        because the input JSONL was filtered or truncated).
        """
        if not expected_sentence_ids:
            return {"expected": 0, "present": 0, "missing": 0, "coverage": 0.0}
        present = sum(1 for sid in expected_sentence_ids if sid in self._by_sentence_id)
        missing = len(expected_sentence_ids) - present
        return {
            "expected": len(expected_sentence_ids),
            "present": present,
            "missing": missing,
            "coverage": present / len(expected_sentence_ids),
        }

    def __len__(self) -> int:
        return len(self._by_sentence_id) or len(self._by_sentence_text)
