@echo off
REM ============================================================
REM  Phase 2: Post-Review Pipeline
REM
REM  Prerequisites:
REM    1. run.bat completed (Phase 1: job_scraping data, LLM-only knowledge, concurrent LLM)
REM    2. Expert review done (feedback saved in feedback_store/)
REM
REM  Flow:
REM    Stage A  (steps  1-6)   Import feedback, apply corrections
REM    Stage B  (steps  7-11)  Re-generate outputs with human input
REM    Stage C  (steps 12-16)  Evaluation and scientific analysis
REM    Stage D  (step  17)     Weight sensitivity analysis
REM    Stage E                 (Optional) Second-round competency review
REM
REM  This script produces the final research outputs including
REM  all evaluation reports and scientific analysis plots.
REM ============================================================

cd /d D:\Projects\skill-extraction\

set SEED=42

echo.
echo ===========================================================
echo  STAGE A: Import and Apply Human Feedback  (steps 1-6)
echo ===========================================================
echo.

echo [1/18] Importing human feedback from feedback_store (includes recommendation IRR)...
python import_feedback.py
if errorlevel 1 goto :error

echo.
echo [2/18] Applying Bloom and type corrections to skills...
python apply_feedback.py
if errorlevel 1 goto :error

echo.
echo [3/18] Validating/calibrating scoring parameters (AUC, Brier, CV threshold)...
python validate_parameters.py
if errorlevel 1 goto :error

echo.
echo [4/18] Re-verifying skills (using calibrated threshold if available)...
python verify_skills.py
if errorlevel 1 goto :error

echo.
echo [5/18] Merging gold labels (if multi-reviewer labeling was done)...
python merge_gold_labels.py
if errorlevel 1 echo [WARN] Merge gold labels had issues (may be single labeler)

echo.
echo [6/18] Re-mapping skills to future domains (post-correction, with tier sensitivity)...
python future_weight_mapping.py --input_type skills
if errorlevel 1 goto :error

echo.
echo ===========================================================
echo  STAGE B: Re-Generate Outputs With Human Input  (steps 7-11)
echo ===========================================================
echo.

echo [7/18] Re-generating competencies (comprehensive mode with corrections)...
python generate_competencies.py --comprehensive
if errorlevel 1 goto :error

echo.
echo [8/18] Exporting new competencies for optional second-round review...
python export_competencies_for_review.py
if errorlevel 1 goto :error

echo.
echo [9/18] Re-analyzing time trends (FDR-controlled, with stability + DW distribution)...
python skill_time_trend_analysis.py --only_hard --stability
if errorlevel 1 goto :error

echo.
echo [10/18] Re-generating recommendations (ablation + evaluation + demand-only baseline + coverage ablation)...
python recommendations.py --ablation --sensitivity --evaluate --baseline --coverage-ablation
if errorlevel 1 goto :error

echo.
echo [11/18] Re-generating plots with updated results...
python plot_generator.py
if errorlevel 1 echo [WARN] Plot generation had issues

echo.
echo ===========================================================
echo  STAGE C: Evaluation and Scientific Analysis  (steps 12-16)
echo ===========================================================
echo.

echo [12/18] Evaluating extraction quality (RQ1 primary: Hybrid vs LLM-only + BERT contribution)...
python evaluate_extraction.py --llmonly-labels-dir results\llm_only\DATA\labels
if errorlevel 1 echo [WARN] Extraction evaluation skipped or incomplete

echo.
echo [13/18] Evaluating future-domain mapping accuracy...
python evaluate_future_mapping.py
if errorlevel 1 echo [WARN] Future mapping evaluation skipped or incomplete

echo.
echo [14/18] Evaluating competency generation quality...
python evaluate_competency_generation.py
if errorlevel 1 echo [WARN] Competency evaluation skipped or incomplete

echo.
echo [15/18] Generating scientific analysis plots...
python plot_scientific_analysis.py
if errorlevel 1 echo [WARN] Scientific plots had issues (some inputs may be missing)

echo.
echo [16/18] Logging final run metadata...
python log_run_metadata.py --seed %SEED%
if errorlevel 1 goto :error

echo.
echo ===========================================================
echo  STAGE D: Weight Sensitivity Analysis  (step 17-18)
echo ===========================================================
echo.

echo [17/18] Running extraction weight sensitivity analysis...
python scripts/weight_sensitivity_extraction.py
if errorlevel 1 echo [WARN] Weight sensitivity analysis had issues

echo.
echo [17b] Running longitudinal holdout validation (RQ3)...
python skill_trend_holdout_validation.py
if errorlevel 1 echo [WARN] Trend holdout validation had issues (needs date-enriched skills file)

echo.
echo [18/18] Exporting updated recommendations for expert review (RQ5 IRR refresh)...
python export_recommendations_for_review.py
if errorlevel 1 echo [WARN] Recommendation export had issues (non-fatal)

echo.
echo ============================================================
echo  Phase 2 COMPLETE.  All final outputs are ready.
echo.
echo  Key evaluation reports (results/):
echo    - extraction_evaluation_report.json        (RQ1: Hybrid vs LLM-only + BERT contribution)
echo    - parameter_validation_report.json          (AUC, Brier, CV threshold)
echo    - future_mapping_evaluation_report.json     (domain accuracy, MRR)
echo    - future_weight_tier_sensitivity.json       (ordinal robustness for RQ4/RQ5)
echo    - competency_evaluation_report.json         (quality metrics)
echo    - recommendations_report.json               (top-N, ablation, P@N, baseline comparison)
echo    - recommendations_baseline.csv              (demand-only floor baseline)
echo    - weight_sensitivity_report.json            (recommendation weights)
echo    - weight_sensitivity_extraction_report.json (extraction weights)
echo    - trend_stability_report.json               (trends: DW distribution + clean top-20)
echo    - run_metadata.json                         (full reproducibility)
echo.
echo  Feedback artifacts (feedback_store/):
echo    - human_verified_skills.csv
echo    - calibrated_threshold.json
echo    - bloom_corrections.json, type_corrections.json
echo    - inter_rater_report.json
echo.
echo  Scientific plots (results/figures/):
echo    - scientific_extraction_precision_*.png
echo    - scientific_trend_volcano.png
echo    - scientific_calibration_curve.png
echo    - scientific_future_mapping.png
echo    - scientific_weight_sensitivity.png
echo ============================================================
echo.
echo  (Optional) To review updated competencies, run:
echo    uvicorn review_ui.app:app --reload
echo    Open http://127.0.0.1:8000/?reviewer_id=YOUR_NAME
echo.

set /p REVIEW_AGAIN="Launch review UI for second-round competency review? (y/N): "
if /i "%REVIEW_AGAIN%"=="y" (
    echo.
    echo  Starting Expert Review UI for second-round review...
    start "Expert Review UI" cmd /c "cd /d D:\Projects\skill-extraction && python -m uvicorn review_ui.app:app --port 8000"
    timeout /t 3 >nul
    start http://127.0.0.1:8000/?reviewer_id=reviewer1
    echo.
    echo  Press any key when second-round review is complete.
    pause >nul
    echo.
    echo  Shutting down Expert Review UI...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
        taskkill /PID %%a /F >nul 2>&1
    )
    echo.
    echo  Re-importing feedback from second-round review...
    python import_feedback.py
    echo  Re-evaluating competency quality...
    python evaluate_competency_generation.py
    echo  Updating metadata...
    python log_run_metadata.py --seed %SEED%
)

goto :end

:error
echo.
echo [ERROR] A step failed. Fix the error above and re-run.
pause
exit /b 1

:end
pause
