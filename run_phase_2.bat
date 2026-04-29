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
REM    Stage D  (step  17-18)  Weight sensitivity + holdout validation
REM    Stage E                 (Optional) Second-round competency review
REM
REM  Usage:
REM    run_phase_2.bat              Fresh run (overwrites existing outputs)
REM    run_phase_2.bat --resume     Skip steps whose output already exists
REM
REM  This script produces the final research outputs including
REM  all evaluation reports and scientific analysis plots.
REM ============================================================

cd /d D:\Projects\skill-extraction\

set SEED=42
set RESUME=0
if /i "%1"=="--resume" set RESUME=1

if "%RESUME%"=="1" (
    echo.
    echo [RESUME MODE] Skipping steps whose output already exists.
    echo.
)

echo.
echo ===========================================================
echo  STAGE A: Import and Apply Human Feedback  (steps 1-6)
echo ===========================================================
echo.

echo [1/18] Importing human feedback from feedback_store (includes recommendation IRR)...
if "%RESUME%"=="1" if exist feedback_store\human_verified_skills.csv (
    echo [SKIP] Step 1 already done ^(feedback_store\human_verified_skills.csv exists^)
    goto :step2
)
python import_feedback.py
if errorlevel 1 goto :error
:step2

echo.
echo [2/18] Applying Bloom and type corrections to skills...
if "%RESUME%"=="1" if exist results\advanced_skills_human_filtered.csv (
    echo [SKIP] Step 2 already done ^(results\advanced_skills_human_filtered.csv exists^)
    goto :step3
)
python apply_feedback.py
if errorlevel 1 goto :error
:step3

echo.
echo [3/18] Validating/calibrating scoring parameters (AUC, Brier, CV threshold)...
if "%RESUME%"=="1" if exist results\parameter_validation_report.json (
    echo [SKIP] Step 3 already done ^(results\parameter_validation_report.json exists^)
    goto :step4
)
python validate_parameters.py
if errorlevel 1 goto :error
:step4

echo.
echo [4/18] Re-verifying skills (using calibrated threshold if available)...
if "%RESUME%"=="1" if exist feedback_store\calibrated_threshold.json (
    if exist results\verified_skills.csv (
        echo [SKIP] Step 4 already done ^(calibrated_threshold.json + verified_skills.csv exist^)
        goto :step5
    )
)
python verify_skills.py
if errorlevel 1 goto :error
:step5

echo.
echo [5/18] Merging gold labels (if multi-reviewer labeling was done)...
python merge_gold_labels.py
if errorlevel 1 echo [WARN] Merge gold labels had issues (may be single labeler)

echo.
echo [6/18] Re-mapping skills to future domains (post-correction, with tier sensitivity)...
if "%RESUME%"=="1" if exist results\future_weight_tier_sensitivity.json (
    echo [SKIP] Step 6 already done ^(results\future_weight_tier_sensitivity.json exists^)
    goto :stageB
)
python future_weight_mapping.py --input_type skills
if errorlevel 1 goto :error
:stageB

echo.
echo ===========================================================
echo  STAGE B: Re-Generate Outputs With Human Input  (steps 7-11)
echo ===========================================================
echo.

echo [7/18] Re-generating competencies (comprehensive mode with corrections)...
if "%RESUME%"=="1" if exist results\competency_proposals.json (
    echo [SKIP] Step 7 already done ^(results\competency_proposals.json exists^)
    goto :step8
)
python generate_competencies.py --comprehensive
if errorlevel 1 goto :error
:step8

echo.
echo [8/18] Exporting new competencies for optional second-round review...
if "%RESUME%"=="1" if exist results\expert_review_competencies.csv (
    echo [SKIP] Step 8 already done ^(results\expert_review_competencies.csv exists^)
    goto :step9
)
python export_competencies_for_review.py
if errorlevel 1 goto :error
:step9

echo.
echo [9/18] Re-analyzing time trends (FDR-controlled, with stability + DW distribution)...
if "%RESUME%"=="1" if exist results\skill_time_trends.csv (
    echo [SKIP] Step 9 already done ^(results\skill_time_trends.csv exists^)
    goto :step9b
)
python skill_time_trend_analysis.py --only_hard --stability
if errorlevel 1 goto :error
:step9b

echo.
echo [9b] Re-computing per-skill KKNI education-level demand...
if "%RESUME%"=="1" if exist results\skill_education_demand.csv (
    echo [SKIP] Step 9b already done ^(results\skill_education_demand.csv exists^)
    goto :step10
)
python compute_education_demand.py --force
if errorlevel 1 echo [WARN] Education-demand recomputation had issues (non-fatal)
:step10

echo.
echo [10/18] Re-generating recommendations (ablation + evaluation + demand-only baseline + coverage ablation)...
if "%RESUME%"=="1" if exist results\recommendations.csv (
    echo [SKIP] Step 10 already done ^(results\recommendations.csv exists^)
    goto :step11
)
python recommendations.py --ablation --sensitivity --evaluate --baseline --coverage-ablation
if errorlevel 1 goto :error
:step11

echo.
echo [11/18] Re-generating plots with updated results...
if "%RESUME%"=="1" if exist results\figures\skills_knowledge_total_per_model.png (
    echo [SKIP] Step 11 already done ^(figures exist^)
    goto :stageC
)
python plot_generator.py
if errorlevel 1 echo [WARN] Plot generation had issues
:stageC

echo.
echo ===========================================================
echo  STAGE C: Evaluation and Scientific Analysis  (steps 12-16)
echo ===========================================================
echo.

echo [12/18] Evaluating extraction quality (RQ1 primary: LLM-only vs hybrid ablation + BERT contribution)...
if "%RESUME%"=="1" if exist results\extraction_evaluation_report.json (
    echo [SKIP] Step 12 already done ^(results\extraction_evaluation_report.json exists^)
    goto :step13
)
python evaluate_extraction.py --llmonly-labels-dir results\hybrid\DATA\labels
if errorlevel 1 echo [WARN] Extraction evaluation skipped or incomplete
:step13

echo.
echo [13/18] Evaluating future-domain mapping accuracy...
if "%RESUME%"=="1" if exist results\future_mapping_evaluation_report.json (
    echo [SKIP] Step 13 already done ^(results\future_mapping_evaluation_report.json exists^)
    goto :step14
)
python evaluate_future_mapping.py
if errorlevel 1 echo [WARN] Future mapping evaluation skipped or incomplete
:step14

echo.
echo [14/18] Evaluating competency generation quality...
if "%RESUME%"=="1" if exist results\competency_evaluation_report.json (
    echo [SKIP] Step 14 already done ^(results\competency_evaluation_report.json exists^)
    goto :step15
)
python evaluate_competency_generation.py
if errorlevel 1 echo [WARN] Competency evaluation skipped or incomplete
:step15

echo.
echo [15/18] Generating scientific analysis plots...
if "%RESUME%"=="1" if exist results\figures\scientific_calibration_curve.png (
    echo [SKIP] Step 15 already done ^(scientific_calibration_curve.png exists^)
    goto :step16
)
python plot_scientific_analysis.py
if errorlevel 1 echo [WARN] Scientific plots had issues (some inputs may be missing)
:step16

echo.
echo [16/18] Logging final run metadata...
if "%RESUME%"=="1" if exist results\run_metadata.json (
    echo [SKIP] Step 16 already done ^(results\run_metadata.json exists^)
    goto :stageD
)
python log_run_metadata.py --seed %SEED%
if errorlevel 1 goto :error
:stageD

echo.
echo ===========================================================
echo  STAGE D: Weight Sensitivity Analysis  (steps 17-18)
echo ===========================================================
echo.

echo [17/18] Running extraction weight sensitivity analysis...
if "%RESUME%"=="1" if exist results\weight_sensitivity_extraction_report.json (
    echo [SKIP] Step 17 already done ^(results\weight_sensitivity_extraction_report.json exists^)
    goto :step17b
)
python scripts/weight_sensitivity_extraction.py
if errorlevel 1 echo [WARN] Weight sensitivity analysis had issues
:step17b

echo.
echo [17b] Running longitudinal holdout validation (RQ3)...
if "%RESUME%"=="1" if exist results\holdout_validation_report.json (
    echo [SKIP] Step 17b already done ^(results\holdout_validation_report.json exists^)
    goto :step18
)
python skill_trend_holdout_validation.py
if errorlevel 1 echo [WARN] Trend holdout validation had issues (needs date-enriched skills file)
:step18

echo.
echo [18/18] Exporting updated recommendations for expert review (RQ5 IRR refresh)...
if "%RESUME%"=="1" if exist DATA\labels\recommendations_for_review.csv (
    echo [SKIP] Step 18 already done ^(DATA\labels\recommendations_for_review.csv exists^)
    goto :complete
)
python export_recommendations_for_review.py
if errorlevel 1 echo [WARN] Recommendation export had issues (non-fatal)
:complete

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
echo    - holdout_validation_report.json            (RQ3: trend direction accuracy + slope correlation)
echo    - coverage_ablation_report.json             (empirical justification for coverage weight = 0.0)
echo    - trend_stability_report.json               (trends: DW distribution + clean top-20)
echo    - run_metadata.json                         (full reproducibility incl. prompt hashes)
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
echo         To resume from where you left off: run_phase_2.bat --resume
pause
exit /b 1

:end
pause
