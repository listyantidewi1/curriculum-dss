@echo off
REM ============================================================
REM  Phase 1: Full Pipeline — Extraction through Gold Labeling
REM
REM  Data source: job_scraping/output/english_jobs.csv (12 months)
REM  Step 1 runs: preprocess -> log_run_metadata -> pipeline
REM
REM  Flow:
REM    Stage A  (steps  1-14)  Automated extraction & analysis
REM    Stage B  (steps 15-18)  Gold labeling (interactive)
REM    Stage C                 Launch expert review UI
REM
REM  After completing Phase 1, run run_phase_2.bat for post-review.
REM
REM  Usage:
REM    run.bat              Fresh run (overwrites existing outputs)
REM    run.bat --resume     Skip steps whose output already exists
REM
REM  For LARGER DATA: python run_with_job_scraping.py --sample_size 5000
REM  Process all rows:  python run_with_job_scraping.py --sample_size 0
REM  LLM concurrency: 5 parallel workers (use --llm_concurrency 1 for sequential)
REM ============================================================

cd /d D:\Projects\skill-extraction\

set SEED=42
set LLM_CONCURRENCY=5
set SAMPLE_SIZE=200
REM ^-- jobs sampled per pipeline run. Bump to 5000+ for production. 0 = all rows.
set RESUME=0
if /i "%1"=="--resume" set RESUME=1

if "%RESUME%"=="1" (
    echo.
    echo [RESUME MODE] Skipping steps whose output already exists.
    echo.
)

echo.
echo ===========================================================
echo  STAGE A: Automated Extraction and Analysis  (steps 1-14)
echo ===========================================================
echo.

echo [1/18] Preprocess + log metadata + pipeline (LLM-only primary run, %LLM_CONCURRENCY% concurrent LLM, dedup enabled)...
if "%RESUME%"=="1" if exist results\advanced_skills.csv (
    echo [SKIP] Step 1 already done ^(results\advanced_skills.csv exists^)
    goto :step2
)
python run_with_job_scraping.py --seed %SEED% --llm_concurrency %LLM_CONCURRENCY% --sample_size %SAMPLE_SIZE% --dedupe
if errorlevel 1 goto :error
:step2

echo.
echo [2/18] RQ1 ablation: hybrid BERT+LLM extraction run (compares against LLM-only primary)...
if "%RESUME%"=="1" if exist results\hybrid\advanced_skills.csv (
    echo [SKIP] Step 2 already done ^(results\hybrid\advanced_skills.csv exists^)
    goto :step3
)
python run_with_job_scraping.py --seed %SEED% --llm_concurrency %LLM_CONCURRENCY% --sample_size %SAMPLE_SIZE% --extraction-mode hybrid --dedupe --output_dir results\hybrid
if errorlevel 1 echo [WARN] Hybrid ablation run failed (non-fatal; RQ1 hybrid comparison will be skipped)
:step3

echo.
echo [3/18] Generating plots...
if "%RESUME%"=="1" if exist results\figures\skills_knowledge_total_per_model.png (
    echo [SKIP] Step 3 already done ^(figures exist^)
    goto :step4
)
python plot_generator.py
if errorlevel 1 goto :error
:step4

echo.
echo [4/18] Verifying extracted skills...
if "%RESUME%"=="1" if exist results\verified_skills.csv (
    echo [SKIP] Step 4 already done ^(results\verified_skills.csv exists^)
    goto :step5
)
python verify_skills.py
if errorlevel 1 goto :error
:step5

echo.
echo [5/18] Mapping knowledge to future job domains...
if "%RESUME%"=="1" if exist results\future_skill_weights_dummy.csv (
    echo [SKIP] Step 5 already done ^(results\future_skill_weights_dummy.csv exists^)
    goto :step6
)
python future_weight_mapping.py
if errorlevel 1 goto :error
:step6

echo.
echo [6/18] Mapping skills to future job domains...
if "%RESUME%"=="1" if exist results\future_skill_weights.csv (
    echo [SKIP] Step 6 already done ^(results\future_skill_weights.csv exists^)
    goto :step7
)
python future_weight_mapping.py --input_type skills
if errorlevel 1 goto :error
:step7

echo.
echo [7/18] Enriching outputs with job dates...
if "%RESUME%"=="1" if exist results\advanced_skills_with_dates.csv (
    echo [SKIP] Step 7 already done ^(results\advanced_skills_with_dates.csv exists^)
    goto :step8
)
python enrich_with_dates.py
if errorlevel 1 goto :error
:step8

echo.
echo [8/18] Analyzing time trends (FDR-controlled, with stability + DW distribution)...
if "%RESUME%"=="1" if exist results\skill_time_trends.csv (
    echo [SKIP] Step 8 already done ^(results\skill_time_trends.csv exists^)
    goto :step8b
)
python skill_time_trend_analysis.py --only_hard --stability
if errorlevel 1 goto :error
:step8b

echo.
echo [8b] Computing per-skill KKNI education-level demand...
if "%RESUME%"=="1" if exist results\skill_education_demand.csv (
    echo [SKIP] Step 8b already done ^(results\skill_education_demand.csv exists^)
    goto :step9
)
python compute_education_demand.py
if errorlevel 1 echo [WARN] Education-demand computation had issues (non-fatal; KKNI faceting will be limited)
:step9

echo.
echo [9/18] Generating initial competency proposals (with future + empirical trends)...
if "%RESUME%"=="1" if exist results\competency_proposals.json (
    echo [SKIP] Step 9 already done ^(results\competency_proposals.json exists^)
    goto :step10
)
python generate_competencies.py
if errorlevel 1 goto :error
:step10

echo.
echo [10/18] Generating curriculum recommendations (with ablation + sensitivity + coverage ablation)...
if "%RESUME%"=="1" if exist results\recommendations.csv (
    echo [SKIP] Step 10 already done ^(results\recommendations.csv exists^)
    goto :step11
)
python recommendations.py --ablation --sensitivity --coverage-ablation
if errorlevel 1 goto :error
:step11

echo.
echo [11/18] Exporting gold set for labeling...
if "%RESUME%"=="1" if exist DATA\labels\gold_skills.csv (
    echo [SKIP] Step 11 already done ^(DATA\labels\gold_skills.csv exists^)
    goto :step12
)
python export_gold_set.py --seed %SEED%
if errorlevel 1 goto :error
:step12

echo.
echo [12/18] Exporting for expert review (jobs, skills, knowledge)...
if "%RESUME%"=="1" if exist results\expert_review_skills.csv (
    echo [SKIP] Step 12 already done ^(results\expert_review_skills.csv exists^)
    goto :step13
)
python export_for_review.py
if errorlevel 1 goto :error
:step13

echo.
echo [13/18] Exporting competencies for expert review...
if "%RESUME%"=="1" if exist results\expert_review_competencies.csv (
    echo [SKIP] Step 13 already done ^(results\expert_review_competencies.csv exists^)
    goto :step14
)
python export_competencies_for_review.py
if errorlevel 1 goto :error
:step14

echo.
echo [14/18] Exporting top-20 recommendations for expert priority labeling (RQ5 IRR)...
if "%RESUME%"=="1" if exist DATA\labels\recommendations_for_review.csv (
    echo [SKIP] Step 14 already done ^(DATA\labels\recommendations_for_review.csv exists^)
    goto :stageB
)
python export_recommendations_for_review.py
if errorlevel 1 echo [WARN] Recommendation export had issues (non-fatal)
:stageB

echo.
echo ===========================================================
echo  Stage A complete. All extraction and analysis outputs ready (steps 1-14).
echo ===========================================================

echo.
echo ===========================================================
echo  STAGE B: Gold Set Labeling  (steps 15-18)
echo ===========================================================
echo.
echo  The gold labeling UI will now open in a new window.
echo  Label all items (skills, knowledge, domain mapping), then
echo  come back to THIS window and press any key to continue.
echo.
echo  URL: http://127.0.0.1:8001/?labeler_id=YOUR_NAME
echo.

start "Gold Labeling UI" cmd /c "cd /d D:\Projects\skill-extraction && python -m uvicorn gold_labeling_ui.app:app --port 8001"

timeout /t 3 >nul
start http://127.0.0.1:8001/?labeler_id=labeler1

echo  Waiting for you to finish labeling...
echo  Press any key when labeling is complete.
pause >nul

echo.
echo  Shutting down Gold Labeling UI...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8001 ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
)

echo.
echo [15/18] Merging gold labels (if multi-reviewer UI used)...
python merge_gold_labels.py
if errorlevel 1 echo [WARN] Merge gold labels had issues (may be single labeler)

echo.
echo [16/18] Evaluating extraction quality (RQ1 primary: LLM-only vs hybrid ablation; secondary: vs chance)...
if "%RESUME%"=="1" if exist results\extraction_evaluation_report.json (
    echo [SKIP] Step 16 already done ^(results\extraction_evaluation_report.json exists^)
    goto :step17
)
python evaluate_extraction.py --llmonly-labels-dir results\hybrid\DATA\labels
if errorlevel 1 echo [WARN] Extraction evaluation had issues (check gold labels)
:step17

echo.
echo [17/18] Evaluating future-domain mapping (using gold labels)...
if "%RESUME%"=="1" if exist results\future_mapping_evaluation_report.json (
    echo [SKIP] Step 17 already done ^(results\future_mapping_evaluation_report.json exists^)
    goto :step18
)
python evaluate_future_mapping.py
if errorlevel 1 echo [WARN] Future mapping evaluation had issues (check gold labels)
:step18

echo.
echo [18/18] Generating scientific analysis plots...
if "%RESUME%"=="1" if exist results\figures\scientific_calibration_curve.png (
    echo [SKIP] Step 18 already done ^(scientific_calibration_curve.png exists^)
    goto :stageC
)
python plot_scientific_analysis.py
if errorlevel 1 echo [WARN] Scientific plots had issues (some inputs may be missing)
:stageC

echo.
echo ===========================================================
echo  STAGE C: Expert Review
echo ===========================================================
echo.
echo  The expert review UI will now open in a new window.
echo  Review skills, knowledge, and competencies, then close
echo  the review UI window when finished.
echo.
echo  URL: http://127.0.0.1:8000/?reviewer_id=YOUR_NAME
echo.

start "Expert Review UI" cmd /c "cd /d D:\Projects\skill-extraction && python -m uvicorn review_ui.app:app --port 8000"

timeout /t 3 >nul
start http://127.0.0.1:8000/?reviewer_id=reviewer1

echo  Waiting for you to finish reviewing...
echo  Press any key when expert review is complete.
pause >nul

echo.
echo  Shutting down Expert Review UI...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
)

echo.
echo ============================================================
echo  Phase 1 COMPLETE.
echo.
echo  Gold labeling results:
echo    DATA/labels/gold_skills.csv, gold_knowledge.csv
echo    results/extraction_evaluation_report.json  (includes RQ1: Hybrid vs LLM-only)
echo    results/future_mapping_evaluation_report.json
echo.
echo  RQ1 ablation outputs (hybrid BERT+LLM run):
echo    results/hybrid/advanced_skills.csv
echo    results/future_weight_tier_sensitivity.json  (ordinal robustness check)
echo.
echo  Expert review feedback saved to:
echo    feedback_store/skill_feedback.csv
echo    feedback_store/knowledge_feedback.csv
echo    feedback_store/competency_feedback.csv
echo.
echo  Recommendation review for RQ5 IRR:
echo    DATA/labels/recommendations_for_review.csv  (fill in priority_label per reviewer)
echo.
echo  Next: run  run_phase_2.bat  to process feedback and
echo        generate final outputs.
echo ============================================================
goto :end

:error
echo.
echo [ERROR] A step failed. Fix the error above and re-run.
echo         To resume from where you left off: run.bat --resume
pause
exit /b 1

:end
pause
