@echo off
REM ============================================================
REM  Phase 1: Full Pipeline — Extraction through Gold Labeling
REM
REM  Data source: job_scraping/output/english_jobs.csv (12 months)
REM  Step 1 runs: preprocess -> log_run_metadata -> pipeline
REM
REM  Flow:
REM    Stage A  (steps  1-12)  Automated extraction & analysis
REM    Stage B  (steps 13-16)  Gold labeling (interactive)
REM    Stage C                 Launch expert review UI
REM
REM  After completing Phase 1, run run_phase_2.bat for post-review.
REM
REM  For LARGER DATA: python run_with_job_scraping.py --sample_size 5000
REM  Process all rows:  python run_with_job_scraping.py --sample_size 0
REM  LLM concurrency: 5 parallel workers (use --llm_concurrency 1 for sequential)
REM ============================================================

cd /d D:\Projects\skill-extraction\

set SEED=42
set LLM_CONCURRENCY=5

echo.
echo ===========================================================
echo  STAGE A: Automated Extraction and Analysis  (steps 1-14)
echo ===========================================================
echo.

echo [1/18] Preprocess + log metadata + pipeline (hybrid BERT+LLM, %LLM_CONCURRENCY% concurrent LLM, dedup enabled)...
python run_with_job_scraping.py --seed %SEED% --llm_concurrency %LLM_CONCURRENCY% --dedupe
if errorlevel 1 goto :error

echo.
echo [2/18] RQ1 ablation: LLM-only extraction run (resolves context-window confound)...
python run_with_job_scraping.py --seed %SEED% --llm_concurrency %LLM_CONCURRENCY% --extraction-mode llm_only --dedupe --output_dir results\llm_only
if errorlevel 1 echo [WARN] LLM-only ablation run failed (non-fatal; RQ1 primary test will be skipped)

echo.
echo [3/18] Generating plots...
python plot_generator.py
if errorlevel 1 goto :error

echo.
echo [4/18] Verifying extracted skills...
python verify_skills.py
if errorlevel 1 goto :error

echo.
echo [5/18] Mapping knowledge to future job domains...
python future_weight_mapping.py
if errorlevel 1 goto :error

echo.
echo [6/18] Mapping skills to future job domains...
python future_weight_mapping.py --input_type skills
if errorlevel 1 goto :error

echo.
echo [7/18] Enriching outputs with job dates...
python enrich_with_dates.py
if errorlevel 1 goto :error

echo.
echo [8/18] Analyzing time trends (FDR-controlled, with stability + DW distribution)...
python skill_time_trend_analysis.py --only_hard --stability
if errorlevel 1 goto :error

echo.
echo [9/18] Generating initial competency proposals (with future + empirical trends)...
python generate_competencies.py
if errorlevel 1 goto :error

echo.
echo [10/18] Generating curriculum recommendations (with ablation + sensitivity + coverage ablation)...
python recommendations.py --ablation --sensitivity --coverage-ablation
if errorlevel 1 goto :error

echo.
echo [11/18] Exporting gold set for labeling...
python export_gold_set.py --seed %SEED%
if errorlevel 1 goto :error

echo.
echo [12/18] Exporting for expert review (jobs, skills, knowledge)...
python export_for_review.py
if errorlevel 1 goto :error

echo.
echo [13/18] Exporting competencies for expert review...
python export_competencies_for_review.py
if errorlevel 1 goto :error

echo.
echo [14/18] Exporting top-20 recommendations for expert priority labeling (RQ5 IRR)...
python export_recommendations_for_review.py
if errorlevel 1 echo [WARN] Recommendation export had issues (non-fatal)

echo.
echo ===========================================================
echo  Stage A complete. All extraction and analysis outputs ready (steps 1-14).
echo ===========================================================

echo.
echo ===========================================================
echo  STAGE B: Gold Set Labeling  (steps 13-16)
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
echo [16/18] Evaluating extraction quality (RQ1 primary: Hybrid vs LLM-only; secondary: vs chance)...
python evaluate_extraction.py --llmonly-labels-dir results\llm_only\DATA\labels
if errorlevel 1 echo [WARN] Extraction evaluation had issues (check gold labels)

echo.
echo [17/18] Evaluating future-domain mapping (using gold labels)...
python evaluate_future_mapping.py
if errorlevel 1 echo [WARN] Future mapping evaluation had issues (check gold labels)

echo.
echo [18/18] Generating scientific analysis plots...
python plot_scientific_analysis.py
if errorlevel 1 echo [WARN] Scientific plots had issues (some inputs may be missing)

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
echo  RQ1 ablation outputs (LLM-only run):
echo    results/llm_only/advanced_skills.csv
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
pause
exit /b 1

:end
pause
