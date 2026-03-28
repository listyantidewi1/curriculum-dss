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
echo  STAGE A: Automated Extraction and Analysis  (steps 1-12)
echo ===========================================================
echo.


echo.
echo [9/16] Generating curriculum recommendations (with ablation + sensitivity)...
python recommendations.py --ablation --sensitivity
if errorlevel 1 goto :error

echo.
echo [10/16] Exporting gold set for labeling...
python export_gold_set.py --seed %SEED%
if errorlevel 1 goto :error

echo.
echo [11/16] Exporting for expert review (jobs, skills, knowledge)...
python export_for_review.py
if errorlevel 1 goto :error

echo.
echo [12/16] Exporting competencies for expert review...
python export_competencies_for_review.py
if errorlevel 1 goto :error

echo.
echo ===========================================================
echo  Stage A complete. All extraction and analysis outputs ready (steps 1-12).
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
echo [13/16] Merging gold labels (if multi-reviewer UI used)...
python merge_gold_labels.py
if errorlevel 1 echo [WARN] Merge gold labels had issues (may be single labeler)

echo.
echo [14/16] Evaluating extraction quality (using gold labels)...
python evaluate_extraction.py
if errorlevel 1 echo [WARN] Extraction evaluation had issues (check gold labels)

echo.
echo [15/16] Evaluating future-domain mapping (using gold labels)...
python evaluate_future_mapping.py
if errorlevel 1 echo [WARN] Future mapping evaluation had issues (check gold labels)

echo.
echo [16/16] Generating scientific analysis plots...
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
echo    results/extraction_evaluation_report.json
echo    results/future_mapping_evaluation_report.json
echo.
echo  Expert review feedback saved to:
echo    feedback_store/skill_feedback.csv
echo    feedback_store/knowledge_feedback.csv
echo    feedback_store/competency_feedback.csv
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
