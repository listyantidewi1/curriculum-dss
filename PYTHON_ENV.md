# Python environment

The project uses **`.venv-ml/`** on the D drive as its primary Python environment for the v2 pipeline, dashboards, and smoke tests. Created 2026-05-12 to relocate Python's ~9 GB of heavy ML packages off the C drive.

## Which Python to use

| Use case | Command |
|---|---|
| **Run any pipeline script, smoke test, or the dashboard** | `D:\Projects\skill-extraction\.venv-ml\Scripts\python.exe` |
| **Activate the venv interactively** | `D:\Projects\skill-extraction\.venv-ml\Scripts\Activate.ps1` |
| **Run pip / pip install / pip freeze** | `D:\Projects\skill-extraction\.venv-ml\Scripts\pip.exe` |

After activation, the prompt prefix changes to `(.venv-ml)` and `python` / `pip` resolve to the D-side binaries automatically.

## Why not the system Python at `C:\Python313\python.exe`?

That system install puts user-scope packages in `C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages\`. Every `pip install` for an ML project there adds 100s of MB to the C drive even though source code lives on D. The C drive hit 97.7% full on 2026-05-12 from this exact pattern.

The D-side venv keeps **every byte of every installed package** on D, including:
- `torch` (CPU build, ~200 MB — sufficient for SBERT clustering + KKNI labelling. GPU work runs on Kaggle.)
- `transformers`, `sentence-transformers`, `scikit-learn`, `scipy`, `hdbscan`
- `streamlit`, `pandas`, `numpy`, `matplotlib`, `langdetect`, `pypdf`, `openai`

If you later need GPU torch inside this venv (e.g. for local Skill-LLM inference once a bigger GPU is available), uninstall and reinstall the CUDA build:
```powershell
D:\Projects\skill-extraction\.venv-ml\Scripts\pip.exe uninstall torch
D:\Projects\skill-extraction\.venv-ml\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cu118
```

## HuggingFace model cache

Models downloaded by `transformers` / `sentence-transformers` land at `D:\hf_cache\` via the `HF_HOME` user environment variable (set 2026-05-12). New PowerShell sessions inherit this automatically.

To verify it's still pointing at D:
```powershell
echo $env:HF_HOME    # should print D:\hf_cache
```

## How to install new packages

Always use the venv's pip — never `pip install` from a system shell that hasn't activated the venv:

```powershell
# Option A: activated session
D:\Projects\skill-extraction\.venv-ml\Scripts\Activate.ps1
pip install <package>

# Option B: explicit path
D:\Projects\skill-extraction\.venv-ml\Scripts\pip.exe install <package>
```

If you accidentally pip-install in a non-venv shell, the package goes to user-scope on C again. Check `pip --version` before installing — its path should contain `\.venv-ml\`.

## Quick smoke

```powershell
D:\Projects\skill-extraction\.venv-ml\Scripts\python.exe -c "import torch, sentence_transformers; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available()); print('sbert', sentence_transformers.__version__)"
```

## Cross-references

- `memory/project_v2_pipeline_complete.md` — what's in the v2 pipeline
- `memory/project_llm_provider_routing.md` — Jatevo vs OpenRouter dispatch (orthogonal to the venv)
- `dashboard_v2/README.md` — how to launch the UI
