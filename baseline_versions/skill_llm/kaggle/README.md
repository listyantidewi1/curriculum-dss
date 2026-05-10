# Running Skill-LLM on Kaggle (free GPU)

Local 4 GB VRAM is too small for LLaMA 3.1 8B at 4-bit. Kaggle gives every
account **30 hours/week of free P100 (16 GB) or T4 ×2 (16 GB)** GPU time,
which is more than enough for the ~1.5–2.5 hour training run. This guide
walks you through end-to-end on Kaggle in ~10 minutes of setup.

## Prerequisites

- Kaggle account: https://www.kaggle.com (free signup; phone-verify your
  account so the GPU accelerator unlocks).
- HuggingFace account with **approved access** to
  https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct.
  Verify by visiting the page logged in — if you see the file tree (no
  "Agree and access" banner), you're approved.
- HuggingFace access token: https://huggingface.co/settings/tokens
  Create a fine-grained token with **Read access to public gated repos**
  permission. Copy the `hf_...` string.

## Setup steps

### 1. Create the Kaggle notebook

1. Go to https://www.kaggle.com/code → **New Notebook**.
2. Top-right gear icon → **Accelerator** → choose **GPU T4 ×2** or **GPU P100**.
   T4 ×2 is the default; either works fine. (We only use one GPU; the
   ×2 just means there are two available.)
3. Top-right gear icon → **Persistence** → leave at "No persistence"
   (output is saved when you commit the notebook anyway).

### 2. Add `HF_TOKEN` as a Kaggle Secret

1. Open the notebook → right sidebar → **Add-ons** → **Secrets**.
2. Click **Add a new secret**.
3. Label: `HF_TOKEN`
4. Value: your `hf_...` token
5. **Toggle the secret on for this notebook** (mandatory — secrets are
   off by default).

### 3. Upload SkillSpan as notebook input

The script expects `train.json`, `dev.json`, `test.json` from this
project's `DATA/` folder. Two ways:

**Option A — quick & dirty (recommended for first run):**

1. Right sidebar → **Input** → **+ Upload Data**.
2. Drag-drop your local `DATA/train.json`, `DATA/dev.json`, `DATA/test.json`.
3. Kaggle will auto-create a temporary dataset for you.
4. Look at the path Kaggle gave you — it'll be something like
   `/kaggle/input/abc123/train.json`. Note the parent path
   (`/kaggle/input/abc123/`).

**Option B — proper Kaggle Dataset (recommended if you'll iterate):**

1. https://www.kaggle.com/datasets → **+ New Dataset**.
2. Title: `skillspan` (the path will become `/kaggle/input/skillspan/`).
3. Upload the three JSON files. Visibility: Private.
4. In your notebook → right sidebar → **Input** → **+ Add Input** →
   search "skillspan" → add the dataset.

### 4. Adjust `DATA_DIR` in the script

Open `run_on_kaggle.py` from this directory and find this line near the
top:

```python
DATA_DIR = Path("/kaggle/input/skillspan")  # change if your dataset name differs
```

Change to whatever path Kaggle gave you in step 3. For Option A above
that's something like `/kaggle/input/abc123`; for Option B it stays
`/kaggle/input/skillspan`.

### 5. Paste & run

1. In the Kaggle notebook, create a single code cell.
2. Paste the **entire contents** of `run_on_kaggle.py` into that cell.
3. Click **Run All** (or `Ctrl+F9`).

That's it. The script runs the three stages back-to-back:

```
Step 1/3: prepare data        (~10 sec)
Step 2/3: train LoRA          (~1.5h on P100, ~2.5h on T4)
Step 3/3: evaluate on dev+test (~30 min)
```

You'll see progress bars and per-epoch metrics. Total time is roughly
2–3 hours; the notebook stays alive as long as your browser tab is open.
If you close the tab the kernel keeps running for ~12 hours.

### 6. Download outputs

When `Step 3/3` finishes, you'll see a "DONE" banner with paths.

**To get the LoRA adapter** (so we can wire it into `pipeline.py`):

1. Right sidebar → **Output** → expand `/kaggle/working/skill_llm/lora_adapter/`.
2. Right-click each file → **Download**. You need:
   - `adapter_model.safetensors` (~50 MB — the trained LoRA weights)
   - `adapter_config.json`
   - `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
     and any other tokenizer files
3. Save them locally to
   `D:\Projects\skill-extraction\baseline_versions\skill_llm\lora_adapter\`.

**To get the metrics** (the F1 numbers we want to interpret):

1. Right sidebar → **Output** → `/kaggle/working/skill_llm/outputs/trained/`.
2. Download `metrics_dev.txt` and `metrics_test.txt`.
3. Optionally also `raw_outputs_dev.jsonl` / `raw_outputs_test.jsonl`
   for post-mortem analysis.

**Easier alternative:** click **Save Version** at the top of the
notebook → "Save & Run All (Commit)". Kaggle reruns everything in the
background and saves the entire `/kaggle/working/` tree as a permanent
notebook output. You can then download as a zip from the Output panel
on the saved version page.

## Common issues

**"GatedRepoError"** during model load
→ Either your `HF_TOKEN` secret isn't toggled on for this notebook,
   or you haven't accepted the LLaMA 3.1 8B license. Re-check both.

**OOM during training**
→ Confirm Accelerator is set to GPU (not "None"). If it's GPU and still
   OOMs, reduce `BATCH_SIZE` from 4 to 2 in the script (re-run from
   step 5).

**"Could not read HF_TOKEN from Kaggle secrets"**
→ Re-toggle the `HF_TOKEN` secret on for the notebook
   (Add-ons → Secrets → click the toggle next to HF_TOKEN).
   You may need to restart the kernel afterwards (Kernel → Restart).

**Training loss goes to NaN**
→ T4 sometimes has bf16 instability with bnb 4-bit. Switch
   `BNB_4BIT_COMPUTE_DTYPE = "float16"` and re-run. Or use the P100
   accelerator (top-right gear icon → Accelerator).

**Notebook disconnects mid-training**
→ Kaggle kernels stay alive for ~12 hours even with the browser
   closed. If it disconnects, top-right → "Kernel run" should show
   the running session and let you reattach. The
   `outputs/checkpoints/` directory has per-epoch checkpoints; you can
   resume from the last one if needed (extra plumbing required).

## After you have the LoRA adapter locally

1. Copy the downloaded `lora_adapter/` files to
   `baseline_versions/skill_llm/lora_adapter/` in this repo.
2. Run local eval to confirm the numbers reproduce
   (will use your local CPU/GPU; takes longer but works):

   ```powershell
   python baseline_versions/skill_llm/eval.py
   ```

3. Send me the `metrics_test.txt` content. I'll interpret it against
   the matrix in `../jjzha_replicate/REPLICATION_REPORT.md` and let you
   know if the model passes the verb-fidelity smoke test (and is
   therefore ready to integrate into `pipeline.py`).
