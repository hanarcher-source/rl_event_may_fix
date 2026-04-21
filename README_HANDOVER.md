# XS combo-bucket factor pipeline — `gitlab_handover_repo`

This folder contains **read-only copies** of the Python and Slurm files needed to **re-run** the cross-sectional combo-bucket factor experiments (rolling LLM, train-1-month + frozen OOS + combo-cache, MLP baseline, Qwen32 v1/v1-nodate vs v2, metrics).

**Originals are unchanged** under `may_fix/scripts/` and `rl_remodeled/batch_script/`. Edit those canonical copies in development; treat this directory as a **snapshot** you can **`git init` + push to GitLab** as its own small repo (paths in batch files / `BASE_DIR` in copied scripts still reference `may_fix` until you change them).

**Path:** `/finance_ML/zhanghaohan/rl_remodeled/may_fix/gitlab_handover_repo`

---

## Layout

| Path | Contents |
|------|-----------|
| `scripts/` | Flat copies of all pipeline libraries and entrypoints (imports resolve when you `cd` here and run a script). |
| `batch/` | Example Slurm launchers (paths inside still point at `may_fix/scripts`; adjust `cd` if you move the bundle). |
| `figures/` | Optional diagram generator for the LoRA + bucket-head schematic. |

---

## How to run (from this bundle)

```bash
cd /finance_ML/zhanghaohan/rl_remodeled/may_fix/gitlab_handover_repo/scripts
conda activate /finance_ML/zhanghaohan/env_py310   # or mycondaenv per job
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Example: Qwen32 v2 rolling (same flags as batch file)
python3 -u roll_xs_bucket_factor_qwen32b_v2_roll.py \
  --start_train_month 2023-01 --n_rolls 12 --max_mem_gib 70 \
  --batch_size 1 --grad_accum 16
```

**Note:** `BASE_DIR` inside `roll_xs_bucket_factor_pipeline_v2.py` (and related modules) is still **`/finance_ML/zhanghaohan/rl_remodeled/may_fix`**. So **logs / results / mappings** continue to write under the main `may_fix` tree unless you edit the copy. That keeps one global results area; for a fully relocatable tree, search-replace `BASE_DIR` in the copied files only.

**Data paths** in `combo_xs_bucket_sft_lib.py` remain absolute paths under `RL_tune/stored_data/` — same on every machine that has that data.

---

## Script map — which file does what

### Data & labels (shared)

| File | Role |
|------|------|
| `combo_xs_bucket_sft_lib.py` | `load_data_xs`, `attach_combo_keys`, `build_combo_day_labels`, `pred_class_spread` — **all** LLM/MLP pipelines import this. |

### Prompts

| File | Role |
|------|------|
| `prompt_xs_bucket_v2.py` | **V2** user text: `build_user_prompt_v2(combo_key)` (Chinese analyst template, no calendar line). |

### Core pipelines (training + inference logic)

| File | Role |
|------|------|
| `roll_xs_bucket_factor_pipeline_v2.py` | **V2 prompt** rolling pipeline; `BACKENDS` includes mistral, qwen, qwen32, llama, mistral3_24b, …; LoRA + `LMWithBucketHead`. |
| `roll_xs_bucket_factor_pipeline_nodate.py` | **V1 no-date** prompt (`build_user_prompt` without calendar line in text); same head idea as v1 dated. |
| `roll_xs_bucket_factor_pipeline.py` | **V1 dated** prompt (includes `交易日期:` line); reference / older runs. |

### MLP baseline

| File | Role |
|------|------|
| `roll_xs_bucket_factor_mlp.py` | Rolling MLP: multi-hot combo + month cycle → trivial MLP → factor. |
| `roll_xs_bucket_factor_mlp_train1m_frozen_oos12.py` | MLP train one month + frozen OOS (optional combo_cache path in that script). |

### Train 1 month + frozen OOS + combo-cache (LLM)

| File | Role |
|------|------|
| `roll_xs_bucket_factor_v2_train1m_frozen_oos12.py` | **V2** prompt: train roll-0 only, then `combo_cache` or `per_month` OOS; imports `roll_xs_bucket_factor_pipeline_v2` as `v2`. |
| `roll_xs_bucket_factor_v2_infer_oos_combo_cache.py` | Standalone **v2** combo-cache inference from a saved checkpoint. |
| `roll_xs_bucket_factor_nodate_train1m_frozen_oos12.py` | **V1 no-date** prompt: same frozen + combo_cache pattern; imports `roll_xs_bucket_factor_pipeline_nodate` as `nd`. |

### Qwen2.5-32B entrypoints

| File | Role |
|------|------|
| `roll_xs_bucket_factor_qwen32b.py` | Thin wrapper: forces `--backend qwen32` into **dated** `roll_xs_bucket_factor_pipeline.py` (not the usual “nodate” ablation). |
| `roll_xs_bucket_factor_qwen32b_nodate_roll.py` | **Canonical** Qwen32 rolling with **nodate** pipeline (`_nodate_` in `run_id`). |
| `roll_xs_bucket_factor_qwen32b_v1_roll.py` | **Alias** → same as `qwen32b_nodate_roll.py` (“**v1 prompt**” = nodate only). |
| `roll_xs_bucket_factor_qwen32b_v2_roll.py` | **Alias** → `pipeline_v2` + `--backend qwen32` (“**v2 prompt**”). |
| `roll_xs_bucket_factor_qwen32b_nodate_train1m_frozen_oos12.py` | Wrapper → `nodate_train1m_frozen_oos12` with `--backend qwen32`. |

### Metrics (evaluation on `final_results.csv`)

| File | Role |
|------|------|
| `metrics_tail_rank_ic.py` | Mean daily **Spearman** on **bottom-N + top-N** by factor (ablation headline metric; default N=100). |
| `compute_daily_ic_factor_top_bottom_n.py` | Mean daily **Pearson** IC on top-N + bottom-N by factor (different from Spearman rank IC). |

### Figures (optional)

| File | Role |
|------|------|
| `figures/draw_lora_hybrid_arch.py` | Matplotlib → `lora_hybrid_arch.svg` / `.png` (run from `figures/` after `cd`). |

---

## Batch examples (`batch/`)

Slurm scripts are **copies**; they still `cd` into **`.../may_fix/scripts`**. If you relocate only the bundle, update the `cd` line in each `.sh`.

| Script | Purpose |
|--------|---------|
| `batch_roll_xs_factor_mistral7b_v2.sh` | Mistral-7B **v2** rolling 12. |
| `batch_roll_xs_factor_mistral7b_v2_train1m_frozen12.sh` | Mistral-7B v2 train-1m + frozen combo OOS. |
| `batch_roll_xs_factor_mlp.sh` | MLP rolling. |
| `batch_roll_xs_factor_mlp_train1m_frozen12.sh` | MLP train-1m frozen. |
| `batch_roll_xs_factor_qwen32b_nodate_roll12.sh` | Original Qwen32 **nodate** rolling (canonical). |
| `batch_roll_xs_factor_qwen32b_v1_roll12.sh` | Same as nodate rolling (**v1** alias). |
| `batch_roll_xs_factor_qwen32b_v2_roll12.sh` | Qwen32 **v2** rolling (apples-to-apples vs v1). |
| `batch_roll_xs_factor_qwen32b_nodate_train1m_frozen12.sh` | Qwen32 nodate train-1m + frozen combo OOS. |

---

## Naming cheat sheet

| You say | Pipeline module | Typical entry script |
|--------|-------------------|----------------------|
| **v1 dated** | `roll_xs_bucket_factor_pipeline.py` | `roll_xs_bucket_factor_qwen32b.py` (qwen32) or call `pipeline.py` with backends |
| **v1 nodate** | `roll_xs_bucket_factor_pipeline_nodate.py` | `roll_xs_bucket_factor_qwen32b_nodate_roll.py` or `*_v1_roll.py` |
| **v2** | `prompt_xs_bucket_v2` + `roll_xs_bucket_factor_pipeline_v2.py` | `roll_xs_bucket_factor_pipeline_v2.py --backend …` or `roll_xs_bucket_factor_qwen32b_v2_roll.py` |

---

## Longer narrative doc

Experiment diary (ablation story, paths, hyperparameters):  
`/finance_ML/zhanghaohan/rl_remodeled/may_fix/DIARY_XS_V2_ABLATION.md`  
(not duplicated here; open that file for the full write-up).

---

*Bundle created as file copies for handover; do not treat as the sole source of truth for ongoing development.*
