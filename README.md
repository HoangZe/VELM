# Visual Anomaly Detection using Multimodal Large Models

This repository implements the code-based pipeline used in "Visual Anomaly Detection using Multimodal Large Models". It pairs a visual segmentor (SAM-2) with multimodal LLMs to detect and localize anomalies in industrial images.

What the code does:
- Generate category-specific structured prompts (reference + query + guidance) using `generate_prompts.py`.
- Query a multimodal HF model (`qwen`, `llama`, `llava`, `gemma`) with the images + prompt and request a strict JSON answer describing label + localization points/bbox.
- Convert LLM-provided points to pixel masks using a SAM-2 adapter (`sam_adapter.py`), save overlays and soft score maps, and optionally retry when predicted regions are implausibly large.
- Compute image-level and pixel-level evaluation metrics with `eval.py` using saved predictions and SAM outputs.

Repository layout (relevant files):

- `generate_prompts.py` — create prompts and save to `configs/prompts/{dataset}_prompts.json`.
- `run_llm.py` — main inference pipeline (HF multimodal models + SAM-2 integration).
- `eval.py` — image- and pixel-level evaluation utilities.
- `sam_adapter.py` — wrapper around SAM-2 to predict masks from points/boxes.
- `utils.py` — I/O, JSON parsing (robust LLM repairs), config helpers.
- `configs/` — prompts, predictions, evaluations, contour config.
- `datasets/` — expected dataset roots (not included).

Quick setup
-----------

1) Create a Python 3.9 environment (conda recommended) and install requirements:

```bash
conda create -n velm_env python=3.9 -y
conda activate velm_env
pip install -r requirements.txt
```

2) (Optional) Set a Hugging Face token if required by model checkpoints:

```bash
export llama_access=hf_xxx_your_token
```

3) Place datasets under `datasets/` (see the dataset section). Optionally populate `configs/*` with description JSONs.

Data layout expected
-------------------

For MVTec/VisA style datasets the code expects:

- Images: `datasets/<dataset>/<category>/test/<defect>/<image>.png`
- Ground-truth masks: `datasets/<dataset>/<category>/ground_truth/<defect>/<image>_mask.png`

Supported dataset keys used by scripts: `mvtec_ad`, `mvtec_ac`, `visa_ac`.

generate_prompts.py
-------------------

Purpose: iterate test folders and write a JSON mapping of sample keys to `{image, text}` where `text` is a strict instruction prompt used by the LLM. If `configs/*_des.json` exists it will be used to inject category-specific guidance.

Usage:

```bash
python generate_prompts.py --dataset mvtec_ac --descriptions_path configs/mvtec_ac_des.json
```

Outputs: `configs/prompts/{dataset}_prompts.json`.

run_llm.py
----------

Purpose: run a multimodal HF model per prompt, parse the structured JSON output, and call SAM-2 to generate masks/overlays and soft score maps.

Key details discovered in the code:
- Supported backends (`--model`): `qwen`, `llama`, `llava`, `gemma` (loaded via `transformers.from_pretrained`).
- JSON enforcement options (`--json_enforce`): `none`, `tools`, `guided`. `guided` uses `lm-format-enforcer` when available; `tools` requests a tool call from the model; `none` uses the legacy repair parser in `utils.parse_llm_json`.
- When `task=localize` (default), the pipeline uses `Sam2Adapter` to produce masks and score maps, saved under `configs/masks/{dataset}/{model}/` and `configs/overlays/{dataset}/{model}/`.
- If the unioned predicted mask covers >15% of the image (hard or soft), the pipeline issues a one-shot correction and re-queries the LLM with stricter instructions.

Typical command:

```bash
python run_llm.py --model qwen --dataset mvtec_ac --num_ref 1 --task localize --json_enforce guided
```

Outputs:

- `configs/predictions/{dataset}_binary_preds_{model}.json` — predictions JSON.
- `configs/masks/{dataset}/{model}/*` — saved masks and `*__score.npy` soft maps.
- `configs/overlays/{dataset}/{model}/*` — image overlays.

Notes:
- HF model downloads require disk space and possibly a HF token. Large models may need GPUs to run efficiently.
- If `lm-format-enforcer` is missing, guided decoding will raise; `run_llm.py` will fall back to legacy parsing where possible.

eval.py
-------

Purpose: compute image-level and pixel-level metrics from saved predictions and SAM outputs.

Capabilities:
- Image-level: accuracy, precision, recall, F1, AUROC (per-category and averaged).
- Pixel-level: AUROC, AUPRO@30% (per-image averaged), pixel-F1@0.5 and F1-max across thresholds.

Usage example:

```bash
python eval.py --dataset mvtec_ac --model qwen --eval_mode both --output configs/evaluations/mvtec_ac_qwen_eval.json
```

Developer notes
---------------

- `utils.parse_llm_json` implements robust repairs for common LLM output issues (fences, trailing commas, misplaced brackets, orphan numerics) and raises clear errors when JSON is invalid.
- `sam_adapter.py` wraps a SAM-2 HF predictor (`facebook/sam2.1-hiera-large` by default).
- `run_llm.py` tries multiple decoding/enforcement strategies and falls back to legacy parsing when necessary; it's designed to be defensive when interacting with LLMs.