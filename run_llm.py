import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
from transformers import Qwen3VLMoeForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor, MllamaForConditionalGeneration, LlavaNextForConditionalGeneration, LlavaNextProcessor, Gemma3ForConditionalGeneration
from qwen_vl_utils import process_vision_info
from utils import load_json, save_json, get_save_path, parse_llm_json
from sam_adapter import Sam2Adapter
import re, json

try:
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
except Exception:
    JsonSchemaParser = None
    build_transformers_prefix_allowed_tokens_fn = None

# a single JSON schema the LLM must emit
def build_anomaly_json_schema() -> dict:
    # normalized in [0,1]
    coord = {"type": "number", "minimum": 0.0, "maximum": 1.0}
    point = {
        "type": "object",
        "properties": {"x": coord, "y": coord},
        "required": ["x", "y"],
        "additionalProperties": False
    }
    bbox = {
        "type": "array",
        "items": coord,
        "minItems": 4, "maxItems": 4
    }
    region = {
        "type": "object",
        "properties": {
            "points_positive": {"type": "array", "items": point, "minItems": 1},
            "points_negative": {"type": "array", "items": point},
            "bbox": bbox
        },
        "required": ["points_positive", "bbox"],
        "additionalProperties": False
    }
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["normal", "anomalous"]},
            "regions": {"type": "array", "items": region},
            "confidence": {"type": "number"}
        },
        "required": ["label"],
        "additionalProperties": False
    }

# build a single, universal message list for all backends
def build_messages_for_images(imgs, instruction_text: str, system_text: str | None = None):
    """
    Universal multimodal messages for HF chat templates.
    - Qwen & LLaVA happily accept images in the user message.
    - Llama/Gemma also work (they may have a system msg too).
    """
    messages = []
    if system_text:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": imgs[0]},
            {"type": "image", "image": imgs[1]},
            {"type": "text",  "text": instruction_text},
        ],
    })
    return messages

def get_qwen_output(
    model: Any,
    processor: Any,
    input_imgs: List[Image.Image],
    input_txt: str,
) -> List[str]:
    """
    Get output from a Qwen3-VL model for binary anomaly detection.

    Only the first two images in ``input_imgs`` (reference and query) are
    utilised.  The ``heatmap_mode`` argument is retained for API
    compatibility but ignored.  The function constructs a chat message
    accordingly and invokes the model.

    Args:
        model: Loaded Qwen3-VL model.
        processor: Qwen processor used to prepare inputs.
        input_imgs: List of PIL images (reference followed by query).
        input_txt: Prompt text instructing the model.

    Returns:
        List[str]: Decoded model outputs.
    """
    imgs = input_imgs[:2]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": imgs[0]},
                {"type": "image", "image": imgs[1]},
                {"type": "text", "text": input_txt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def get_llama_output(
    model: Any,
    processor: Any,
    input_imgs: List[Image.Image],
    input_txt: str,
) -> List[str]:
    imgs = input_imgs[:2]
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a strict classifier. Follow the guidance of the user's message strictly."
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": imgs[0]},
                {"type": "image", "image": imgs[1]},
                {"type": "text", "text": input_txt},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=imgs, padding=True, return_tensors="pt"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)
    pad_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
    generated_ids = model.generate(
        **inputs, max_new_tokens=500, pad_token_id=pad_id
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def get_llava_output(
    model: Any,
    processor: Any,
    input_imgs: List[Image.Image],
    input_txt: str,
) -> List[str]:
    imgs = input_imgs[:2]
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a strict classifier. Follow the guidance of the user's message strictly."
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": imgs[0]},
                {"type": "image", "image": imgs[1]},
                {"type": "text", "text": input_txt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)
    pad_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
    generated_ids = model.generate(
        **inputs, max_new_tokens=500, pad_token_id=pad_id
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def get_gemma_output(
    model: Any,
    processor: Any,
    input_imgs: List[Image.Image],
    input_txt: str,
) -> List[str]:
    imgs = input_imgs[:2]
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a strict classifier. Follow the guidance of the user's message strictly."
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": imgs[0]},
                {"type": "image", "image": imgs[1]},
                {"type": "text", "text": input_txt},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=imgs,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)
    pad_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=500,
        pad_token_id=pad_id,
        temperature=0.0,
        do_sample=False,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def get_dataset_config(dataset: str) -> Tuple[Path, Path]:
    """
    Get configuration for a specific dataset.

    Args:
        dataset: Dataset name ('mvtec_ad', 'mvtec_ac', or 'visa_ac')

    Returns:
        Tuple[Path, Path]: Data directory and JSON file path

    Raises:
        ValueError: If the dataset is not supported
    """
    if dataset == 'mvtec_ad':
        data_dir = Path.cwd() / 'datasets' / 'mvtec_ad'
        json_file_path = Path.cwd() / 'configs' / 'prompts' / 'mvtec_ad_prompts.json'
    elif dataset == 'mvtec_ac':
        data_dir = Path.cwd() / 'datasets' / 'mvtec_ac'
        json_file_path = Path.cwd() / 'configs' / 'prompts' / 'mvtec_ac_prompts.json'
    elif dataset == 'visa_ac':
        data_dir = Path.cwd() / 'datasets' / 'visa_ac'
        json_file_path = Path.cwd() / 'configs' / 'prompts' / 'visa_ac_prompts.json'
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return data_dir, json_file_path

def generate_json_with_tools(
    model, processor, imgs, instruction_text: str, *,
    max_new_tokens: int = 400, temperature: float = 0.0
) -> tuple[dict, str]:
    """
    Enforce JSON by asking the model to call a tool whose parameters are the schema.
    Works on models whose chat template supports `tools`.
    Returns (parsed_json, raw_text).
    """
    schema = build_anomaly_json_schema()
    tools = [{
        "type": "function",
        "function": {
            "name": "report_anomaly",
            "description": (
                "Return the anomaly decision and localization strictly as JSON. "
                "Do not write narrative text."
            ),
            "parameters": schema,
        },
    }]

    # Strong system hint: always call the tool
    system_text = ("You must call the tool `report_anomaly` and return only valid arguments. "
                   "Do not output any text besides the tool call.")

    messages = build_messages_for_images(imgs, instruction_text, system_text=system_text)

    # Ask HF to use tool-use template if available
    chat = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, tools=tools
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[chat], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=temperature,
        no_repeat_ngram_size=6,
        repetition_penalty=1.15,
    )
    out_ids = gen_ids[:, inputs.input_ids.shape[1]:]
    raw = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Extract the tool arguments JSON. Pattern works across common templates.
    m = re.search(r'"arguments"\s*:\s*(\{.*?\})', raw, flags=re.S)
    json_str = m.group(1) if m else re.search(r'\{.*\}', raw, flags=re.S).group(0)
    return json.loads(json_str), raw

def generate_json_with_guidance(
    model, processor, imgs, instruction_text: str, *,
    max_new_tokens: int = 400
) -> tuple[dict, str]:
    """
    Enforce JSON with token-level constraints using LM-Format-Enforcer.
    Model-agnostic; works even when tool calling isn't available.
    Returns (parsed_json, raw_text).
    """
    if JsonSchemaParser is None or build_transformers_prefix_allowed_tokens_fn is None:
        raise RuntimeError("lm-format-enforcer is not installed. `pip install lm-format-enforcer`")

    schema = build_anomaly_json_schema()
    parser = JsonSchemaParser(schema)
    messages = build_messages_for_images(imgs, instruction_text)

    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[chat], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)

    # Build the constrained decoding hook
    prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(
        processor.tokenizer, parser
    )

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    )
    out_ids = gen_ids[:, inputs.input_ids.shape[1]:]
    raw = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # The output should already be valid JSON; still guard with a simple first-object extract
    m = re.search(r'\{.*\}', raw, flags=re.S)
    return json.loads(m.group(0)), raw

def run_llm(
    model_type: str,
    prompts_dict: Dict[str, Dict[str, Any]],
    data_dir: Path,
    num_ref: int,
    dataset: str,
    model: Optional[Any] = None,
    processor: Optional[Any] = None,
    *,
    task: str = "localize",
    sam2_repo: str = "facebook/sam2.1-hiera-large",
    masks_dir: Optional[str] = None,
    overlays_dir: Optional[str] = None,
    multimask: bool = False,
    max_points: int = 10,
    json_enforce: str = "guided"
) -> Dict[str, str]:

    """
    Run a multimodal LLM for binary anomaly detection.

    For each entry in ``prompts_dict`` this function loads a reference
    image from the dataset (assumed to be in ``test/good``) and the query
    image specified in the prompts dictionary.  It resizes both images to
    ``image_size`` and passes them to the selected model backend along with
    the prompt text.  The backend should return either ``anomalous`` or
    ``normal``.

    Args:
        model_type: One of 'qwen', 'llama', 'llava' or 'gemma'.
        prompts_dict: Mapping from sample keys to dictionaries containing
            'image' (path to the query image) and 'text' (prompt text).
        data_dir: Root of the dataset (e.g. datasets/mvtec_ad).
        image_size: Side length to resize images to before inference.
        num_ref: Number of reference images to use.
        dataset: Dataset name ('mvtec_ad', 'mvtec_ac', 'visa_ac').
        model: Model instance for HF backends.
        processor: Processor associated with the HF model.

    Returns:
        Dict[str, str]: Dictionary mapping sample keys to raw model responses.
    """
    predictions: Dict[str, str] = {}
    # SAM-2 (HF) initialization and output dirs 
    masks_root = overlays_root = None
    if task == 'localize':
        sam_adapter = Sam2Adapter(repo_id=sam2_repo)
        masks_root = Path(masks_dir or f"configs/masks/{dataset}/{model_type}")
        overlays_root = Path(overlays_dir or f"configs/overlays/{dataset}/{model_type}")
        masks_root.mkdir(parents=True, exist_ok=True)
        overlays_root.mkdir(parents=True, exist_ok=True)
    else:
        sam_adapter = None
    total_tokens = 0
    for key, value in tqdm(prompts_dict.items()):
        images: List[Any] = []
        # Extract the object category (everything before the first underscore)
        object_cat = key.split('_')[0]
        # Load reference image from test/good folder
        if num_ref > 0:
            # Determine the directory containing good images for this category
            if dataset in ('mvtec_ad', 'mvtec_ac'):
                ref_cat = 'metal_nut' if object_cat == 'metal' else object_cat
                ref_dir = data_dir / ref_cat / 'test' / 'good'
            else:  # visa_ac
                ref_cat = 'pipe_fryum' if object_cat == 'pipe' else object_cat
                ref_dir = data_dir / ref_cat / 'test' / 'good'
            ref_files = sorted(f for f in os.listdir(ref_dir) if f.lower().endswith(('.jpg','.png','.jpeg')))
            if not ref_files:
                raise FileNotFoundError(f"No valid reference images found in {ref_dir}")
            ref_path = ref_dir / ref_files[0]
            ref_img = Image.open(ref_path).convert('RGB')
            images.append(ref_img)
        # Load query image from prompts_dict
        query_path = value['image']
        query_img = Image.open(query_path).convert('RGB')
        images.append(query_img)
        # Retrieve prompt text
        text = value['text']
        # Enforce-JSON strategy selection
        obj = None
        raw_text = None
        try:
            if json_enforce == 'tools':
                obj, raw_text = generate_json_with_tools(model, processor, images[:2], text)
            elif json_enforce == 'guided':
                obj, raw_text = generate_json_with_guidance(model, processor, images[:2], text)
            else:
                # fall back to legacy free-form + repair
                out = (
                    get_qwen_output if model_type=='qwen' else
                    get_llama_output if model_type=='llama' else
                    get_llava_output if model_type=='llava' else
                    get_gemma_output
                )(model, processor, images, text)
                raw_text = out[0]
                obj = parse_llm_json(raw_text)
        except Exception as e:
            print(f"[warn] {key}: structured decode failed ({e}); trying legacy parse.")
            if raw_text is None:
                out = (
                    get_qwen_output if model_type=='qwen' else
                    get_llama_output if model_type=='llama' else
                    get_llava_output if model_type=='llava' else
                    get_gemma_output
                )(model, processor, images, text)
                raw_text = out[0]
            try:
                obj = parse_llm_json(raw_text)
            except Exception as e2:
                print(f"[warn] {key}: JSON parse failed; marking as normal and continuing: {e2}")
                predictions[key] = {"label": "normal", "_parse_error": str(e2)}
                continue

        print(f"{key}: {raw_text}")

        # SAM-2 localization
        sam_adapter.set_image(query_img)
        H, W = query_img.height, query_img.width

        region_masks = []
        region_scores = []  # soft score maps (for AUROC/AUPRO)
        region_paths = []
        for r_idx, region in enumerate(obj.get('regions', [])):
            pos = region.get('points_positive', [])[:max_points]
            neg = region.get('points_negative', [])[:max_points]
            pos_px = np.array([[p['x']*W, p['y']*H] for p in pos], dtype=np.float32)
            neg_px = np.array([[p['x']*W, p['y']*H] for p in neg], dtype=np.float32) if neg else None

            box_px = None
            def _coverage(mask_u8, pts):
                if pts is None or len(pts) == 0: 
                    return 0.0
                h, w = mask_u8.shape
                ii = np.clip(np.round(pts[:, 1]).astype(int), 0, h-1)
                jj = np.clip(np.round(pts[:, 0]).astype(int), 0, w-1)
                return float(mask_u8[ii, jj].mean()) / 255.0

            # try as-is (x,y)
            masks, scores, _logits = sam_adapter.predict_region(
                pos_pts_px=pos_px, neg_pts_px=neg_px, box_px=box_px, multimask_output=multimask
            )
            k = int(np.argmax(scores))
            mask_u8_xy = (masks[k].astype(np.uint8) * 255)

            # try swapped (y,x) in case LLM returned row/col
            pos_yx = pos_px[:, [1, 0]]
            neg_yx = neg_px[:, [1, 0]] if neg_px is not None else None
            masks2, scores2, _logits2 = sam_adapter.predict_region(
                pos_pts_px=pos_yx, neg_pts_px=neg_yx, box_px=box_px, multimask_output=multimask
            )
            k2 = int(np.argmax(scores2))
            mask_u8_yx = (masks2[k2].astype(np.uint8) * 255)

            # choose orientation by positive-vs-negative coverage
            def _score(mask, p, n):
                return _coverage(mask, p) - 0.5*_coverage(mask, n)

            score_xy = _score(mask_u8_xy, pos_px,  neg_px)
            score_yx = _score(mask_u8_yx, pos_yx,  neg_yx)
            use_yx   = score_yx > score_xy
            mask_u8  = mask_u8_yx if use_yx else mask_u8_xy
            pos_best = pos_yx if use_yx else pos_px
            neg_best = neg_yx if use_yx else neg_px

            # one-time polarity flip if negatives are covered more than positives
            if _coverage(mask_u8, pos_best) < _coverage(mask_u8, neg_best):
                masks3, scores3, _logits3 = sam_adapter.predict_region(
                    pos_pts_px=neg_best if neg_best is not None else np.empty((0,2), np.float32),
                    neg_pts_px=pos_best,
                    box_px=box_px,
                    multimask_output=multimask
                )
                mask_u8 = (masks3[int(np.argmax(scores3))].astype(np.uint8) * 255)

            # optional box from positives; keeps your original behavior
            if (pos_best is not None) and len(pos_best) >= 2 and box_px is None:
                x0, y0 = pos_best.min(axis=0); x1, y1 = pos_best.max(axis=0)
                pad = 0.05 * max(W, H)
                box_px = np.array([max(x0-pad,0), max(y0-pad,0), min(x1+pad,W-1), min(y1+pad,H-1)], dtype=np.float32)

            if 'bbox' in region:
                x0, y0, x1, y1 = region['bbox']
                box_px = np.array([x0*W, y0*H, x1*W, y1*H], dtype=np.float32)

            # final predict using the resolved polarity/orientation/box
            masks_final, scores_final, logits_final = sam_adapter.predict_region(
                pos_pts_px=pos_best if pos_best is not None else np.empty((0,2), np.float32),
                neg_pts_px=neg_best,
                box_px=box_px,
                multimask_output=multimask,
            )
            kf = int(np.argmax(scores_final))
            rmask = (masks_final[kf].astype(np.uint8) * 255)

            # build a FLOAT soft score map from logits for AUROC/AUPRO
            lg = np.array(logits_final[kf])
            if lg.ndim > 2:
                lg = lg.squeeze()
            prob = 1.0 / (1.0 + np.exp(-lg.astype(np.float32)))  # sigmoid on logits (float)
            if prob.shape != (H, W):
                # resize in FLOAT, never quantize before resize
                prob_img = Image.fromarray(prob.astype(np.float32), mode='F')
                prob_img = prob_img.resize((W, H), resample=Image.BILINEAR)
                prob = np.asarray(prob_img, dtype=np.float32)

            rpath = masks_root / f"{key}__r{r_idx}.png"
            Sam2Adapter.save_mask(rmask, rpath, hw_expected=(H, W))
            region_masks.append(rmask)
            region_paths.append(str(rpath))
            region_scores.append(prob)  # AUROC/AUPRO soft map for this region


        final_mask = Sam2Adapter.union_masks(region_masks) if region_masks else np.zeros((H, W), np.uint8)
        # one-shot correction if region area > 15% (total failure guard)
        area_ratio = float((final_mask > 0).sum()) / float(H * W)
        if obj.get("label") == "anomalous" and area_ratio > 0.15 and not obj.get("_retry_done", False):
            strict_text = text + (
                "\n\nOne-shot correction:\n"
                "- Your last region covered more than 15% of the image, which violates the small-defect rule.\n"
                "- Re-analyze Image B and RETURN A NEW JSON with a much smaller, tighter region around the most salient defect.\n"
                "- Keep coordinates normalized [0,1], top-left origin, y down. Use 3-6 positive points inside the smallest visible defect and 2-4 negatives tightly around it."
            )
            try:
                # regenerate JSON with the same backend & stricter instruction
                if json_enforce == 'tools':
                    obj, _ = generate_json_with_tools(model, processor, [ref_img, query_img], strict_text)
                elif json_enforce == 'guided':
                    obj, _ = generate_json_with_guidance(model, processor, [ref_img, query_img], strict_text)
                else:
                    out = (
                        get_qwen_output if model_type=='qwen' else
                        get_llama_output if model_type=='llama' else
                        get_llava_output if model_type=='llava' else
                        get_gemma_output
                    )(model, processor, [ref_img, query_img], strict_text)
                    obj = parse_llm_json(out[0])
                obj["_retry_done"] = True  # avoid loops

                # re-run the same region-processing code for obj['regions'] ---
                # clear and redo
                region_masks, region_scores, region_paths = [], [], []
                # re-enter the region loop 
                for r_idx, region in enumerate(obj.get('regions', [])):
                    pos = region.get('points_positive', [])[:max_points]
                    neg = region.get('points_negative', [])[:max_points]
                    pos_px = np.array([[p['x']*W, p['y']*H] for p in pos], dtype=np.float32)
                    neg_px = np.array([[p['x']*W, p['y']*H] for p in neg], dtype=np.float32) if neg else None

                    box_px = None
                    def _coverage(mask_u8, pts):
                        if pts is None or len(pts) == 0: 
                            return 0.0
                        h, w = mask_u8.shape
                        ii = np.clip(np.round(pts[:, 1]).astype(int), 0, h-1)
                        jj = np.clip(np.round(pts[:, 0]).astype(int), 0, w-1)
                        return float(mask_u8[ii, jj].mean()) / 255.0

                    # try as-is (x,y)
                    masks, scores, _logits = sam_adapter.predict_region(
                        pos_pts_px=pos_px, neg_pts_px=neg_px, box_px=box_px, multimask_output=multimask
                    )
                    k = int(np.argmax(scores))
                    mask_u8_xy = (masks[k].astype(np.uint8) * 255)

                    # try swapped (y,x) in case LLM returned row/col
                    pos_yx = pos_px[:, [1, 0]]
                    neg_yx = neg_px[:, [1, 0]] if neg_px is not None else None
                    masks2, scores2, _logits2 = sam_adapter.predict_region(
                        pos_pts_px=pos_yx, neg_pts_px=neg_yx, box_px=box_px, multimask_output=multimask
                    )
                    k2 = int(np.argmax(scores2))
                    mask_u8_yx = (masks2[k2].astype(np.uint8) * 255)

                    # choose orientation by positive-vs-negative coverage
                    def _score(mask, p, n):
                        return _coverage(mask, p) - 0.5*_coverage(mask, n)

                    score_xy = _score(mask_u8_xy, pos_px,  neg_px)
                    score_yx = _score(mask_u8_yx, pos_yx,  neg_yx)
                    use_yx   = score_yx > score_xy
                    mask_u8  = mask_u8_yx if use_yx else mask_u8_xy
                    pos_best = pos_yx if use_yx else pos_px
                    neg_best = neg_yx if use_yx else neg_px

                    # one-time polarity flip if negatives are covered more than positives
                    if _coverage(mask_u8, pos_best) < _coverage(mask_u8, neg_best):
                        masks3, scores3, _logits3 = sam_adapter.predict_region(
                            pos_pts_px=neg_best if neg_best is not None else np.empty((0,2), np.float32),
                            neg_pts_px=pos_best,
                            box_px=box_px,
                            multimask_output=multimask
                        )
                        mask_u8 = (masks3[int(np.argmax(scores3))].astype(np.uint8) * 255)

                    # optional box from positives; keeps your original behavior
                    if (pos_best is not None) and len(pos_best) >= 2 and box_px is None:
                        x0, y0 = pos_best.min(axis=0); x1, y1 = pos_best.max(axis=0)
                        pad = 0.05 * max(W, H)
                        box_px = np.array([max(x0-pad,0), max(y0-pad,0), min(x1+pad,W-1), min(y1+pad,H-1)], dtype=np.float32)

                    if 'bbox' in region:
                        x0, y0, x1, y1 = region['bbox']
                        box_px = np.array([x0*W, y0*H, x1*W, y1*H], dtype=np.float32)

                    # final predict using the resolved polarity/orientation/box
                    masks_final, scores_final, logits_final = sam_adapter.predict_region(
                        pos_pts_px=pos_best if pos_best is not None else np.empty((0,2), np.float32),
                        neg_pts_px=neg_best,
                        box_px=box_px,
                        multimask_output=multimask,
                    )
                    kf = int(np.argmax(scores_final))
                    rmask = (masks_final[kf].astype(np.uint8) * 255)

                    # build a FLOAT soft score map from logits for AUROC/AUPRO
                    lg = np.array(logits_final[kf])
                    if lg.ndim > 2:
                        lg = lg.squeeze()
                    prob = 1.0 / (1.0 + np.exp(-lg.astype(np.float32)))  # sigmoid on logits (float)
                    if prob.shape != (H, W):
                        # resize in FLOAT, never quantize before resize
                        prob_img = Image.fromarray(prob.astype(np.float32), mode='F')
                        prob_img = prob_img.resize((W, H), resample=Image.BILINEAR)
                        prob = np.asarray(prob_img, dtype=np.float32)

                    rpath = masks_root / f"{key}__r{r_idx}.png"
                    Sam2Adapter.save_mask(rmask, rpath, hw_expected=(H, W))
                    region_masks.append(rmask)
                    region_paths.append(str(rpath))
                    region_scores.append(prob)  # AUROC/AUPRO soft map for this region

                final_mask = Sam2Adapter.union_masks(region_masks) if region_masks else np.zeros((H, W), np.uint8)
            except Exception as _:
                pass  # fall back to the original mask if the retry fails
        if region_scores:
            union_score = region_scores[0]
            for s in region_scores[1:]:
                union_score = np.maximum(union_score, s)
        else:
            union_score = np.zeros((H, W), dtype=np.float32)
        score_path = masks_root / f"{key}__score.npy"
        np.save(score_path, union_score.astype(np.float32))
        top1_path = masks_root / f"{key}.png"
        Sam2Adapter.save_mask(final_mask, top1_path, hw_expected=(H, W))
        Sam2Adapter.save_overlay(query_img, final_mask, overlays_root / f"{key}.png")

        obj['sam2'] = {"top1_path": str(top1_path), "score_path": str(score_path), "all_paths": region_paths}
        predictions[key] = obj
    
    return predictions


def main():
    """Entry point for running the binary anomaly detection pipeline."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Run a multimodal LLM for binary anomaly detection."
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['qwen', 'llama', 'llava', 'gemma'],
        default='qwen',
        help='Model backend to use.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['mvtec_ad', 'mvtec_ac', 'visa_ac'],
        default='mvtec_ac',
        help='Dataset to process.'
    )
    parser.add_argument(
        '--num_ref',
        type=int,
        default=1,
        help='Number of reference images to use (only the first is used).'
    )
    # +++ new flags
    parser.add_argument('--task', choices=['binary','localize'], default='localize')
    parser.add_argument('--sam2_repo', type=str, default='facebook/sam2.1-hiera-large')
    parser.add_argument('--masks_dir', type=str, default=None)
    parser.add_argument('--overlays_dir', type=str, default=None)
    parser.add_argument('--multimask', action='store_true', default=False)
    parser.add_argument('--max_points', type=int, default=10)
    parser.add_argument(
        '--json_enforce',
        choices=['none', 'tools', 'guided'],
        default='guided',
        help='Enforce JSON outputs via tool-calling or guided decoding.'
    )

    args = parser.parse_args()
    # Retrieve dataset directory and prompts file
    data_dir, json_file_path = get_dataset_config(args.dataset)
    # Load prompts
    prompts_dict = load_json(json_file_path)
    # Initialise backend model/processor
    if args.model == 'qwen':
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct", dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
    elif args.model == 'llama':
        hf_token = os.getenv("llama_access")
        model = MllamaForConditionalGeneration.from_pretrained(
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token
        )
        processor = AutoProcessor.from_pretrained(
            "meta-llama/Llama-3.2-11B-Vision-Instruct", token=hf_token
        )
    elif args.model == 'llava':
        hf_token = os.getenv("llama_access")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token
        )
        processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf", token=hf_token
        )
    elif args.model == 'gemma':
        hf_token = os.getenv("llama_access")
        model = Gemma3ForConditionalGeneration.from_pretrained(
            "google/gemma-3-12b-it",
            dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token
        )
        processor = AutoProcessor.from_pretrained(
            "google/gemma-3-12b-it", token=hf_token
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    # Execute inference
    preds = run_llm(
        model_type=args.model,
        prompts_dict=prompts_dict,
        data_dir=data_dir,
        num_ref=args.num_ref,
        dataset=args.dataset,
        model=model,
        processor=processor,
        task=args.task,
        sam2_repo=args.sam2_repo,
        masks_dir=args.masks_dir,
        overlays_dir=args.overlays_dir,
        multimask=args.multimask,
        max_points=args.max_points,
        json_enforce=args.json_enforce,
    )
    save_path = get_save_path(
        args.dataset,
        args.model,
    )
    save_json(preds, save_path)
    print(f"[✓] Predictions saved to: {save_path}")
    print("Done")

if __name__ == '__main__':
    main() 