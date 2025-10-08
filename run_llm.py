import os
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
import torch
import numpy as np
from openai import OpenAI
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor, MllamaForConditionalGeneration, LlavaNextForConditionalGeneration, LlavaNextProcessor, Gemma3ForConditionalGeneration
from qwen_vl_utils import process_vision_info
from utils import load_json, save_json, get_save_path, parse_llm_json
from sam_adapter import Sam2Adapter


def encode_image(image: Image.Image) -> str:
    """
    Encode a PIL Image to base64 string.

    Args:
        image: PIL Image to encode

    Returns:
        str: Base64 encoded image string
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def get_gpt_output(
    client: OpenAI,
    images: List[str],
    text: str,
    model_name: str,
) -> Tuple[str, Dict[str, int]]:
    """
    Query an OpenAI GPT model with a pair of images and a text prompt.

    The GPT API uses a chat format where each message may contain multiple
    images.  For binary anomaly detection only the first two images in
    ``images`` are considered: the reference (good) image and the query
    image.  The ``heatmap_mode`` parameter is accepted for backwards
    compatibility but ignored.

    Args:
        client: OpenAI client instance.
        images: List of base64‑encoded images.  Only the first two entries
            are used.
        text: Prompt text instructing the model to respond ``anomalous``
            or ``normal``.
        model_name: Name of the GPT model variant to use.

    Returns:
        Tuple[str, Dict[str, int]]: The model's text response and token usage
            statistics.
    """
    # Ensure we have at least two images; ignore extras
    ref_img, query_img = images[:2]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_img}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{query_img}"}},
            ],
        }
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
    )
    return response.choices[0].message.content, response.usage


def get_qwen_output(
    model: Any,
    processor: Any,
    input_imgs: List[Image.Image],
    input_txt: str,
) -> List[str]:
    """
    Get output from a Qwen3‑VL model for binary anomaly detection.

    Only the first two images in ``input_imgs`` (reference and query) are
    utilised.  The ``heatmap_mode`` argument is retained for API
    compatibility but ignored.  The function constructs a chat message
    accordingly and invokes the model.

    Args:
        model: Loaded Qwen3‑VL model.
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
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
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
        **inputs, max_new_tokens=1024, pad_token_id=pad_id
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
        **inputs, max_new_tokens=1024, pad_token_id=pad_id
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
        max_new_tokens=1024,
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


def run_llm(
    model_type: str,
    prompts_dict: Dict[str, Dict[str, Any]],
    data_dir: Path,
    num_ref: int,
    dataset: str,
    client: Optional[OpenAI] = None,
    model: Optional[Any] = None,
    processor: Optional[Any] = None,
    gpt_model_name: str = "gpt-4o",
    *,
    task: str = "localize",
    sam2_repo: str = "facebook/sam2.1-hiera-large",
    masks_dir: Optional[str] = None,
    overlays_dir: Optional[str] = None,
    multimask: bool = False,
    max_points: int = 10,
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
        model_type: One of 'gpt', 'qwen', 'llama', 'llava' or 'gemma'.
        prompts_dict: Mapping from sample keys to dictionaries containing
            'image' (path to the query image) and 'text' (prompt text).
        data_dir: Root of the dataset (e.g. datasets/mvtec_ad).
        image_size: Side length to resize images to before inference.
        num_ref: Number of reference images to use.
        dataset: Dataset name ('mvtec_ad', 'mvtec_ac', 'visa_ac').
        client: OpenAI client instance when model_type=='gpt'.
        model: Model instance for HF backends.
        processor: Processor associated with the HF model.
        gpt_model_name: Name of the GPT variant when model_type=='gpt'.

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
            ref_files = [f for f in os.listdir(ref_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if not ref_files:
                raise FileNotFoundError(f"No valid reference images found in {ref_dir}")
            ref_path = ref_dir / ref_files[0]
            ref_img = Image.open(ref_path).convert('RGB')
            if model_type == 'gpt':
                images.append(encode_image(ref_img))
            else:
                images.append(ref_img)
        # Load query image from prompts_dict
        query_path = value['image']
        query_img = Image.open(query_path).convert('RGB')
        if model_type == 'gpt':
            images.append(encode_image(query_img))
        else:
            images.append(query_img)
        # Retrieve prompt text
        text = value['text']
        # Invoke backend → get raw text (LLM must return strict JSON)
        if model_type == 'gpt':
            response, usage = get_gpt_output(client, images, text, gpt_model_name)
            total_tokens += usage.total_tokens
            raw_text = response
        elif model_type == 'qwen':
            out = get_qwen_output(model, processor, images, text)
            raw_text = out[0]
        elif model_type == 'llama':
            out = get_llama_output(model, processor, images, text)
            raw_text = out[0]
        elif model_type == 'llava':
            out = get_llava_output(model, processor, images, text)
            raw_text = out[0]
        elif model_type == 'gemma':
            out = get_gemma_output(model, processor, images, text)
            raw_text = out[0]
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        print(f"{key}: {raw_text}")

        # Parse JSON; in binary mode or 'normal' label, store as-is
        obj = parse_llm_json(raw_text)
        if task == 'binary' or obj.get('label') == 'normal':
            predictions[key] = obj
            continue

        # SAM-2 localization
        sam_adapter.set_image(query_img)
        H, W = query_img.height, query_img.width

        region_masks = []
        region_paths = []
        for r_idx, region in enumerate(obj.get('regions', [])):
            pos = region.get('points_positive', [])[:max_points]
            neg = region.get('points_negative', [])[:max_points]
            pos_px = np.array([[float(p['x']) * W, float(p['y']) * H] for p in pos], dtype=np.float32)
            neg_px = np.array([[float(p['x']) * W, float(p['y']) * H] for p in neg], dtype=np.float32) if neg else None

            box_px = None
            def _coverage(mask_u8, pts):
                if pts is None or len(pts) == 0: return 0.0
                h, w = mask_u8.shape
                ii = np.clip(np.round(pts[:, 1]).astype(int), 0, h-1)
                jj = np.clip(np.round(pts[:, 0]).astype(int), 0, w-1)
                return float(mask_u8[ii, jj].mean()) / 255.0

            # try as-is (x,y)
            masks, scores, _ = sam_adapter.predict_region(pos_pts_px=pos_px, neg_pts_px=neg_px, box_px=box_px, multimask_output=multimask)
            k = int(np.argmax(scores))
            mask_u8_xy = (masks[k].astype(np.uint8) * 255)

            # try swapped (y,x) in case LLM returned row/col
            pos_yx = pos_px[:, [1, 0]]
            neg_yx = neg_px[:, [1, 0]] if neg_px is not None else None
            masks2, scores2, _ = sam_adapter.predict_region(pos_pts_px=pos_yx, neg_pts_px=neg_yx, box_px=box_px, multimask_output=multimask)
            k2 = int(np.argmax(scores2))
            mask_u8_yx = (masks2[k2].astype(np.uint8) * 255)

            # choose orientation by positive-vs-negative coverage
            def _score(mask, p, n): 
                return _coverage(mask, p) - 0.5*_coverage(mask, n)

            score_xy = _score(mask_u8_xy, pos_px, neg_px)
            score_yx = _score(mask_u8_yx, pos_yx, neg_yx)
            use_yx = score_yx > score_xy
            mask_u8 = mask_u8_yx if use_yx else mask_u8_xy
            pos_best = pos_yx if use_yx else pos_px
            neg_best = neg_yx if use_yx else neg_px

            # one-time polarity flip if negatives are covered more than positives
            if _coverage(mask_u8, pos_best) < _coverage(mask_u8, neg_best):
                masks3, scores3, _ = sam_adapter.predict_region(
                    pos_pts_px=neg_best if neg_best is not None else np.empty((0,2), np.float32),
                    neg_pts_px=pos_best,
                    box_px=box_px,
                    multimask_output=multimask
                )
                mask_u8 = (masks3[int(np.argmax(scores3))].astype(np.uint8) * 255)

            if (pos_best is not None) and len(pos_best) >= 2 and box_px is None:
                x0, y0 = pos_best.min(axis=0); x1, y1 = pos_best.max(axis=0)
                pad = 0.05 * max(W, H)
                box_px = np.array([max(x0-pad,0), max(y0-pad,0), min(x1+pad,W-1), min(y1+pad,H-1)], dtype=np.float32)

            if 'bbox' in region:
                x0, y0, x1, y1 = region['bbox']
                box_px = np.array([x0*W, y0*H, x1*W, y1*H], dtype=np.float32)

            # Predict using the resolved polarity 
            masks_final, scores_final, _ =  sam_adapter.predict_region(
                pos_pts_px=pos_best if pos_best is not None else np.empty((0,2), np.float32),
                neg_pts_px=neg_best,
                box_px=box_px,
                multimask_output=multimask,
            )
            kf = int(np.argmax(scores_final))
            rmask = (masks_final[kf].astype(np.uint8) * 255)
            rpath = masks_root / f"{key}__r{r_idx}.png"
            Sam2Adapter.save_mask(rmask, rpath, hw_expected=(H, W))
            region_masks.append(rmask)
            region_paths.append(str(rpath))

        final_mask = Sam2Adapter.union_masks(region_masks) if region_masks else np.zeros((H, W), np.uint8)
        top1_path = masks_root / f"{key}.png"
        Sam2Adapter.save_mask(final_mask, top1_path, hw_expected=(H, W))
        Sam2Adapter.save_overlay(query_img, final_mask, overlays_root / f"{key}.png")

        obj['sam2'] = {"top1_path": str(top1_path), "all_paths": region_paths}
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
        choices=['gpt', 'qwen', 'llama', 'llava', 'gemma'],
        default='gpt',
        help='Model backend to use.'
    )
    parser.add_argument(
        '--gpt_model',
        type=str,
        choices=['gpt-4o', 'gpt-4o-mini'],
        default='gpt-4o',
        help='GPT model variant (only when --model=gpt).'
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
    parser.add_argument('--multimask', action='store_true')
    parser.add_argument('--max_points', type=int, default=10)

    args = parser.parse_args()
    # Retrieve dataset directory and prompts file
    data_dir, json_file_path = get_dataset_config(args.dataset)
    # Load prompts
    prompts_dict = load_json(json_file_path)
    # Initialise backend model/processor
    if args.model == 'gpt':
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = None
        processor = None
    elif args.model == 'qwen':
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
        client = None
    elif args.model == 'llama':
        hf_token = os.getenv("llama_access")
        model = MllamaForConditionalGeneration.from_pretrained(
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token
        )
        processor = AutoProcessor.from_pretrained(
            "meta-llama/Llama-3.2-11B-Vision-Instruct", token=hf_token
        )
        client = None
    elif args.model == 'llava':
        hf_token = os.getenv("llama_access")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token
        )
        processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf", token=hf_token
        )
        client = None
    elif args.model == 'gemma':
        hf_token = os.getenv("llama_access")
        model = Gemma3ForConditionalGeneration.from_pretrained(
            "google/gemma-3-12b-it",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token
        )
        processor = AutoProcessor.from_pretrained(
            "google/gemma-3-12b-it", token=hf_token
        )
        client = None
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    # Execute inference
    preds = run_llm(
        model_type=args.model,
        prompts_dict=prompts_dict,
        data_dir=data_dir,
        num_ref=args.num_ref,
        dataset=args.dataset,
        client=client,
        model=model,
        processor=processor,
        gpt_model_name=args.gpt_model if args.model == 'gpt' else None,
        task=args.task,
        sam2_repo=args.sam2_repo,
        masks_dir=args.masks_dir,
        overlays_dir=args.overlays_dir,
        multimask=args.multimask,
        max_points=args.max_points,
    )
    save_path = get_save_path(
        "binary",
        args.dataset,
        args.model,
        gpt_model_name=(args.gpt_model if args.model == 'gpt' else args.model),
    )
    save_json(preds, save_path)
    print(f"[✓] Predictions saved to: {save_path}")
    print("Done")

if __name__ == '__main__':
    main() 