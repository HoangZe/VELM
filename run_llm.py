"""
Run multimodal LLMs for binary anomaly detection.

This script provides a simplified inference pipeline for determining whether a
query image contains any anomaly compared to a reference (good) image.  It
supports several multimodal large language model backends including OpenAI
GPT‑4o, Qwen2.5‑VL, LLaMa‑3.2‑Vision‑Instruct, Llava‑1.6‑Mistral and
Gemma‑3.  The original VELM framework used a two‑stage approach with a
vision expert producing heatmaps and a language model performing
multi‑class classification.  Here we instead pass only the reference and
query images to the LMM together with a simple binary detection prompt.
The model is expected to answer ``anomalous`` or ``normal``.
Heatmaps and contour overlays are no longer supported; the ``heatmap_mode``
argument is retained for API compatibility but ignored.
"""

import os
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
import torch
from openai import OpenAI
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, MllamaForConditionalGeneration, LlavaNextForConditionalGeneration, LlavaNextProcessor, Gemma3ForConditionalGeneration
from qwen_vl_utils import process_vision_info
from utils import load_json, save_json, get_save_path


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
    heatmap_mode: str,
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
        heatmap_mode: Deprecated; retained to avoid breaking existing calls.

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
    heatmap_mode: str,
) -> List[str]:
    """
    Get output from a Qwen2.5‑VL model for binary anomaly detection.

    Only the first two images in ``input_imgs`` (reference and query) are
    utilised.  The ``heatmap_mode`` argument is retained for API
    compatibility but ignored.  The function constructs a chat message
    accordingly and invokes the model.

    Args:
        model: Loaded Qwen2.5‑VL model.
        processor: Qwen processor used to prepare inputs.
        input_imgs: List of PIL images (reference followed by query).
        input_txt: Prompt text instructing the model.
        heatmap_mode: Ignored parameter.

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
    generated_ids = model.generate(**inputs, max_new_tokens=128)
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
    heatmap_mode: str,
) -> List[str]:
    """
    Invoke a LLaMa‑3.2‑Vision‑Instruct model for binary anomaly detection.

    Only the first two images (reference and query) are considered.  A
    system instruction is prepended to enforce that the model replies with
    exactly one of the expected labels.  The ``heatmap_mode`` argument is
    ignored and retained solely for compatibility.

    Args:
        model: The LLaMa model instance.
        processor: Associated processor.
        input_imgs: List of PIL images (first is reference, second is query).
        input_txt: Prompt text.
        heatmap_mode: Ignored parameter.

    Returns:
        List[str]: Decoded model responses.
    """
    imgs = input_imgs[:2]
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a strict classifier. Answer with EXACTLY ONE label from the options in the user message. Lowercase, no punctuation, no extra words."
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
        **inputs, max_new_tokens=128, pad_token_id=pad_id
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
    heatmap_mode: str,
) -> List[str]:
    """
    Query a Llava‑1.6‑Mistral‑7B model for binary anomaly detection.

    Only the first two images (reference and query) are used.  The
    ``heatmap_mode`` argument is ignored.  A single user message is
    constructed containing both images and the text prompt.

    Args:
        model: Llava model instance.
        processor: Llava processor instance.
        input_imgs: List of PIL images (reference and query).
        input_txt: Prompt text.
        heatmap_mode: Ignored parameter.

    Returns:
        List[str]: Decoded model outputs.
    """
    imgs = input_imgs[:2]
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a strict classifier. Answer with EXACTLY ONE label from the options in the user message. Lowercase, no punctuation, no extra words."
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
        **inputs, max_new_tokens=128, pad_token_id=pad_id
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
    heatmap_mode: str,
) -> List[str]:
    """
    Call a Gemma‑3 model for binary anomaly detection.

    Similar to other backends, only the first two images (reference and query)
    are passed to the model.  A system prompt constrains the response to
    exactly one label.  The heatmap mode parameter is ignored.

    Args:
        model: Gemma‑3 model instance.
        processor: Corresponding processor.
        input_imgs: List of PIL images (first is reference, second is query).
        input_txt: Prompt text instructing the model.
        heatmap_mode: Ignored parameter for compatibility.

    Returns:
        List[str]: Decoded outputs.
    """
    imgs = input_imgs[:2]
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a strict classifier. Answer with EXACTLY ONE label from the options in the user message. Lowercase, no punctuation, no extra words."
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
        do_pan_and_scan=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)
    pad_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=16,
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
    image_size: int,
    num_ref: int,
    heatmap_mode: str,
    dataset: str,
    client: Optional[OpenAI] = None,
    model: Optional[Any] = None,
    processor: Optional[Any] = None,
    gpt_model_name: str = "gpt-4o",
) -> Dict[str, str]:
    """
    Run a multimodal LLM for binary anomaly detection.

    For each entry in ``prompts_dict`` this function loads a reference
    image from the dataset (assumed to be in ``test/good``) and the query
    image specified in the prompts dictionary.  It resizes both images to
    ``image_size`` and passes them to the selected model backend along with
    the prompt text.  The backend should return either ``anomalous`` or
    ``normal``.  Heatmap functionality from the original VELM framework is
    disabled; the ``heatmap_mode`` argument is ignored.

    Args:
        model_type: One of 'gpt', 'qwen', 'llama', 'llava' or 'gemma'.
        prompts_dict: Mapping from sample keys to dictionaries containing
            'image' (path to the query image) and 'text' (prompt text).
        data_dir: Root of the dataset (e.g. datasets/mvtec_ad).
        image_size: Side length to resize images to before inference.
        num_ref: Number of reference images to use.  Only the first is used.
        heatmap_mode: Ignored parameter, kept for backwards compatibility.
        dataset: Dataset name ('mvtec_ad', 'mvtec_ac', 'visa_ac').
        client: OpenAI client instance when model_type=='gpt'.
        model: Model instance for HF backends.
        processor: Processor associated with the HF model.
        gpt_model_name: Name of the GPT variant when model_type=='gpt'.

    Returns:
        Dict[str, str]: Dictionary mapping sample keys to raw model responses.
    """
    predictions: Dict[str, str] = {}
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
            ref_img = Image.open(ref_path).convert('RGB').resize((image_size, image_size))
            if model_type == 'gpt':
                images.append(encode_image(ref_img))
            else:
                images.append(ref_img)
        # Load query image from prompts_dict
        query_path = value['image']
        query_img = Image.open(query_path).convert('RGB').resize((image_size, image_size))
        if model_type == 'gpt':
            images.append(encode_image(query_img))
        else:
            images.append(query_img)
        # Retrieve prompt text
        text = value['text']
        # Invoke appropriate model backend
        if model_type == 'gpt':
            response, usage = get_gpt_output(client, images, text, gpt_model_name, heatmap_mode)
            total_tokens += usage.total_tokens
            print(f"{key}: {response} (Tokens used: {usage.total_tokens})")
            predictions[key] = response
        elif model_type == 'qwen':
            out = get_qwen_output(model, processor, images, text, heatmap_mode)
            print(f"{key}: {out[0]}")
            predictions[key] = out[0]
        elif model_type == 'llama':
            out = get_llama_output(model, processor, images, text, heatmap_mode)
            print(f"{key}: {out[0]}")
            predictions[key] = out[0]
        elif model_type == 'llava':
            out = get_llava_output(model, processor, images, text, heatmap_mode)
            print(f"{key}: {out[0]}")
            predictions[key] = out[0]
        elif model_type == 'gemma':
            out = get_gemma_output(model, processor, images, text, heatmap_mode)
            print(f"{key}: {out[0]}")
            predictions[key] = out[0]
    # Summarise token usage for GPT models
    if model_type == 'gpt':
        print(f"Total tokens used: {total_tokens}")
        cost_estimate = (total_tokens / 1e6) * 2.5
        print(f"Estimated cost: ${cost_estimate:.2f}")
    # Determine save path and persist predictions
    save_path = get_save_path(heatmap_mode, dataset, model_type, gpt_model_name)
    save_json(predictions, save_path)
    print(f"[✓] Predictions saved to: {save_path}")
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
        '--image_size',
        type=int,
        default=448,
        help='Resize images to this square dimension.'
    )
    parser.add_argument(
        '--num_ref',
        type=int,
        default=1,
        help='Number of reference images to use (only the first is used).'
    )
    # Retain heatmap_mode for backwards compatibility but default to 'none'
    parser.add_argument(
        '--heatmap_mode',
        type=str,
        choices=['contour', 'none'],
        default='none',
        help='Heatmap mode (ignored in binary detection).'
    )
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
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
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
    run_llm(
        model_type=args.model,
        prompts_dict=prompts_dict,
        data_dir=data_dir,
        image_size=args.image_size,
        num_ref=args.num_ref,
        heatmap_mode=args.heatmap_mode,
        dataset=args.dataset,
        client=client,
        model=model,
        processor=processor,
        gpt_model_name=args.gpt_model if args.model == 'gpt' else None,
    )
    print("Done")


if __name__ == '__main__':
    main() 