import os
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
import torch
import random  # Added for random selection
from openai import OpenAI
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, MllamaForConditionalGeneration
from qwen_vl_utils import process_vision_info
from utils import load_json, save_json, get_save_path
from collections import defaultdict  # Moved import to top

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

def get_gpt_output(client: OpenAI, images: List[str], text: str, model_name: str, heatmap_mode: str) -> Tuple[str, Dict[str, int]]:
    """
    Get output from GPT model.
    
    Args:
        client: OpenAI client
        images: List of base64 encoded images
        text: Prompt text
        model_name: Name of the GPT model to use
        heatmap_mode: Heatmap visualization mode
    
    Returns:
        Tuple[str, Dict[str, int]]: Model output and token usage
    """
    if heatmap_mode == "none":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{images[0]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{images[1]}"}},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{images[0]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{images[1]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{images[2]}"}},
                ],
            }
        ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0
    )
    return response.choices[0].message.content, response.usage

def get_qwen_output(model: Any, processor: Any, input_imgs: List[Image.Image], input_txt: str, heatmap_mode: str) -> List[str]:
    """
    Get output from Qwen2-VL model.
    
    Args:
        model: Qwen2-VL model
        processor: Qwen2-VL processor
        input_imgs: List of PIL Images
        input_txt: Prompt text
        heatmap_mode: Heatmap visualization mode
    
    Returns:
        List[str]: Model output
    """
    if heatmap_mode == "none":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": input_imgs[0]},
                    {"type": "image", "image": input_imgs[1]},
                    {"type": "text", "text": input_txt},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": input_imgs[0]},
                    {"type": "image", "image": input_imgs[1]},
                    {"type": "image", "image": input_imgs[2]},
                    {"type": "text", "text": input_txt},
                ],
            }
        ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
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
    heatmap_mode: str
) -> List[str]:
    if heatmap_mode == "none":
        messages = [
            {
                "role": "system",
                "content": "You are a strict classifier. Answer with EXACTLY ONE label from the options in the user message. Lowercase, no punctuation, no extra words."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": input_imgs[0]},
                    {"type": "image", "image": input_imgs[1]},
                    {"type": "text", "text": input_txt},
                ],
            }
        ]
        image_inputs = [input_imgs[0], input_imgs[1]]
    else:
        messages = [
            {
                "role": "system",
                "content": "You are a strict classifier. Answer with EXACTLY ONE label from the options in the user message. Lowercase, no punctuation, no extra words."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": input_imgs[0]},
                    {"type": "image", "image": input_imgs[1]},
                    {"type": "image", "image": input_imgs[2]},
                    {"type": "text", "text": input_txt},
                ],
            }
        ]
        image_inputs = [input_imgs[0], input_imgs[1], input_imgs[2]]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Build tensors (IMPORTANT: use keyword args + return_tensors)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)

    pad_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
    generated_ids = model.generate(**inputs, max_new_tokens=128, pad_token_id=pad_id)
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

def parse_image_filename(image_name: str) -> Tuple[str, str]:
    """
    Parse image filename to extract category and defect class.
    
    Args:
        image_name: Image filename or key (e.g., 'candle_burnt_001')
        
    Returns:
        Tuple[str, str]: Category and defect class
    """
    temp, _, _ = image_name.rpartition('_')
    if temp.lower().startswith("metal"):
        category1, _, gt_class = temp.partition('_')
        category2, _, gt_class = gt_class.partition('_')
        category = f'{category1}_{category2}'.lower()
        gt_class = gt_class.lower()
    elif temp.lower().startswith("pipe"):
        category1, _, gt_class = temp.partition('_')
        category2, _, gt_class = gt_class.partition('_')
        category = f'{category1}_{category2}'.lower()
        gt_class = gt_class.lower()
    else:
        category, _, gt_class = temp.partition('_')
        category = category.lower()
        gt_class = gt_class.lower()
    return category, gt_class

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
    gpt_model_name: str = "gpt-4o"
) -> Dict[str, str]:
    """
    Run LLM model for anomaly detection.
    
    Args:
        model_type: Type of model to use ('gpt-4o', 'gpt-4o-mini' or 'qwen')
        prompts_dict: Dictionary of prompts
        data_dir: Directory containing the dataset
        image_size: Size to resize images to
        num_ref: Number of reference images to use
        heatmap_mode: Heatmap visualization mode
        dataset: Dataset name
        client: OpenAI client (for GPT model)
        model: Qwen model (for Qwen model)
        processor: Qwen processor (for Qwen model)
        gpt_model_name: Name of the GPT model to use (for GPT model)
    
    Returns:
        Dict[str, str]: Dictionary of predictions
    """
    predictions = {}
    total_tokens = 0
    
    # Group prompts by category and defect class
    grouped_prompts = defaultdict(list)
    for key, value in prompts_dict.items():
        object_cat, defect_class = parse_image_filename(key)
        grouped_prompts[(object_cat, defect_class)].append((key, value))
    
    # Select 2 random images per defect class per category
    filtered_prompts = {}
    for (object_cat, defect_class), items in grouped_prompts.items():
        if len(items) > 2:
            selected_items = random.sample(items, 2)
        else:
            selected_items = items
        for key, value in selected_items:
            filtered_prompts[key] = value
    
    for key, value in tqdm(filtered_prompts.items()):
        images = []
        object_cat = key.split('_')[0]
        
        # Load reference image
        if num_ref != 0:
            if dataset == 'mvtec_ad' or dataset == 'mvtec_ac':
                for i in range(num_ref):
                    if object_cat == 'metal':
                        image_path = Path(data_dir) / 'metal_nut' / 'test' / 'good' / f'00{i}.png'
                    else:
                        image_path = Path(data_dir) / object_cat / 'test' / 'good' / f'00{i}.png'
                    image = Image.open(image_path)
                    image = image.resize((image_size, image_size))
                    if model_type == 'gpt':
                        images.append(encode_image(image))
                    else:
                        images.append(image)
            elif dataset == 'visa_ac':
                for i in range(num_ref):
                    if object_cat == 'pipe':
                        good_dir = Path(data_dir) / 'pipe_fryum' / 'test' / 'good'
                    else:
                        good_dir = Path(data_dir) / object_cat / 'test' / 'good'
                    image_files = [f for f in os.listdir(good_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                    if not image_files:
                        raise FileNotFoundError(f"No valid images found in {good_dir}")
                    image_path = good_dir / image_files[0]
                    image = Image.open(image_path)
                    image = image.resize((image_size, image_size))
                    if model_type == 'gpt':
                        images.append(encode_image(image))
                    else:
                        images.append(image)
        
        image_path = value['image']
        query_image = Image.open(image_path)
        query_image = query_image.resize((image_size, image_size))
        
        if model_type == 'gpt':
            images.append(encode_image(query_image))
        else:
            images.append(query_image)
        
        defect_class = key.split('_')[1:-1]
        defect_class = '_'.join(defect_class)
        
        if heatmap_mode == "contour":
            if object_cat == 'metal':
                object_cat = 'metal_nut'
                defect_class = defect_class[4:]
            if object_cat == 'pipe':
                object_cat = 'pipe_fryum'
                defect_class = defect_class[6:]
            heatmap_dir = Path.cwd() / f'contour_gt_{dataset}'
            heatmap_path = heatmap_dir / object_cat / 'test' / defect_class / f'{key.split("_")[-1]}.png'
            heatmap = Image.open(heatmap_path)
            heatmap = heatmap.resize((image_size, image_size))
            if model_type == 'gpt':
                images.append(encode_image(heatmap))
            else:
                images.append(heatmap)
        
        text = value['text']
        
        if model_type == 'gpt':
            response, usage = get_gpt_output(client, images, text, gpt_model_name, heatmap_mode)
            total_tokens += usage.total_tokens
            print(f"{key}: {response} (Tokens used: {usage.total_tokens})")
            predictions[key] = response
        elif model_type == 'qwen':
            qwen_output = get_qwen_output(model, processor, images, text, heatmap_mode)
            print(f"{key}: {qwen_output[0]}")
            predictions[key] = qwen_output[0]
        elif model_type == 'llama':
            llama_output = get_llama_output(model, processor, images, text, heatmap_mode)
            print(f"{key}: {llama_output[0]}")
            predictions[key] = llama_output[0]
    
    if model_type == 'gpt':
        print(f"Total tokens used: {total_tokens}")
        cost_estimate = (total_tokens / 1e6) * 2.5
        print(f"Estimated cost: ${cost_estimate:.2f}")
    
    save_path = get_save_path(heatmap_mode, dataset, model_type, gpt_model_name)
    save_json(predictions, save_path)
    print(f"[✓] Predictions saved to: {save_path}")
    
    return predictions

def main():
    """Main function to run the LLM model."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM model for anomaly detection.")
    parser.add_argument('--model', type=str, choices=['gpt', 'qwen', 'llama'],
                        default='gpt', help='Model to use (gpt or qwen).')
    parser.add_argument('--gpt_model', type=str, choices=['gpt-4o', 'gpt-4o-mini'],
                        default='gpt-4o', help='GPT model to use (only applicable if model=gpt).')
    parser.add_argument('--dataset', type=str, choices=['mvtec_ad', 'mvtec_ac', 'visa_ac'],
                        default='mvtec_ac', help='Dataset to use.')
    parser.add_argument('--heatmap_mode', type=str, choices=['contour', 'none'],
                        default='contour', help='Heatmap visualization mode.')
    parser.add_argument('--image_size', type=int, default=448,
                        help='Size to resize images to.')
    parser.add_argument('--num_ref', type=int, default=1,
                        help='Number of reference images to use.')
    args = parser.parse_args()
    
    data_dir, json_file_path = get_dataset_config(args.dataset)
    
    prompts_dict = load_json(json_file_path)
    
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
            torch_dtype = torch.bfloat16,
            device_map = "auto",
            token = hf_token,
        )
        processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct", token = hf_token)
        client = None
    
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
        gpt_model_name=args.gpt_model if args.model == 'gpt' else None
    )
    
    print("Done")

if __name__ == '__main__':
    main()