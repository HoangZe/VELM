"""
Run LLM models for anomaly classification.

This script supports multiple LLM backends (GPT-4o, GPT-4o-mini, and Qwen2-VL) for anomaly classification
with various heatmap visualization modes.
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
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
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


def get_gpt_output(client: OpenAI, images: List[str], text: str, model_name: str, heatmap_mode: str) -> Tuple[str, Dict[str, int]]:
    """
    Get output from GPT model.
    
    Args:
        client: OpenAI client
        images: List of base64 encoded images
        text: Prompt text
        model_name: Name of the GPT model to use
    
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
    # Multiple image message
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
    
    # Preparation for inference
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
    model = model.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
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
    
    # Loop over the prompts
    for key, value in tqdm(prompts_dict.items()):
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
                    image = image.resize((image_size, image_size))  # Resize the image
                    if model_type == 'gpt':
                        images.append(encode_image(image))
                    else:
                        images.append(image)
            elif dataset == 'visa_ac':
                for i in range(num_ref):
                    if object_cat == 'pipe':
                        image_path = Path(data_dir) / 'pipe_fryum' / 'test' / 'good' / f'00{i}.png'
                    else:
                        image_path = Path(data_dir) / object_cat / 'test' / 'good' / f'00{i}.png'
                    image = Image.open(image_path)
                    image = image.resize((image_size, image_size))
                    if model_type == 'gpt':
                        images.append(encode_image(image))
                    else:
                        images.append(image)
        
        # Define the path to the image file
        image_path = value['image']
        query_image = Image.open(image_path)
        query_image = query_image.resize((image_size, image_size))
        
        if model_type == 'gpt':
            images.append(encode_image(query_image))
        else:
            images.append(query_image)
        
        defect_class = key.split('_')[1:-1]
        defect_class = '_'.join(defect_class)
        
        # Load the heatmap based on the heatmap mode
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
        
        # Get the text
        text = value['text']
        
        # Call the appropriate model
        if model_type == 'gpt':
            response, usage = get_gpt_output(client, images, text, gpt_model_name)
            total_tokens += usage.total_tokens
            print(f"{key}: {response} (Tokens used: {usage.total_tokens})")
            predictions[key] = response
        else:  # qwen
            qwen_output = get_qwen_output(model, processor, images, text, heatmap_mode)
            print(f"{key}: {qwen_output[0]}")
            predictions[key] = qwen_output[0]
    
    # Print token usage for GPT model
    if model_type == 'gpt':
        print(f"Total tokens used: {total_tokens}")
        # Estimate the cost
        cost_estimate = (total_tokens / 1e6) * 2.5  # assuming $2.5 per 1M tokens
        print(f"Estimated cost: ${cost_estimate:.2f}")
    
    # Save the predictions to a JSON file
    save_path = get_save_path(heatmap_mode, dataset, model_type, gpt_model_name)
    save_json(predictions, save_path)
    print(f"[✓] Predictions saved to: {save_path}")
    
    return predictions


def main():
    """Main function to run the LLM model."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM model for anomaly detection.")
    parser.add_argument('--model', type=str, choices=['gpt', 'qwen'],
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
    
    # Get dataset configuration
    data_dir, json_file_path = get_dataset_config(args.dataset)
    
    # Load the prompts from the JSON file
    prompts_dict = load_json(json_file_path)
    
    # Initialize model based on type
    if args.model == 'gpt':
        # Set OpenAI API key
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = None
        processor = None
    else:  # qwen
        # Load the model on the available device(s)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        client = None
    
    # Run the LLM
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