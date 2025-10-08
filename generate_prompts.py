import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
from utils import save_json, load_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def make_localize_prompt(category: str, defect: str, guidance_text: str) -> str:
    return (
        f"You are a visual anomaly inspector for '{category}'. "
        "The first image - Image A - is a normal reference; the second image - Image B - is the query to analyze. "
        "Domain guidance for this query: "
        f"{guidance_text}. "
        "Return exactly one JSON object as the entire message — no other characters. For cases that there were no anomalies seen in Image B, return just an image-level label; For cases that anomalies were found on the query image B, return an image-level label, and sets of coordinates that represent positive points (which lie within and indicate the anomalous region) and negative points (which lie around the anomalous region to outline the anomaly for localization), and a bounding box which covers the entire region of points. The format for these 2 cases of responses are defined as follows: "
        "If no anomaly: {\"label\":\"normal\"}. "
        "If anomaly: {"
        "  \"label\": \"anomalous\","
        "  \"regions\": [{"
        "    \"points_positive\": [{\"x\": 0.xx, \"y\": 0.yy}, ...],"
        "    \"points_negative\": [{\"x\": 0.xx, \"y\": 0.yy}, ...],"
        "    \"bbox\": [x0, y0, x1, y1]"
        "  }],"
        "  \"confidence\": 0.0-1.0"
        "} "
        "Rules (must follow): "
        "- Coordinates are normalized to [0,1] on Image B at its original resolution (WxH). "
        "- Provide 6-10 points_positive strictly INSIDE the anomalous region only (distribute across its area and edges). "
        "- Provide 2-4 points_negative on the IMMEDIATELY ADJACENT intact area bordering the defect; these exclude the surrounding normal structure. "
        "- Provide a TIGHT bbox that encloses ONLY the defect with a small margin (~0.02-0.03 of image size), NOT the entire object/opening. "
        "- Do not add any prose—respond with the single JSON object only."
    )

def collect_prompts(
    data_dir: Path,
    object_categories: List[str],
    defects_data: Dict[str, Dict[str, List[str]]] | None = None,
) -> Dict[str, Dict[str, str]]:
    """
    Collect prompts for each test image across all categories.

    For each object category this function traverses the ``test`` subfolders
    (broken down by defect class) and records the path to every image file.
    A binary detection prompt is generated for the category and attached to each
    entry.  Keys in the resulting dictionary follow the pattern
    ``<category>_<defect_class>_<image_id>`` where ``image_id`` is derived from
    the filename stem.

    Args:
        data_dir: Root directory of the dataset (containing per-category
            subdirectories).
        object_categories: List of category names to process.

    Returns:
        Dict[str, Dict[str, str]]: A mapping from unique keys to dictionaries
            containing the image path and the binary prompt.
    """
    images_dict: Dict[str, Dict[str, str]] = {}
    for category in object_categories:
        logger.info(f"Processing category: {category}")
        test_dir = data_dir / category / 'test'
        if not test_dir.exists():
            logger.warning(f"Test directory not found for category {category}: {test_dir}")
            continue

        try:
            # Each defect class (including 'good') has its own subdirectory
            defect_classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(test_dir / d)])
            for defect_class in defect_classes:
                defect_dir = test_dir / defect_class
                if not defect_dir.exists():
                    logger.warning(f"Defect class directory not found: {defect_dir}")
                    continue
                image_files = sorted([f for f in os.listdir(defect_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                for image_file in image_files:
                    image_path = defect_dir / image_file
                    # Build a key that captures the category, defect class and image name without extension
                    key = f"{category}_{defect_class}_{Path(image_file).stem}"
                    # --- build guidance text from descriptions (if provided) ---
                    guidance_parts: List[str] = []
                    if defects_data and category in defects_data:
                        obj = defects_data[category]
                        # object-level "normal" overview
                        if isinstance(obj.get('normal'), list) and len(obj['normal']) >= 2:
                            guidance_parts.append(obj['normal'][1])
                        # defect-level description
                        if defect_class in obj and isinstance(obj[defect_class], list) and len(obj[defect_class]) >= 2:
                            guidance_parts.append(obj[defect_class][1])
                    guidance_text = " ".join(guidance_parts).strip()

                    prompt_text = make_localize_prompt(category, defect_class, guidance_text)
                    images_dict[key] = {'image': str(image_path), 'text': prompt_text}
        except Exception as e:
            logger.error(f"Error processing category {category}: {e}")
            continue
    logger.info(f"Collected {len(images_dict)} prompts")
    return images_dict

def get_dataset_config(dataset: str) -> Tuple[Path, Path, List[str], str]:
    """
    Get configuration for a specific dataset.

    Returns the data directory, list of object categories and output filename
    for the provided dataset.  Since binary detection no longer requires
    defect descriptions, the JSON description paths are omitted.

    Args:
        dataset: Name of the dataset ('mvtec_ad', 'mvtec_ac', or 'visa_ac').

    Returns:
        Tuple[Path, List[str], str]: Data directory, list of object categories, and
        output filename for the prompts JSON file.

    Raises:
        ValueError: If the dataset is not supported.
    """
    if dataset == 'mvtec_ad':
        data_dir = Path.cwd() / 'datasets' / 'mvtec_ad'
        json_path = Path.cwd() / 'configs' / 'mvtec_ad_des.json'
        filename = "mvtec_ad_prompts.json"
        object_categories = [
            'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
            'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
            'transistor', 'wood', 'zipper'
        ]
    elif dataset == 'mvtec_ac':
        data_dir = Path.cwd() / 'datasets' / 'mvtec_ac'
        json_path = Path.cwd() / 'configs' / 'mvtec_ac_des.json'
        filename = "mvtec_ac_prompts.json"
        object_categories = [
            'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
            'leather', 'metal_nut', 'pill', 'screw', 'tile', 'transistor',
            'wood', 'zipper'
        ]
    elif dataset == 'visa_ac':
        data_dir = Path.cwd() / 'datasets' / 'visa_ac'
        json_path = Path.cwd() / 'configs' / 'visa_ac_des.json'
        filename = "visa_ac_prompts.json"
        object_categories = [
            'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
            'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4',
            'pipe_fryum'
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return data_dir, json_path, object_categories, filename


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Only the dataset argument is required for binary prompt generation.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate binary prompts for anomaly detection.")
    parser.add_argument(
        '--dataset',
        default='mvtec_ac',
        type=str,
        choices=['mvtec_ad', 'mvtec_ac', 'visa_ac'],
        help='Dataset to use.'
    )
    parser.add_argument(
        '--descriptions_path',
        type = str,
        default=None,
        help='To override the default description JSON path.'
    )
    return parser.parse_args()


def main() -> None:
    """Main function to run the binary prompt generation."""
    args = parse_arguments()
    try:
        logger.info(f"Starting prompt generation for dataset: {args.dataset}")
        # Get dataset configuration
        data_dir, default_json_path, object_categories, filename = get_dataset_config(args.dataset)
        # Validate dataset directory
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        # Resolve descriptions path (CLI override takes precedence)
        json_path = Path(args.descriptions_path) if args.descriptions_path else default_json_path
        defects_data = {}
        if json_path.exists():
            defects_data = load_json(json_path)
            logger.info(f"Loaded descriptions from: {json_path}")
        else:
            logger.warning(f"Descriptions JSON not found, continuing without: {json_path}")

        # Collect prompts with injected guidance
        prompts = collect_prompts(data_dir, object_categories, defects_data=defects_data)
        
        # Prepare output path
        save_dir = Path.cwd() / 'configs' / 'prompts'
        save_path = save_dir / filename
        # Save prompts to JSON file
        save_json(prompts, save_path)
        logger.info(f"Successfully processed {len(prompts)} prompts")
        print(f"[✓] Prompts saved to: {save_path}")
    except Exception as e:
        logger.error(f"Error during prompt generation: {e}")
        raise


if __name__ == '__main__':
    main()




