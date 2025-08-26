"""
Generate prompts for binary anomaly detection.

This module prepares simple prompts for a binary anomaly detection task.  For
each test image in a supported dataset, it constructs a prompt instructing a
multimodal language model (LMM) to compare a reference "good" image with the
query image and decide whether the query contains any anomalies.  The prompts
produced by this script do not rely on dataset-specific defect descriptions,
contours or heatmaps.  Instead they use a generic instruction pattern that
works across categories and datasets.

Example prompt:

    The first image shows a normal OBJECT.  The second image shows another
    OBJECT.  Compare the two images and decide whether the second image
    contains any anomaly or defect.  If it does, reply with "anomalous".
    Otherwise reply with "normal".  Only reply with the single word answer.

Replace `OBJECT` with the category name.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
from utils import save_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_binary_prompt(category: str) -> str:
    """
    Generate a binary detection prompt for a given category.

    This helper takes a category name (e.g. "bottle" or "pipe_fryum") and
    substitutes it into a generic template instructing the model to compare a
    normal reference image with a query image.  The prompt tells the model to
    respond with either "anomalous" or "normal" only.

    Args:
        category: The name of the object category.

    Returns:
        str: A completed prompt string suitable for inference.
    """
    base_text = (
        "The first image shows a normal OBJECT. The second image shows another OBJECT. "
        "Compare the two images and decide whether the second image contains any anomaly or defect. "
        "If it does, reply with \"anomalous\". Otherwise reply with \"normal\". "
        "Only reply with the single word answer."
    )
    return base_text.replace("OBJECT", category)


def collect_prompts(
    data_dir: Path,
    object_categories: List[str],
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
            defect_classes = [d for d in os.listdir(test_dir) if os.path.isdir(test_dir / d)]
            prompt_text = generate_binary_prompt(category)
            for defect_class in defect_classes:
                defect_dir = test_dir / defect_class
                if not defect_dir.exists():
                    logger.warning(f"Defect class directory not found: {defect_dir}")
                    continue
                image_files = [f for f in os.listdir(defect_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.png'))]
                for image_file in image_files:
                    image_path = defect_dir / image_file
                    # Build a key that captures the category, defect class and image name without extension
                    key = f"{category}_{defect_class}_{Path(image_file).stem}"
                    images_dict[key] = {
                        'image': str(image_path),
                        'text': prompt_text
                    }
        except Exception as e:
            logger.error(f"Error processing category {category}: {e}")
            continue
    logger.info(f"Collected {len(images_dict)} prompts")
    return images_dict

def get_dataset_config(dataset: str) -> Tuple[Path, List[str], str]:
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
        filename = "mvtec_ad_prompts.json"
        object_categories = [
            'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
            'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
            'transistor', 'wood', 'zipper'
        ]
    elif dataset == 'mvtec_ac':
        data_dir = Path.cwd() / 'datasets' / 'mvtec_ac'
        filename = "mvtec_ac_prompts.json"
        object_categories = [
            'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
            'leather', 'metal_nut', 'pill', 'screw', 'tile', 'transistor',
            'wood', 'zipper'
        ]
    elif dataset == 'visa_ac':
        data_dir = Path.cwd() / 'datasets' / 'visa_ac'
        filename = "visa_ac_prompts.json"
        object_categories = [
            'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
            'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4',
            'pipe_fryum'
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return data_dir, object_categories, filename


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
    return parser.parse_args()


def main() -> None:
    """Main function to run the binary prompt generation."""
    args = parse_arguments()
    try:
        logger.info(f"Starting prompt generation for dataset: {args.dataset}")
        # Get dataset configuration without defect descriptions
        data_dir, object_categories, filename = get_dataset_config(args.dataset)
        # Validate the dataset directory
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        # Collect prompts
        prompts = collect_prompts(data_dir, object_categories)
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




