"""
Generate prompts for anomaly classification.

This module generates and processes prompts for anomaly classification tasks,
supporting multiple datasets and text generation formats.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
from utils import load_json, save_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_text_conditioned(
    base_text: str, 
    category: str, 
    defects_data: Dict[str, Dict[str, List[str]]]
) -> str:
    """
    Generate conditioned text with detailed defect descriptions.
    
    Args:
        base_text: Template text with placeholders
        category: Object category name
        defects_data: Dictionary containing defect descriptions
    
    Returns:
        str: Generated text with replaced placeholders and defect descriptions
    """
    try:
        normal_description = defects_data[category]['normal'][0]
        defect_classes = []
        defect_descriptions = []
        
        for defect_class, descriptions in defects_data[category].items():
            if defect_class != 'normal':
                defect_classes.append(descriptions[0])
                defect_descriptions.append(f"{descriptions[0]}: {descriptions[1]}")
        
        defect_str = ', '.join(defect_classes)
        numbered_descriptions_str = ', '.join([f"{i+1}. {descr}" for i, descr in enumerate(defect_descriptions)])
        
        # Replace the placeholders in the base text
        modified_text = base_text.replace('OBJECT', category)
        modified_text = modified_text.replace('NORMAL_DES', normal_description)
        modified_text = modified_text.replace('NEW_DEFECTS', defect_str)
        
        # Append the description block
        description_block = f"The anomaly classes are: {numbered_descriptions_str}. Choose the anomaly class of the image. Only return the class name."
        modified_text += description_block

        return modified_text
    except KeyError as e:
        logger.error(f"Missing key in defects data for category {category}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error generating text for category {category}: {e}")
        raise


def collect_prompts(
    data_dir: Path, 
    defects_data: Dict[str, Dict[str, List[str]]], 
    object_categories: List[str], 
    base_text: str, 
    ddad: bool = False
) -> Dict[str, Dict[str, str]]:
    """
    Collect prompts and associated images for each category and defect class.
    
    Args:
        data_dir: Directory containing the dataset
        defects_data: Dictionary containing defect descriptions
        object_categories: List of object categories to process
        base_text: Template text for prompts
        ddad: Whether to use DDAD format
    
    Returns:
        dict: Dictionary containing prompts and associated image paths
    """
    images_dict = {}
    
    # Iterating over each object category
    for category in object_categories:
        logger.info(f"Processing category: {category}")
        test_dir = data_dir / category / 'test'
        
        if not test_dir.exists():
            logger.warning(f"Test directory not found for category {category}: {test_dir}")
            continue
            
        try:
            all_defect_classes = [d for d in os.listdir(test_dir) if os.path.isdir(test_dir / d)]
            defect_classes = [defect_class for defect_class in all_defect_classes if defect_class != 'good']
            
            # Generate text based on category and defect classes
            text = generate_text_conditioned(base_text, category, defects_data)
            
            # Iterating over each defect class
            for defect_class in defect_classes:
                logger.debug(f"Processing defect class: {defect_class}")
                
                if ddad:
                    recons_dir = Path.cwd() / 'ddad_results' / 'reconstructed' / category / 'test' / defect_class
                    query_dir = Path.cwd() / 'ddad_results' / 'queries' / category / defect_class
                    
                    if not recons_dir.exists() or not query_dir.exists():
                        logger.warning(f"DDAD directories not found for {category}/{defect_class}")
                        continue
                        
                    recons_files = [f for f in os.listdir(recons_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                    query_files = [f for f in os.listdir(query_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                    
                    if len(recons_files) != len(query_files):
                        logger.warning(f"Mismatch in number of files for {category}/{defect_class}: {len(recons_files)} vs {len(query_files)}")
                        # Use the minimum length to avoid index errors
                        min_len = min(len(recons_files), len(query_files))
                        recons_files = recons_files[:min_len]
                        query_files = query_files[:min_len]
                    
                    for i in range(len(recons_files)):
                        key = f"{category}_{defect_class}_{recons_files[i].split('.')[0]}"
                        images_dict[key] = {
                            'image': str(query_dir / query_files[i]), 
                            'recons': str(recons_dir / recons_files[i]), 
                            'text': text
                        }
                else:
                    defect_class_dir = test_dir / defect_class
                    if not defect_class_dir.exists():
                        logger.warning(f"Defect class directory not found: {defect_class_dir}")
                        continue
                        
                    image_files = [f for f in os.listdir(defect_class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                    
                    for image_file in image_files:
                        # Define the path to the current image file
                        image_path = defect_class_dir / image_file
                        # Define the key as per your requirement
                        key = f"{category}_{defect_class}_{image_file.split('.')[0]}"
                        # Add the image to the dictionary
                        images_dict[key] = {'image': str(image_path), 'text': text}
        except Exception as e:
            logger.error(f"Error processing category {category}: {e}")
            continue
            
    logger.info(f"Collected {len(images_dict)} prompts")
    return images_dict

def get_dataset_config(dataset: str) -> Tuple[Path, Path, List[str], str]:
    """
    Get configuration for a specific dataset.
    
    Args:
        dataset: Dataset name ('mvtec_ad', 'mvtec_ac', or 'visa_ac')
        
    Returns:
        Tuple[Path, Path, List[str], str]: Data directory, JSON path, object categories, and output filename
        
    Raises:
        ValueError: If the dataset is not supported
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
            'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    return data_dir, json_path, object_categories, filename


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Preprocess prompts for anomaly detection.")
    parser.add_argument('--dataset', default='mvtec_ac', type=str,
                        choices=['mvtec_ad', 'mvtec_ac', 'visa_ac'],
                        help='Dataset to use.')
    parser.add_argument('--text_type', default='conditioned', type=str,
                        choices=['raw', 'conditioned'],
                        help='Text generation format.')
    parser.add_argument('--ddad_format', default= False, action='store_true',
                        help='Whether to use DDAD format.')
    return parser.parse_args()


def main() -> None:
    """Main function to run the prompt generation."""
    args = parse_arguments()
    
    try:
        logger.info(f"Starting prompt preprocessing for dataset: {args.dataset}")
        
        # Get dataset configuration
        data_dir, json_path, object_categories, filename = get_dataset_config(args.dataset)
        
        # Validate paths
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
        if not json_path.exists():
            logger.error(f"JSON file not found: {json_path}")
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        # Load defect descriptions
        defects_data = load_json(json_path)
        
        # Define base texts for different formats
        base_texts = {
            'raw': (
                "The first image is a normal OBJECT. The second image is an anomalous OBJECT. The normal OBJECT NORMAL_DES. "
            ),
            'conditioned': (
                "The first image is a normal OBJECT. The second image is an anomalous OBJECT. The third image indicates the detected anomaly by a red line contour. The normal OBJECT NORMAL_DES. "
            )
        }
        
        base_text = base_texts[args.text_type]
        
        # Collect prompts
        logger.info(f"Collecting prompts with text type: {args.text_type}")
        prompts = collect_prompts(data_dir, defects_data, object_categories, base_text, ddad=args.ddad_format)
        
        save_dir = Path.cwd() / 'configs' / 'prompts'  
        save_path = save_dir / filename
        
        # Save prompts
        save_json(prompts, save_path)
        logger.info(f"Successfully processed {len(prompts)} prompts")
        print(f"[✓] Prompts saved to: {save_path}")
        
    except Exception as e:
        logger.error(f"Error during prompt preprocessing: {e}")
        raise


if __name__ == '__main__':
    main()




