"""
Draw red contour lines on the query image based on detected anomalies.

Given a binary heatmap of detected anomalies and a query image, this module
draws red contour lines around the anomalies on the image.

To adapt to different scenarios, modify the configuration file at:
"/configs/contour_config.yaml".
"""

import os
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from typing import List, Tuple
from utils import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_image(
    image_path: Path, 
    heatmap_path: Path, 
    image_size: int, 
    dataset: str
) -> Tuple[np.ndarray, bool]:
    """
    Process a single image and its corresponding heatmap.
    
    Args:
        image_path: Path to the original image
        heatmap_path: Path to the heatmap image
        image_size: Target size for resizing images
        dataset: Dataset name ('mvtec_ad', 'mvtec_ac', or 'visa_ac')
        
    Returns:
        Tuple containing the processed image and a flag indicating if it's grayscale
    """
    # Load and resize original image
    original_image = Image.open(image_path)
    original_image = original_image.resize((image_size, image_size))
    original_image = np.array(original_image)
    
    # Check if the image is grayscale
    is_grayscale = len(original_image.shape) == 2
    if is_grayscale:
        # Convert grayscale image to 3-channel grayscale image
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    
    # Load and resize heatmap
    heatmap = Image.open(heatmap_path)
    heatmap = heatmap.resize((image_size, image_size))
    heatmap_image_resized = np.array(heatmap)
    
    # Create binary heatmap based on dataset
    if dataset == 'visa_ac':
        binary_heatmap = (heatmap_image_resized > 0).astype(np.uint8) * 255
    else:  # mvtec_ad or mvtec_ac
        _, binary_heatmap = cv2.threshold(heatmap_image_resized, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary heatmap
    contours, _ = cv2.findContours(binary_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the contours on the original image
    contour_image = cv2.drawContours(original_image.copy(), contours, -1, (255, 0, 0), 2)
    
    return contour_image, is_grayscale

def save_contour_image(
    contour_image: np.ndarray, 
    save_path: Path, 
    is_grayscale: bool
) -> None:
    """
    Save the contour image to disk.
    
    Args:
        contour_image: The image with contours drawn
        save_path: Path where the image should be saved
        is_grayscale: Flag indicating if the image is grayscale
    """
    # Ensure the save directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the image
    plt.figure(figsize=(10, 10))
    plt.imshow(contour_image, cmap='gray' if is_grayscale else None)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def get_heatmap_path(
    heatmap_dir: Path, 
    category: str, 
    defect_class: str, 
    image_file: str, 
    dataset: str
) -> Path:
    """
    Get the path to the heatmap image based on dataset type.
    
    Args:
        heatmap_dir: Base directory for heatmaps
        category: Category name
        defect_class: Defect class name
        image_file: Image filename
        dataset: Dataset name
        
    Returns:
        Path to the heatmap image
    """
    image_name = image_file.split(".")[0]
    
    if dataset == 'mvtec_ad' or dataset == 'mvtec_ac':
        return heatmap_dir / category / 'ground_truth' / defect_class / f'{image_name}_mask.png'
    elif dataset == 'visa_ac':
        return heatmap_dir / category / 'groundtruth' / defect_class / f'{image_name}.png'
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def process_category(
    heatmap_dir: Path, 
    image_dir: Path, 
    save_dir: Path, 
    category: str, 
    image_size: int, 
    dataset: str
) -> None:
    """
    Process all images in a category.
    
    Args:
        heatmap_dir: Directory containing heatmap images
        image_dir: Directory containing original images
        save_dir: Directory to save output images
        category: Category name
        image_size: Target size for resizing images
        dataset: Dataset name
    """
    if category.endswith('.txt'):
        return
        
    test_dir = image_dir / category / 'test'
    if not test_dir.exists():
        logger.warning(f"Test directory not found for category {category}: {test_dir}")
        return
        
    defect_classes = os.listdir(test_dir)
    for defect_class in defect_classes:
        if defect_class == 'good':
            continue
            
        defect_class_dir = test_dir / defect_class
        if not defect_class_dir.exists():
            logger.warning(f"Defect class directory not found: {defect_class_dir}")
            continue
            
        image_files = os.listdir(defect_class_dir)
        for image_file in image_files:
            try:
                image_path = defect_class_dir / image_file
                heatmap_path = get_heatmap_path(heatmap_dir, category, defect_class, image_file, dataset)
                
                if not heatmap_path.exists():
                    logger.warning(f"Heatmap not found: {heatmap_path}")
                    continue
                
                # Process the image
                contour_image, is_grayscale = process_image(
                    image_path, heatmap_path, image_size, dataset
                )
                
                # Save the result
                save_path = save_dir / category / "test" / defect_class / f'{image_file.split(".")[0]}.png'
                save_contour_image(contour_image, save_path, is_grayscale)
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")

def create_contour_visualizations(
    heatmap_dir: Path, 
    image_dir: Path, 
    save_dir: Path, 
    categories: List[str], 
    image_size: int, 
    dataset: str
) -> None:
    """
    Create contour visualizations for all images in the dataset.
    
    Args:
        heatmap_dir: Directory containing heatmap images
        image_dir: Directory containing original images
        save_dir: Directory to save output images
        categories: List of categories to process
        image_size: Target size for resizing images
        dataset: Dataset name
    """
    logger.info(f"Starting contour visualization for dataset: {dataset}")
    logger.info(f"Processing {len(categories)} categories")
    
    # Process each category
    for category in tqdm(categories, desc="Processing categories"):
        process_category(heatmap_dir, image_dir, save_dir, category, image_size, dataset)
    
    logger.info("Contour visualization completed successfully")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create contour visualizations for anomaly detection datasets')
    parser.add_argument('--config', type=str, default='configs/contour_config.yaml', help='Path to configuration file')
    parser.add_argument('--dataset', type=str, choices=['mvtec_ad', 'mvtec_ac', 'visa_ac'], 
                        help='Dataset to process (overrides config file)')
    parser.add_argument('--image_size', type=int, help='Image size for resizing (overrides config file)')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Configuration file not found: {args.config}. Using default values.")
        config = {}
    
    # Override config with command line arguments if provided
    dataset = args.dataset if args.dataset else config.get('dataset', 'mvtec_ac')
    image_size = args.image_size if args.image_size else config.get('image_size', 448)
    
    # Get dataset-specific parameters from config
    dataset_config = config.get(dataset, {})
    
    # Get paths from config with fallbacks
    heatmap_dir = Path.cwd() / Path(dataset_config.get('heatmap_dir', dataset))
    image_dir = Path.cwd() / Path(dataset_config.get('image_dir', dataset))
    save_dir = Path.cwd() / Path(dataset_config.get('save_dir', f'contour_gt_{dataset}'))
    
    # Get categories from config
    categories = dataset_config.get('categories', [])
    
    # Create contour visualizations
    create_contour_visualizations(heatmap_dir, image_dir, save_dir, categories, image_size, dataset) 