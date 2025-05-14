"""
DDAD Reorganizer Module.

This module reorganizes the output of DDAD to make it compatible with the VELM framework. It restructures the heatmap files 
to match the expected directory structure for evaluation.
"""

import argparse
import logging
import re
import shutil
from pathlib import Path
from typing import List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def numerical_sort(value: str) -> List[Union[str, int]]:
    """
    Helper function to sort filenames numerically.
    
    This function extracts numeric parts from a string and converts them to integers
    for proper numerical sorting.
    
    Args:
        value: String to sort
        
    Returns:
        List[Union[str, int]]: List of parts for sorting
    """
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def reorganize_raw_data(
    raw_data_dir: Path, 
    save_dir: Path, 
    model_dir: Path
) -> None:
    """
    Reorganize raw DDAD output data to match the expected directory structure.
    
    This function takes the raw output from DDAD and reorganizes it to match
    the directory structure expected by the LLMAD framework. It copies files
    to their appropriate locations based on object category and defect class.
    
    Args:
        raw_data_dir: Directory containing raw DDAD output
        save_dir: Directory to save reorganized data
        model_dir: Directory containing the original model data (for reference)
    """
    raw_data_path = Path(raw_data_dir)
    save_path = Path(save_dir)
    model_path = Path(model_dir)

    # Ensure save directory exists
    save_path.mkdir(parents=True, exist_ok=True)

    # Get object categories from model directory
    object_categories = [d for d in model_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(object_categories)} object categories")

    for category in object_categories:
        category_name = category.name
        logger.info(f"Processing category: {category_name}")

        # Find all reconstructed heatmap files for this category
        raw_heatmap_files = sorted(
            (raw_data_path / category_name).glob(f"{category_name}_reconstructed_*.png"), 
            key=lambda x: numerical_sort(str(x))
        )
        logger.info(f"Found {len(raw_heatmap_files)} reconstructed files for category {category_name}")

        # Get the defect class names and their counts from the model directory
        defect_classes = sorted([d for d in (model_path / category_name / 'test').iterdir() if d.is_dir()])
        count_classes = {d.name: len(list(d.glob("*.png"))) for d in defect_classes}
        logger.info(f"Defect classes for {category_name}: {count_classes}")

        # Create save folders for each defect class
        for defect_class in count_classes.keys():
            if defect_class == 'good':
                continue  # Skip good class for saving
            defect_save_path = save_path / category_name / defect_class
            defect_save_path.mkdir(parents=True, exist_ok=True)

        # Copy files to their appropriate locations
        start_idx = 0  # Initialize starting index
        for defect_class, count in count_classes.items():
            end_idx = start_idx + count
            logger.info(f"Processing {defect_class} with {count} samples (indices {start_idx}-{end_idx-1})")
            
            for idx in range(start_idx, end_idx):
                new_filename = f"{idx - start_idx:03}.png"
                src_file = raw_heatmap_files[idx]
                
                if defect_class != 'good':
                    dst_file = save_path / category_name / defect_class / new_filename
                    shutil.copy(src_file, dst_file)  # Copy the file to the new location
                    logger.debug(f"Copied {src_file} to {dst_file}")
            
            start_idx = end_idx  # Update starting index for the next class

    logger.info(f"Reorganization complete. Files saved to {save_path}")

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Reorganize DDAD output data')
    parser.add_argument('--raw_data_dir', type=str, default='results',
                        help='Directory containing raw DDAD output')
    parser.add_argument('--save_dir', type=str, default='ddad_results/reconstructed',
                        help='Directory to save reorganized data')
    parser.add_argument('--model_dir', type=str, default='mvtec_ad',
                        help='Directory containing the original model data')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    return parser.parse_args()

def main():
    """Main function to run the reorganization."""
    args = parse_arguments()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Convert string paths to Path objects
        raw_data_dir = Path.cwd() / args.raw_data_dir
        save_dir = Path.cwd() / args.save_dir
        model_dir = Path.cwd() / args.model_dir
        
        logger.info(f"Starting DDAD reorganization")
        logger.info(f"Raw data directory: {raw_data_dir}")
        logger.info(f"Save directory: {save_dir}")
        logger.info(f"Model directory: {model_dir}")
        
        # Run reorganization
        reorganize_raw_data(
            raw_data_dir=raw_data_dir,
            save_dir=save_dir,
            model_dir=model_dir
        )
        
        logger.info("DDAD reorganization completed successfully")
        
    except Exception as e:
        logger.error(f"Reorganization failed: {e}")
        raise

if __name__ == '__main__':
    main()