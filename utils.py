"""
Utility functions for LLM-based Anomaly Classification.

This module provides common functionality used across multiple scripts in the Repo.
"""

import os
import json
import logging
from pathlib import Path
import yaml
from typing import Dict, List, Union, Any

def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configure logging with appropriate verbosity.
    
    Args:
        verbose: Whether to enable verbose logging
        
    Returns:
        logging.Logger: Configured logger
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        Dict[str, Any]: Loaded JSON data
    """
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def get_save_path(heatmap_mode: str, dataset: str, model_type: str, gpt_model_name: str = "gpt-4o") -> Path:
    """
    Get the save path for predictions based on heatmap mode and dataset.
    
    Args:
        heatmap_mode: Heatmap visualization mode
        dataset: Dataset name
        model_type: Type of model used
        gpt_model_name: Name of the GPT model used (for GPT model)
    
    Returns:
        Path: Save path for predictions
    """
    # For GPT models, include the specific model name in the filename
    if model_type == 'gpt':
        model_identifier = gpt_model_name.replace('-', '_')
    else:
        model_identifier = model_type
        
    if heatmap_mode == "contour":
        if dataset == 'mvtec_ad':
            return Path.cwd() / 'configs' / 'predictions' / f'mvtec_ad_preds_{model_identifier}.json'
        elif dataset == 'mvtec_ac':
            return Path.cwd() / 'configs' / 'predictions' / f'mvtec_ac_preds_{model_identifier}.json'
        elif dataset == 'visa_ac':
            return Path.cwd() / 'configs' / 'predictions' / f'visa_ac_preds_{model_identifier}.json'
    elif heatmap_mode == "none":
        return Path.cwd() / 'configs' / 'predictions' / f'normal_preds_{model_identifier}.json'
    else:
        raise ValueError(f"Invalid heatmap mode: {heatmap_mode}")
    

def map_predictions(predictions: Dict[str, str], mapping: Dict[str, Dict[str, List[str]]]) -> Dict[str, str]:
    """
    Map predicted class names to their standardized names using the mapping dictionary.
    
    Args:
        predictions: Dictionary mapping sample names to predicted classes
        mapping: Dictionary mapping object categories to class mappings
        
    Returns:
        Dict[str, str]: Mapped predictions
    """
    reverse_mapping = {}
    for _, classes in mapping.items():
        for actual, given_names in classes.items():
            if actual != "normal":  # Ignore normal class
                reverse_mapping.update({name.lower(): actual for name in given_names})
    
    mapped_predictions = {key: reverse_mapping.get(value.lower(), value.lower()) 
                         for key, value in predictions.items()}
    return mapped_predictions

def load_config(config_path: Union[str, Path]) -> Dict:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

def validate_paths(heatmap_dir: Path, image_dir: Path, save_dir: Path) -> None:
    """
    Validate that required directories exist.
    
    Args:
        heatmap_dir: Directory containing heatmap images
        image_dir: Directory containing original images
        save_dir: Directory to save output images
        
    Raises:
        FileNotFoundError: If any required directory doesn't exist
    """
    if not heatmap_dir.exists():
        raise FileNotFoundError(f"Heatmap directory not found: {heatmap_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    # Create save directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)