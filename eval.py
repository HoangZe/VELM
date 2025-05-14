"""
Evaluation module for anomaly classification models.

This module provides functionality to evaluate model predictions against ground truth
labels using various metrics including accuracy, F1 score, Cohen's Kappa, and confusion matrix.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from utils import load_json
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsCalculator:
    """
    A class to calculate various evaluation metrics for model predictions.
    
    This class encapsulates the calculation of common classification metrics
    including accuracy, F1 score (macro and micro), Cohen's Kappa, and confusion matrix.
    """
    
    def __init__(self, gt_labels: List[str], pred_labels: List[str], class_names: List[str]):
        """
        Initialize the MetricsCalculator with ground truth and predicted labels.
        
        Args:
            gt_labels: List of ground truth labels
            pred_labels: List of predicted labels
            class_names: List of unique class names
        """
        self.gt_labels = gt_labels
        self.pred_labels = pred_labels
        self.class_names = class_names

    def compute_accuracy(self) -> float:
        """
        Compute the accuracy score.
        
        Returns:
            float: The accuracy score
        """
        return accuracy_score(self.gt_labels, self.pred_labels)

    def compute_macro_f1(self) -> float:
        """
        Compute the macro-averaged F1 score.
        
        Returns:
            float: The macro-averaged F1 score
        """
        return f1_score(self.gt_labels, self.pred_labels, average='macro', labels=self.class_names)

    def compute_micro_f1(self) -> float:
        """
        Compute the micro-averaged F1 score.
        
        Returns:
            float: The micro-averaged F1 score
        """
        return f1_score(self.gt_labels, self.pred_labels, average='micro', labels=self.class_names)

    def compute_cohens_kappa(self) -> float:
        """
        Compute Cohen's Kappa score.
        
        Returns:
            float: The Cohen's Kappa score
        """
        return cohen_kappa_score(self.gt_labels, self.pred_labels, labels=self.class_names)

    def compute_confusion_matrix(self) -> np.ndarray:
        """
        Compute the confusion matrix.
        
        Returns:
            np.ndarray: The confusion matrix
        """
        return confusion_matrix(self.gt_labels, self.pred_labels, labels=self.class_names)
    
    def compute_all_metrics(self) -> Dict[str, Union[float, List[List[int]]]]:
        """
        Compute all available metrics.
        
        Returns:
            Dict[str, Union[float, List[List[int]]]]: Dictionary containing all metrics
        """
        return {
            'accuracy': self.compute_accuracy(),
            'macro_f1': self.compute_macro_f1(),
            'micro_f1': self.compute_micro_f1(),
            'cohen_kappa': self.compute_cohens_kappa(),
            'confusion_matrix': self.compute_confusion_matrix().tolist()
        }

def map_predicted_class(json_data: Dict[str, Any], category: str, predicted_class: str) -> str:
    """
    Map a predicted class to its corresponding class in the mapping data.
    
    Args:
        json_data: Dictionary containing class mappings
        category: Category of the object
        predicted_class: Predicted class name
        
    Returns:
        str: Mapped class name or "Unknown class" if not found
    """
    try:
        if predicted_class == 'good':
            return 'good'
        else:
            mapping = {values[0].lower(): key for key, values in json_data[category].items() if key != 'normal'}
            return mapping.get(predicted_class.lower(), "Unknown class")
    except KeyError:
        logger.warning(f"Category '{category}' not found in mapping data")
        return "Unknown class"

def parse_image_filename(image_name: str) -> Tuple[str, str]:
    """
    Parse an image filename to extract category and ground truth class.
    
    Args:
        image_name: Image filename
        
    Returns:
        Tuple[str, str]: Category and ground truth class
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

def prepare_labels(
    predictions: Dict[str, str], 
    json_data: Dict[str, Any], 
    object_category: str
) -> Tuple[List[str], List[str]]:
    """
    Prepare ground truth and predicted labels for evaluation.
    
    Args:
        predictions: Dictionary mapping image names to predicted classes
        json_data: Dictionary containing class mappings
        object_category: Category of the object
        
    Returns:
        Tuple[List[str], List[str]]: Ground truth labels and predicted labels
    """
    gt_labels = []
    pred_labels = []
    
    for image, predicted_class in predictions.items():
        try:
            _, gt_class = parse_image_filename(image)
            predicted_class = predicted_class.lower()
            mapped_predicted_class = map_predicted_class(json_data, object_category, predicted_class)
            
            gt_labels.append(gt_class)
            pred_labels.append(mapped_predicted_class)
        except Exception as e:
            logger.error(f"Error processing image {image}: {e}")
    
    return gt_labels, pred_labels

def evaluate_predictions(
    predictions_path: Path, 
    mapping_path: Path, 
    object_category: str
) -> Dict[str, Union[float, List[List[int]]]]:
    """
    Evaluate model predictions against ground truth labels.
    
    Args:
        predictions_path: Path to the predictions JSON file
        mapping_path: Path to the mapping JSON file
        object_category: Category of the object
        
    Returns:
        Dict[str, Union[float, List[List[int]]]]: Dictionary containing evaluation metrics
    """
    logger.info(f"Loading predictions from {predictions_path}")
    predictions = load_json(predictions_path)
    #  Filter predictions to current object category
    predictions = {
        img: pred for img, pred in predictions.items()
        if parse_image_filename(img)[0] == object_category
    }
    logger.info(f"Loading mapping data from {mapping_path}")
    json_data = load_json(mapping_path)
    
    logger.info("Preparing labels for evaluation")
    gt_labels, pred_labels = prepare_labels(predictions, json_data, object_category)
    
    if not gt_labels or not pred_labels:
        logger.warning("No valid labels found for evaluation")
        return {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'micro_f1': 0.0,
            'cohen_kappa': 0.0,
            'confusion_matrix': []
        }
    
    class_names = list({label for label in gt_labels + pred_labels})
    logger.info(f"Found {len(class_names)} unique classes: {class_names}")
    
    metrics = MetricsCalculator(gt_labels, pred_labels, class_names)
    results = metrics.compute_all_metrics()
    
    logger.info(f"Evaluation complete. Accuracy: {results['accuracy']:.4f}")
    return results

def get_dataset_config(dataset: str, model_type: str, heatmap_mode: str) -> Tuple[Path, Path, List[str]]:
    """
    Get configuration for a specific dataset.
    
    Args:
        dataset: Dataset name ('mvtec_ad', 'mvtec_ac', or 'visa_ac')
        model_type: Type of model used ('gpt-4o', 'gpt-4o-mini', or 'qwen')
        heatmap_mode: Heatmap visualization mode ('contour' or 'none')
        
    Returns:
        Tuple[Path, Path, List[str]]: Predictions path, mapping path, and object categories
        
    Raises:
        ValueError: If the dataset is not supported
    """
    # Create model identifier for file naming
    model_identifier = model_type.replace('-', '_')
    
    if dataset == 'mvtec_ad':
        if heatmap_mode == "contour":
            predictions_path = Path.cwd() / 'configs' / 'predictions' / f"mvtec_ad_preds_{model_identifier}.json"
        else:  # heatmap_mode == "none"
            predictions_path = Path.cwd() / 'configs' / 'predictions' / f"normal_preds_{model_identifier}.json"
        mapping_path = Path.cwd() / 'configs' / 'mvtec_ad_des.json'
        object_categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                           'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
                           'wood', 'zipper']
    elif dataset == 'mvtec_ac':
        if heatmap_mode == "contour":
            predictions_path = Path.cwd() / 'configs' / 'predictions' / f"mvtec_ac_preds_{model_identifier}.json"
        else:  # heatmap_mode == "none"
            predictions_path = Path.cwd() / 'configs' / 'predictions' / f"normal_preds_{model_identifier}.json"
        mapping_path = Path.cwd() / 'configs' / 'mvtec_ac_des.json'
        object_categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                           'leather', 'metal_nut', 'pill', 'screw', 'tile', 'transistor',
                           'wood', 'zipper']
    elif dataset == 'visa_ac':
        if heatmap_mode == "contour":
            predictions_path = Path.cwd() / 'configs' / 'predictions' / f"visa_ac_preds_{model_identifier}.json"
        else:  # heatmap_mode == "none"
            predictions_path = Path.cwd() / 'configs' / 'predictions' / f"normal_preds_{model_identifier}.json"
        mapping_path = Path.cwd() / 'configs' / 'visa_ac_des.json'
        object_categories = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
                           'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    return predictions_path, mapping_path, object_categories

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate model predictions for anomaly detection')
    parser.add_argument('--dataset', type=str, choices=['mvtec_ad', 'mvtec_ac', 'visa_ac'], 
                        default='mvtec_ac', help='Dataset to evaluate')
    parser.add_argument('--model', type=str, choices=['gpt-4o', 'gpt-4o-mini', 'qwen'],
                        default='gpt-4o', help='Model type used for predictions')
    parser.add_argument('--heatmap_mode', type=str, choices=['contour', 'none'],
                        default='contour', help='Heatmap visualization mode')
    parser.add_argument('--output', type=str, help='Path to save evaluation results (optional)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    return parser.parse_args()

def main():
    """Main function to run the evaluation."""
    args = parse_arguments()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        logger.info(f"Starting evaluation for dataset: {args.dataset}, model: {args.model}, heatmap mode: {args.heatmap_mode}")
        predictions_path, mapping_path, object_categories = get_dataset_config(args.dataset, args.model, args.heatmap_mode)
        
        # Evaluate for each object category
        all_results = {}
        for category in object_categories:
            logger.info(f"Evaluating category: {category}")
            results = evaluate_predictions(predictions_path, mapping_path, category)
            all_results[category] = results
        
        # Calculate overall metrics
        overall_results = {
            'accuracy': np.mean([results['accuracy'] for results in all_results.values()]),
            'macro_f1': np.mean([results['macro_f1'] for results in all_results.values()]),
            'micro_f1': np.mean([results['micro_f1'] for results in all_results.values()]),
            'cohen_kappa': np.mean([results['cohen_kappa'] for results in all_results.values()])
        }
        all_results['overall'] = overall_results
        
        # Print or save results
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=4)
            logger.info(f"Results saved to {output_path}")
        else:
            print(json.dumps(all_results, indent=4))
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == '__main__':
    main()
