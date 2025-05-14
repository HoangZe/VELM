"""
Anomaly vs. Defect Classification Evaluation Module.

This module evaluates model performance by categorizing anomaly classes into two groups:
1. Critical defects (randomly selected)
2. Negligible anomalies (remaining anomalies)

The evaluation measures how well the model can distinguish between normal samples,
critical defects, and negligible anomalies.
"""

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
from utils import load_json, save_json, map_predictions


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def count_samples_per_class(predictions: Dict[str, str], mapping: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, int]]:
    """
    Count the number of samples per class for each object category.
    
    Args:
        predictions: Dictionary mapping sample names to predicted classes
        mapping: Dictionary mapping object categories to class mappings
        
    Returns:
        Dict[str, Dict[str, int]]: Counts of samples per class for each object
    """
    class_counts = defaultdict(lambda: defaultdict(int))
    for sample, pred in predictions.items():
        parts = sample.split('_')
        obj = parts[0] if parts[0] != "metal" else "metal_nut"
        if obj == "metal_nut":
            extracted_true_class = "_".join(parts[2:-1])
        else:
            extracted_true_class = "_".join(parts[1:-1])  
        
        mapped_class = mapping.get(obj, {}).get(extracted_true_class, extracted_true_class)
        mapped_class = mapped_class[0] if isinstance(mapped_class, list) else mapped_class
        class_counts[obj][mapped_class] += 1  
    return class_counts

def evaluate(
    predictions: Dict[str, str], 
    mapping: Dict[str, Dict[str, List[str]]], 
    seeds: int = 5
) -> Dict[str, Dict[str, Union[Dict[str, float], Dict[str, float]]]]:
    """
    Evaluate model performance by categorizing anomalies into critical and negligible groups.
    
    Args:
        predictions: Dictionary mapping sample names to predicted classes
        mapping: Dictionary mapping object categories to class mappings
        seeds: Number of random seeds to use for evaluation
        
    Returns:
        Dict[str, Dict[str, Union[Dict[str, float], Dict[str, float]]]]: Evaluation results
    """
    # Get anomaly classes for each object (excluding normal)
    obj_anomaly_classes = {obj: [cls for cls in classes.keys() if cls != "normal"] 
                          for obj, classes in mapping.items()}
    
    # Count samples per class
    sample_counts = count_samples_per_class(predictions, mapping)
    
    # Initialize results storage
    obj_accuracies = defaultdict(lambda: {"normal": [], "anomaly": [], "defect": []})
    
    # Run evaluation with different random seeds
    for seed in range(seeds):
        logger.info(f"Running evaluation with seed {seed}")
        random.seed(seed)
        chosen_anomalies = {}
        
        # Select critical anomalies for each object
        for obj, anomaly_classes in obj_anomaly_classes.items():
            # Select 1 anomaly class if there are 4 or fewer, otherwise select 2
            num_anomalies = 1 if len(anomaly_classes) <= 4 else 2
            shuffled_classes = sorted(anomaly_classes, key=lambda cls: -sample_counts[obj][cls])
            random.shuffle(shuffled_classes)
            chosen_anomalies[obj] = set(shuffled_classes[:num_anomalies])
        
        # Initialize accuracy counters
        accuracy_counts = defaultdict(lambda: {
            "normal": {"correct": 0, "total": 0},
            "anomaly": {"correct": 0, "total": 0},
            "defect": {"correct": 0, "total": 0}
        })
        
        # Evaluate each sample
        for sample, predicted in predictions.items():
            parts = sample.split('_')
            obj = parts[0] if parts[0] != "metal" else "metal_nut"
            if obj == "metal_nut":
                extracted_true_class = "_".join(parts[2:-1])
            else:
                extracted_true_class = "_".join(parts[1:-1])  
            
            true_class = extracted_true_class
            predicted_class = predicted
            
            # Determine true category
            if true_class == "good":
                true_category = "normal"
            elif true_class in chosen_anomalies[obj]:
                true_category = "anomaly"
            else:
                true_category = "defect"
            
            # Determine predicted category
            if predicted_class == "good":
                predicted_category = "normal"
            elif predicted_class in chosen_anomalies[obj]:
                predicted_category = "anomaly"
            else:
                predicted_category = "defect"
            
            # Update accuracy counts
            correct = predicted_category == true_category
            accuracy_counts[obj][true_category]["correct"] += int(correct)
            accuracy_counts[obj][true_category]["total"] += 1
        
        # Calculate accuracies for this seed
        for obj, counts in accuracy_counts.items():
            for category in ["normal", "anomaly", "defect"]:
                total = counts[category]["total"]
                correct = counts[category]["correct"]
                acc = correct / total if total > 0 else 0
                obj_accuracies[obj][category].append(acc)
    
    # Calculate mean and standard deviation of accuracies
    mean_accuracies = {obj: {cat: np.mean(acc_list) for cat, acc_list in accs.items()} 
                      for obj, accs in obj_accuracies.items()}
    std_accuracies = {obj: {cat: np.std(acc_list) for cat, acc_list in accs.items()} 
                     for obj, accs in obj_accuracies.items()}
    
    # Calculate overall metrics
    overall_mean = {cat: np.mean([mean_accuracies[obj][cat] for obj in mean_accuracies]) 
                   for cat in ["normal", "anomaly", "defect"]}
    overall_std = {cat: np.mean([std_accuracies[obj][cat] for obj in std_accuracies]) 
                  for cat in ["normal", "anomaly", "defect"]}
    
    # Compile results
    results = {
        "accuracy_per_object": mean_accuracies,
        "std_per_object": std_accuracies,
        "overall_accuracy": overall_mean,
        "overall_std": overall_std
    }
    
    return results

def get_dataset_config(dataset: str, model_type: str, heatmap_mode: str) -> Tuple[Path, Path]:
    """
    Get configuration for a specific dataset.
    
    Args:
        dataset: Dataset name ('mvtec_ad', 'mvtec_ac', or 'visa_ac')
        model_type: Type of model used ('gpt-4o', 'gpt-4o-mini', or 'qwen')
        heatmap_mode: Heatmap visualization mode ('contour' or 'none')
        
    Returns:
        Tuple[Path, Path]: Predictions path and mapping path
        
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
    elif dataset == 'mvtec_ac':
        if heatmap_mode == "contour":
            predictions_path = Path.cwd() / 'configs' / 'predictions' / f"mvtec_ac_preds_{model_identifier}.json"
        else:  # heatmap_mode == "none"
            predictions_path = Path.cwd() / 'configs' / 'predictions' / f"normal_preds_{model_identifier}.json"
        mapping_path = Path.cwd() / 'configs' / 'mvtec_ac_des.json'
    elif dataset == 'visa_ac':
        if heatmap_mode == "contour":
            predictions_path = Path.cwd() / 'configs' / 'predictions' / f"visa_ac_preds_{model_identifier}.json"
        else:  # heatmap_mode == "none"
            predictions_path = Path.cwd() / 'configs' / 'predictions' / f"normal_preds_{model_identifier}.json"
        mapping_path = Path.cwd() / 'configs' / 'visa_ac_des.json'
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    return predictions_path, mapping_path

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate anomaly vs. defect classification')
    parser.add_argument('--dataset', type=str, choices=['mvtec_ad', 'mvtec_ac', 'visa_ac'], 
                        default='mvtec_ac', help='Dataset to evaluate')
    parser.add_argument('--model', type=str, choices=['gpt-4o', 'gpt-4o-mini', 'qwen'],
                        default='gpt-4o', help='Model type used for predictions')
    parser.add_argument('--heatmap_mode', type=str, choices=['contour', 'none'],
                        default='contour', help='Heatmap visualization mode')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of random seeds to use for evaluation')
    parser.add_argument('--output', type=str, default='anom_def_results.json',
                        help='Path to save evaluation results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    return parser.parse_args()

def main():
    """Main function to run the evaluation."""
    args = parse_arguments()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        logger.info(f"Starting anomaly vs. defect evaluation for dataset: {args.dataset}, "
                   f"model: {args.model}, heatmap mode: {args.heatmap_mode}")
        
        # Get dataset configuration
        predictions_path, mapping_path = get_dataset_config(args.dataset, args.model, args.heatmap_mode)
        
        # Load data
        logger.info(f"Loading predictions from {predictions_path}")
        predictions = load_json(predictions_path)
        
        logger.info(f"Loading mapping data from {mapping_path}")
        mapping = load_json(mapping_path)
        
        # Map predictions to standardized class names
        logger.info("Mapping predictions to standardized class names")
        mapped_predictions = map_predictions(predictions, mapping)
        
        # Run evaluation
        logger.info(f"Running evaluation with {args.seeds} random seeds")
        results = evaluate(mapped_predictions, mapping, seeds=args.seeds)
        
        # Save results
        output_path = Path.cwd() / 'configs' / 'evaluations'/ args.output
        save_json(results, output_path)
        logger.info(f"Results saved to {output_path}")
        
        # Print summary
        logger.info("Evaluation summary:")
        logger.info(f"Overall accuracy - Normal: {results['overall_accuracy']['normal']:.4f}, "
                   f"Anomaly: {results['overall_accuracy']['anomaly']:.4f}, "
                   f"Defect: {results['overall_accuracy']['defect']:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == '__main__':
    main()
