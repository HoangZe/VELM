import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
from PIL import Image
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
)

from utils import load_json
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsCalculator:
    """
    Compute binary classification metrics.

    This class encapsulates the calculation of accuracy, precision, recall,
    F1 score (for the positive class) and the confusion matrix.  It accepts
    lists of ground truth and predicted labels and assumes that the label
    names used are consistent (e.g. 'normal' and 'anomalous').
    """

    def __init__(self, gt_labels: List[str], pred_labels: List[str]):
        self.gt_labels = gt_labels
        self.pred_labels = pred_labels

    def compute_accuracy(self) -> float:
        return accuracy_score(self.gt_labels, self.pred_labels)

    def compute_precision(self) -> float:
        return precision_score(self.gt_labels, self.pred_labels, pos_label='anomalous', zero_division=0)

    def compute_recall(self) -> float:
        return recall_score(self.gt_labels, self.pred_labels, pos_label='anomalous', zero_division=0)

    def compute_f1(self) -> float:
        return f1_score(self.gt_labels, self.pred_labels, pos_label='anomalous', zero_division=0)

    def compute_confusion_matrix(self) -> np.ndarray:
        # Ensure consistent order of labels
        return confusion_matrix(self.gt_labels, self.pred_labels, labels=['normal', 'anomalous'])

    def compute_all_metrics(self) -> Dict[str, Union[float, List[List[int]]]]:
        return {
            'accuracy': self.compute_accuracy(),
            'precision': self.compute_precision(),
            'recall': self.compute_recall(),
            'f1': self.compute_f1(),
            'confusion_matrix': self.compute_confusion_matrix().tolist(),
        }

# --- add: pixel-level helpers ---

def load_binary_mask(path: Path, target_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load a binary mask as uint8 {0,1}. If target_hw is given (H, W), resize with nearest neighbor.
    """
    m = Image.open(path).convert("L")
    if target_hw is not None:
        # PIL sizes are (W,H); we pass (W,H) in reverse order
        m = m.resize((target_hw[1], target_hw[0]), resample=Image.NEAREST)
    arr = np.array(m)
    # binarize: >0 is foreground
    return (arr > 0).astype(np.uint8)

def pixel_confusion(pred: np.ndarray, gt: np.ndarray) -> Tuple[int,int,int,int]:
    """
    Return TP, FP, FN, TN for anomaly class (1=anomaly, 0=background).
    """
    assert pred.shape == gt.shape, f"Shape mismatch: pred{pred.shape} vs gt{gt.shape}"
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    return tp, fp, fn, tn

def iou_dice_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    denom_iou = (tp + fp + fn)
    denom_dice = (2*tp + fp + fn)
    iou = float(tp) / denom_iou if denom_iou > 0 else 0.0
    dice = float(2*tp) / denom_dice if denom_dice > 0 else 0.0
    return iou, dice

def pixel_prec_rec_f1(tp:int, fp:int, fn:int) -> Tuple[float,float,float]:
    prec = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2*prec*rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def parse_key_for_gt(key: str) -> Tuple[str,str,str]:
    """
    Parse '<category>_<defect>_<id>' into (category, defect, id).
    Handles 'metal_nut' and 'pipe_fryum' as two-word categories.
    """
    temp, _, img_id = key.rpartition('_')
    if temp.startswith('metal') or temp.startswith('pipe'):
        cat1, _, rest = temp.partition('_')
        cat2, _, defect = rest.partition('_')
        category = f"{cat1}_{cat2}"
    else:
        category, _, defect = temp.partition('_')
    return category.lower(), defect.lower(), img_id.lower()

def gt_mask_path_for_mvtec(data_root: Path, category: str, defect: str, img_id: str) -> Path:
    """
    MVTec AD/AC convention:
      test image:        <category>/test/<defect>/<img_id>.png
      ground-truth mask: <category>/ground_truth/<defect>/<img_id>_mask.png
    """
    return (data_root / category / "ground_truth" / defect / f"{img_id}_mask.png")

def normalize_prediction(predicted: str) -> str:
    """
    Normalize a predicted label to either 'normal' or 'anomalous'.

    The language models may return synonyms or slight variations of the
    expected answers.  This helper maps a variety of acceptable responses to
    the canonical labels.  Any unknown value defaults to 'anomalous' to
    penalize uncertain or incorrect predictions conservatively.

    Args:
        predicted: Raw predicted label from the model.

    Returns:
        str: Normalized label ('normal' or 'anomalous').
    """
    if not predicted:
        return 'anomalous'
    p = predicted.strip().lower()
    # Common synonyms for normal/good
    normal_synonyms = {'normal', 'good', 'ok', 'no anomaly', 'no anomalies', 'no defect', 'none', 'defect free'}
    anomalous_synonyms = {'anomalous', 'anomaly', 'defect', 'yes', 'abnormal', 'faulty', 'defective'}
    if p in normal_synonyms:
        return 'normal'
    if p in anomalous_synonyms:
        return 'anomalous'
    # Handle cases like 'normal.' or 'anomalous.' by stripping punctuation
    p_clean = ''.join(ch for ch in p if ch.isalnum() or ch.isspace())
    if p_clean in normal_synonyms:
        return 'normal'
    if p_clean in anomalous_synonyms:
        return 'anomalous'
    # Default: treat unknown as anomalous
    return 'anomalous'

def coerce_labels_from_preds(preds_obj: Dict[str, object]) -> Dict[str, str]:
    labels = {}
    for k, v in preds_obj.items():
        if isinstance(v, str):
            labels[k] = v
        elif isinstance(v, dict):
            lab = v.get("label")
            if lab not in ("normal","anomalous"):
                raise ValueError(f"Bad label for {k}: {lab}")
            labels[k] = lab
        else:
            raise ValueError(f"Unsupported prediction type for {k}")
    return labels

def parse_image_filename(image_name: str) -> Tuple[str, str]:
    """
    Parse an image key to extract category and ground truth defect class.

    The input ``image_name`` is the key used in the predictions JSON, which
    follows the pattern ``<category>_<defect>_<imageid>``.  This function
    extracts the category and defect parts taking into account two-part
    categories such as ``metal_nut`` and ``pipe_fryum``.

    Args:
        image_name: Key from the predictions dictionary.

    Returns:
        Tuple[str, str]: (category, defect_class)
    """
    temp, _, _ = image_name.rpartition('_')
    if temp.lower().startswith("metal"):
        cat1, _, rest = temp.partition('_')
        cat2, _, gt_class = rest.partition('_')
        category = f"{cat1}_{cat2}".lower()
        gt_class = gt_class.lower()
    elif temp.lower().startswith("pipe"):
        cat1, _, rest = temp.partition('_')
        cat2, _, gt_class = rest.partition('_')
        category = f"{cat1}_{cat2}".lower()
        gt_class = gt_class.lower()
    else:
        category, _, gt_class = temp.partition('_')
        category = category.lower()
        gt_class = gt_class.lower()
    return category, gt_class

def prepare_labels(
    predictions: Dict[str, str],
    object_category: str
) -> Tuple[List[str], List[str]]:
    """
    Generate ground truth and normalized predicted labels for a category.

    Ground truth labels are derived from the prediction keys: if the defect
    class extracted from the key is 'good' then the ground truth label is
    'normal'; otherwise it is 'anomalous'.  Predicted labels are
    normalized using :func:`normalize_prediction`.

    Args:
        predictions: Mapping from image keys to raw predicted strings.
        object_category: Category of interest.

    Returns:
        Tuple[List[str], List[str]]: Lists of ground truth and predicted labels.
    """
    coerced = coerce_labels_from_preds(predictions)
    gt_labels: List[str] = []
    pred_labels: List[str] = []
    for image_key, predicted in coerced.items():
        try:
            category, defect_class = parse_image_filename(image_key)
            if category != object_category:
                continue
            # Ground truth label: 'normal' if defect_class == 'good' else 'anomalous'
            gt_label = 'normal' if defect_class == 'good' else 'anomalous'
            gt_labels.append(gt_label)
            pred_labels.append(normalize_prediction(predicted))
        except Exception as e:
            logger.error(f"Error processing key {image_key}: {e}")
    return gt_labels, pred_labels

def evaluate_predictions(
    predictions_path: Path,
    object_category: str
) -> Dict[str, Union[float, List[List[int]]]]:
    """
    Evaluate binary anomaly detection predictions for a single category.

    Args:
        predictions_path: Path to the predictions JSON produced by the LLM pipeline.
        object_category: Category to evaluate.

    Returns:
        Dict[str, Union[float, List[List[int]]]]: Computed metrics for the category.
    """
    logger.info(f"Loading predictions from {predictions_path}")
    predictions = load_json(predictions_path)
    logger.info("Preparing labels for evaluation")
    gt_labels, pred_labels = prepare_labels(predictions, object_category)
    if not gt_labels:
        logger.warning("No valid labels found for evaluation in category '%s'", object_category)
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'confusion_matrix': []}
    metrics = MetricsCalculator(gt_labels, pred_labels)
    results = metrics.compute_all_metrics()
    logger.info(
        f"Evaluation complete. Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}"
    )
    return results

def evaluate_pixel_masks(
    predictions_path: Path,
    dataset: str,
    data_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate pixel-level anomaly localization for datasets with GT masks.
    Uses the binary masks saved by the SAM-2 stage in the predictions JSON
    (obj['sam2']['top1_path']).
    Returns per-category and overall IoU, Dice (F1), pixel-precision/recall.

    Notes:
      - Designed for MVTec AD/AC folder layout (binary masks under ground_truth).
      - If a sample's GT mask file doesn't exist (e.g., 'good' images), it is skipped.
    """
    preds = load_json(predictions_path)
    if data_root is None:
        # default dataset roots under ./datasets/<dataset_name>
        data_root = Path.cwd() / "datasets" / dataset

    per_cat_counts: Dict[str, Dict[str,int]] = {}  # {cat: {'tp':..,'fp':..,'fn':..}}
    num_images: Dict[str,int] = {}

    for key, obj in preds.items():
        try:
            category, defect, img_id = parse_key_for_gt(key)
        except Exception:
            continue
        # only evaluate defective samples that have GT masks
        if defect == "good":
            continue
        # predicted mask from SAM-2
        sam2_info = obj.get("sam2", {})
        pred_path = sam2_info.get("top1_path")
        if not pred_path or not os.path.exists(pred_path):
            continue

        # ground-truth mask (MVTec AD/AC convention)
        gt_path = gt_mask_path_for_mvtec(data_root, category, defect, img_id)
        if not gt_path.exists():
            # no GT mask -> skip
            continue

        gt = load_binary_mask(gt_path)
        pred = load_binary_mask(Path(pred_path), target_hw=gt.shape)  # ensure H,W match

        tp, fp, fn, _ = pixel_confusion(pred, gt)
        if category not in per_cat_counts:
            per_cat_counts[category] = {'tp':0,'fp':0,'fn':0}
            num_images[category] = 0
        per_cat_counts[category]['tp'] += tp
        per_cat_counts[category]['fp'] += fp
        per_cat_counts[category]['fn'] += fn
        num_images[category] += 1

    # aggregate per category
    per_category: Dict[str, Any] = {}
    all_tp = all_fp = all_fn = 0
    for cat, c in per_cat_counts.items():
        iou, dice = iou_dice_from_counts(c['tp'], c['fp'], c['fn'])
        prec, rec, f1 = pixel_prec_rec_f1(c['tp'], c['fp'], c['fn'])
        per_category[cat] = {
            'num_images': num_images.get(cat, 0),
            'IoU': iou,
            'Dice': dice,
            'pixel_precision': prec,
            'pixel_recall': rec,
            'pixel_F1': f1,
        }
        all_tp += c['tp']; all_fp += c['fp']; all_fn += c['fn']

    # overall (micro-average)
    overall_iou, overall_dice = iou_dice_from_counts(all_tp, all_fp, all_fn)
    overall_prec, overall_rec, overall_f1 = pixel_prec_rec_f1(all_tp, all_fp, all_fn)
    overall = {
        'IoU': overall_iou,
        'Dice': overall_dice,
        'pixel_precision': overall_prec,
        'pixel_recall': overall_rec,
        'pixel_F1': overall_f1,
    }

    return {'per_category': per_category, 'overall': overall}

def get_dataset_config(dataset: str, model_type: str) -> Tuple[Path, List[str]]:
    """
    Determine the predictions file path and object categories for a dataset.

    Args:
        dataset: Dataset name ('mvtec_ad', 'mvtec_ac', or 'visa_ac').
        model_type: Model identifier used when saving predictions.

    Returns:
        Tuple[Path, List[str]]: Path to the predictions JSON file and list
        of object categories.

    Raises:
        ValueError: If the dataset is unsupported.
    """
    model_identifier = model_type.replace('-', '_')
    if dataset == 'mvtec_ad':
        pred_path = (
            Path.cwd()
            / 'configs'
            / 'predictions'
            / f"mvtec_ad_binary_preds_{model_identifier}.json"
        )
        object_categories = [
            'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
            'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
            'transistor', 'wood', 'zipper'
        ]
    elif dataset == 'mvtec_ac':
        pred_path = (
            Path.cwd()
            / 'configs'
            / 'predictions'
            / f"mvtec_ac_binary_preds_{model_identifier}.json"
        )
        object_categories = [
            'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
            'leather', 'metal_nut', 'pill', 'screw', 'tile', 'transistor',
            'wood', 'zipper'
        ]
    elif dataset == 'visa_ac':
        pred_path = (
            Path.cwd()
            / 'configs'
            / 'predictions'
            / f"visa_ac_binary_preds_{model_identifier}.json"
        )
        object_categories = [
            'candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
            'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return pred_path, object_categories

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for binary anomaly detection evaluation.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Evaluate binary anomaly detection predictions.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['mvtec_ad', 'mvtec_ac', 'visa_ac'],
        default='mvtec_ac',
        help='Dataset to evaluate.'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['gpt-4o', 'gpt-4o-mini', 'qwen', 'llama', 'llava', 'gemma'],
        default='gpt-4o',
        help='Model type used for predictions.'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Optional path to save the evaluation results as JSON.'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging.'
    )
    parser.add_argument(
        '--eval_mode',
        type=str,
        choices=['image', 'pixel', 'both'],
        default='both',
        help='Do the full pipeline with pixel-level AD'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default=None,
        help='Override the dataset root directory'
    )
    return parser.parse_args()

def main() -> None:
    """Entry point for anomaly detection evaluation."""
    args = parse_arguments()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    try:
        logger.info(f"Starting evaluation for dataset: {args.dataset}, model: {args.model}")
        predictions_path, object_categories = get_dataset_config(args.dataset, args.model)

        results_to_save: Dict[str, Any] = {}

        # --- image-level evaluation (existing path) ---
        if args.eval_mode in ('image', 'both'):
            img_all: Dict[str, Any] = {}
            for category in object_categories:
                logger.info(f"[image] Evaluating category: {category}")
                img_all[category] = evaluate_predictions(predictions_path, category)
            # overall (micro-avg across cats)
            accuracies = [res['accuracy'] for res in img_all.values()]
            precisions = [res['precision'] for res in img_all.values()]
            recalls = [res['recall'] for res in img_all.values()]
            f1s = [res['f1'] for res in img_all.values()]
            img_overall = {
                'accuracy': float(np.mean(accuracies)) if accuracies else 0.0,
                'precision': float(np.mean(precisions)) if precisions else 0.0,
                'recall': float(np.mean(recalls)) if recalls else 0.0,
                'f1': float(np.mean(f1s)) if f1s else 0.0,
            }
            img_all['overall'] = img_overall
            results_to_save['image_level'] = img_all

        # --- pixel-level evaluation (new) ---
        if args.eval_mode in ('pixel', 'both'):
            data_root = Path(args.data_root) if args.data_root else None
            logger.info("[pixel] Evaluating pixel-level masks")
            px = evaluate_pixel_masks(predictions_path, args.dataset, data_root=data_root)
            results_to_save['pixel_level'] = px

        # output
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results_to_save, f, indent=4)
            logger.info(f"Results saved to {args.output}")
        else:
            print(json.dumps(results_to_save, indent=4))

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == '__main__':
    main()
