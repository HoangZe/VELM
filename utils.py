import os
import json
import logging
from pathlib import Path
import yaml
from typing import Dict, List, Union, Any
import numpy as np
import re, json

# module logger used by helpers (e.g., load_config)
logger = logging.getLogger(__name__)

def parse_llm_json(raw_text: str):
    """
    Parse a single JSON object from an LLM response.
    Tolerates common formatting artifacts (fences, trailing commas),
    repairs frequent bbox ']'→'}' typos, and handles:
      - orphan numerics inside point objects ({"x": 644, 528, "y": 498})
      - regions that leaked outside the "regions" array
      - premature top-level close before "confidence"
    Falls back to parsing the first balanced top-level JSON object if needed.
    """
    s = raw_text.strip()

    # strip code fences if present
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)

    # strip any non-JSON prefix (e.g., 'name: {...}')
    first = s.find("{")
    if first != -1:
        s = s[first:]

    # keep the widest {...} block (pre-repair)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        s = m.group(0)

    # ---- repairs (order matters) ----

    # (A) orphan numeric in point objects:
    # {"x": A, B, "y": C}  -> {"x": A, "y": C}
    s = re.sub(
        r'("x"\s*:\s*-?\d+(?:\.\d+)?),\s*-?\d+(?:\.\d+)?\s*,\s*("y"\s*:)',
        r'\1, \2',
        s
    )
    # {"y": A, B, "x": C}  -> {"y": A, "x": C}
    s = re.sub(
        r'("y"\s*:\s*-?\d+(?:\.\d+)?),\s*-?\d+(?:\.\d+)?\s*,\s*("x"\s*:)',
        r'\1, \2',
        s
    )

    # remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # repair frequent bbox bracket typo: ... "bbox": [ ... }  ->  ... "bbox": [ ... ] }
    s = re.sub(r'("bbox"\s*:\s*\[[^\]]*?)(\})', r'\1]\2', s)

    # (B) regions leaked outside the "regions" array (premature close, then another region object)
    #   ... "regions": [ {...} ] } , { "points_positive": ... }  -->  ... "regions": [ {...}, { "points_positive": ... } ]
    s = re.sub(
        r'\]\s*}\s*,\s*{\s*"points_(positive|negative)"',
        r'], {"points_\1"',
        s
    )
    # If a region started with bbox (rare), handle that too
    s = re.sub(
        r'\]\s*}\s*,\s*{\s*"bbox"',
        r'], {"bbox"',
        s
    )

    # (C) premature top-level close before "confidence":  ] } , "confidence": ... }
    # turn into:  ], "confidence": ... }
    s = re.sub(
        r'}\s*,\s*("confidence"\s*:)',
        r', \1',
        s
    )

    # ---- try strict load; on failure, fall back to first balanced object ----
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        # cut the first balanced top-level object
        depth = 0
        in_str = False
        esc = False
        start = s.find("{")
        if start == -1:
            raise
        end = None
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
        if end is None:
            raise
        s_balanced = s[start:end]
        obj = json.loads(s_balanced)

    # ---- lightweight schema checks ----
    if "label" not in obj:
        raise ValueError("missing 'label'")
    if obj.get("label") == "anomalous":
        if "regions" not in obj or not isinstance(obj["regions"], list) or not obj["regions"]:
            raise ValueError("missing/empty 'regions'")
        r0 = obj["regions"][0]
        if "bbox" not in r0 or not isinstance(r0["bbox"], list) or len(r0["bbox"]) != 4:
            raise ValueError("invalid 'bbox'")

    return obj

def scale_points_norm_to_px(points: List[Dict], W: int, H: int, cap: int=10, allow_empty: bool=False) -> np.ndarray:
    if (not points) and allow_empty:
        return np.zeros((0,2), dtype=np.float32)
    pts = []
    for p in points[:cap]:
        x = float(p["x"]) * W
        y = float(p["y"]) * H
        pts.append([x, y])
    if not pts and not allow_empty:
        raise ValueError("No points provided.")
    return np.array(pts, dtype=np.float32)

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

def get_save_path(dataset: str, model_type: str) -> Path:
    model_identifier = model_type.replace('-', '_')
    return (Path.cwd() / 'configs' / 'predictions'
            / f"{dataset}_binary_preds_{model_identifier}.json")

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