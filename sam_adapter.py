from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image, ImageOps

import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

class Sam2Adapter:
    def __init__(self, repo_id: str = "facebook/sam2.1-hiera-large", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # HF route: downloads weights automatically if not cached
        self.predictor = SAM2ImagePredictor.from_pretrained(repo_id)
        self.image_hw: Optional[Tuple[int, int]] = None  # (H, W)

    def set_image(self, img: Image.Image | np.ndarray) -> None:
        if isinstance(img, Image.Image):
            img = ImageOps.exif_transpose(img).convert("RGB")  # EXIF orientation fix
            img = np.array(img)
        assert img.ndim == 3 and img.shape[2] == 3, "Expect HxWx3 image"
        self.image_hw = (img.shape[0], img.shape[1])
        with torch.inference_mode():
            self.predictor.set_image(img)

    def predict_region(
        self,
        pos_pts_px: np.ndarray,                    # N×2 (x,y) pixels
        neg_pts_px: Optional[np.ndarray] = None,  # M×2 (x,y) pixels
        box_px: Optional[np.ndarray] = None,      # [x0,y0,x1,y1] in pixels
        multimask_output: bool = True
    ):
        neg_pts_px = neg_pts_px if neg_pts_px is not None else np.zeros((0,2), dtype=np.float32)
        coords = np.vstack([pos_pts_px, neg_pts_px]).astype(np.float32)
        labels = np.concatenate([
            np.ones(len(pos_pts_px), dtype=np.int32),   # 1 = positive
            np.zeros(len(neg_pts_px), dtype=np.int32)   # 0 = negative
        ])
        with torch.inference_mode(), torch.autocast(
            "cuda" if self.device == "cuda" else "cpu", dtype=torch.bfloat16, enabled=(self.device=="cuda")
        ):
            masks, scores, logits = self.predictor.predict(
                point_coords=coords if len(coords) else None,
                point_labels=labels if len(coords) else None,
                box=box_px.astype(np.float32) if box_px is not None else None,
                multimask_output=multimask_output
            )
        # masks: (K,H,W) bool; scores: (K,)
        return masks, scores, logits

    @staticmethod
    def union_masks(mask_list: List[np.ndarray]) -> np.ndarray:
        if not mask_list:
            raise ValueError("No masks to union.")
        acc = np.zeros_like(mask_list[0], dtype=bool)
        for m in mask_list:
            acc |= m.astype(bool)
        return (acc.astype(np.uint8) * 255)

    @staticmethod
    def save_mask(mask: np.ndarray, path: Path, hw_expected: Optional[tuple[int,int]] = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        m = mask
        if hw_expected is not None and m.shape != hw_expected:
            m = np.array(Image.fromarray(m).resize((hw_expected[1], hw_expected[0]), resample=Image.NEAREST))
        Image.fromarray(m).save(path)

    @staticmethod
    def save_overlay(rgb: Image.Image | np.ndarray, mask_bin: np.ndarray, path: Path, alpha: float=0.45) -> None:
        if isinstance(rgb, Image.Image): rgb = np.array(rgb.convert("RGB"))
        path.parent.mkdir(parents=True, exist_ok=True)
        color = np.zeros_like(rgb); color[..., 0] = 255  # red overlay
        overlay = (rgb * (1 - alpha) + color * (alpha * (mask_bin > 0)[..., None])).astype(np.uint8)
        Image.fromarray(overlay).save(path)
