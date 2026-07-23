"""
results.py

Standard result objects used throughout PerceptionLab.
"""

from dataclasses import dataclass
from typing import Any, List
import torch


@dataclass
class DetectionResult:
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: List[str]

    @property
    def num_objects(self):
        return len(self.labels)

    def summary(self):
        if self.num_objects == 0:
            return "No objects detected."

        return "\n".join(
            f"{i+1}. {label} ({score:.2f})"
            for i, (label, score) in enumerate(zip(self.labels, self.scores))
        )


@dataclass
class SegmentationResult:
    """
    Output of an image segmentation model.
    """

    masks: Any
    scores: Any
    detections: DetectionResult

    @property
    def num_masks(self):
        return len(self.masks)

    def summary(self):

        if self.num_masks == 0:
            return "No masks generated."

        return f"{self.num_masks} mask(s)"


@dataclass
class DepthResult:
    """
    Stores the predicted depth map.
    """

    depth_map: Any

    @property
    def is_empty(self):
        return self.depth_map is None

    def summary(self):
        if self.is_empty:
            return "No depth map generated."

        return "Depth map generated successfully."

@dataclass
class CaptionResult:
    """
    Stores image caption.
    """

    caption: str

    @property
    def is_empty(self):
        return self.caption is None or self.caption == ""

    def summary(self):

        if self.is_empty:
            return "No caption generated."

        return self.caption