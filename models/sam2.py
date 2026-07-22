"""
SAM 2 wrapper for PerceptionLab.
"""

import numpy as np
from utils.logger import logger
from models.base_model import VisionModel
from core.results import SegmentationResult

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from config import (
    SAM2_DEVICE,
    SAM2_CHECKPOINT,
    SAM2_CONFIG,
)


class SAM2(VisionModel):

    def __init__(self):
        self.predictor = None

    def load(self):

        if self.predictor is not None:
            return

       

        model = build_sam2(
            config_file=SAM2_CONFIG,
            ckpt_path=str(SAM2_CHECKPOINT),
            device=SAM2_DEVICE,
        )

        self.predictor = SAM2ImagePredictor(model)

        logger.info("SAM 2 initialized.")

    def segment(self, image, detection_result):

        self.load()

        if detection_result is None or detection_result.num_objects == 0:
            return SegmentationResult(
                masks=[],
                scores=[],
                detections=detection_result,
            )

        # SAM2 expects a NumPy RGB image
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        self.predictor.set_image(image)

        masks = []
        scores = []

        for box in detection_result.boxes:

            # Convert tensor -> numpy if needed
            if hasattr(box, "cpu"):
                box = box.cpu().numpy()

            mask, score, _ = self.predictor.predict(
                box=box,
                multimask_output=False,
            )
            

            masks.append(mask[0])
            scores.append(float(score[0]))

        return SegmentationResult(
            masks=masks,
            scores=scores,
            detections=detection_result,
        )

    def predict(self, image):
        return self.segment(image, None)