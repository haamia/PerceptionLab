"""
Grounding DINO wrapper for PerceptionLab.
"""

import torch

from PIL import Image
from core.results import DetectionResult
from utils.logger import logger


from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)

from config import (
    GROUNDING_DINO_MODEL,
    DEVICE,
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
)

from models.base_model import VisionModel


class GroundingDINO(VisionModel):

    def __init__(self):

        self.processor = None
        self.model = None

    def load(self):

        if self.processor is not None:
            return

        

        self.processor = AutoProcessor.from_pretrained(
            GROUNDING_DINO_MODEL
        )

    
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_DINO_MODEL
        )

       
        self.model.to(DEVICE)

        logger.info("Grounding DINO initialized.")

    def detect(
        self,
        image,
        prompt,
    ):

        self.load()

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(DEVICE)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = (
            self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                target_sizes=[image.size[::-1]],
            )
        )

        result=results[0]
        return DetectionResult(
         boxes=result["boxes"],
         scores=result["scores"],
         labels=result["text_labels"],)
        

    def predict(
     self,
     image,
     prompt,
    ):

     return self.detect(
        image,
        prompt,
     )