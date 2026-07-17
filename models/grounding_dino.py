"""
Grounding DINO wrapper for PerceptionLab.
"""

import torch

from PIL import Image

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

        print("Loading processor...")

        self.processor = AutoProcessor.from_pretrained(
            GROUNDING_DINO_MODEL
        )

        print("Processor loaded.")

        print("Loading model...")

        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_DINO_MODEL
        )

        print("Model loaded.")

        self.model.to(DEVICE)

        print("Grounding DINO loaded.")

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

        return results[0]

    def predict(self, image):

        return self.detect(
            image,
            "person . car . chair . dog . cat ."
        )