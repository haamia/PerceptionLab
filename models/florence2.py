"""
florence2.py

Microsoft Florence-2 wrapper.
"""

import numpy as np
import torch

from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
)

from models.base_model import VisionModel
from core.results import CaptionResult

from config import (
    FLORENCE_MODEL,
    FLORENCE_DEVICE,
)

from utils.logger import logger


class Florence2(VisionModel):

    def __init__(self):

        self.model = None
        self.processor = None

    def load(self):

        if self.model is not None:
            return

        self.processor = AutoProcessor.from_pretrained(
            FLORENCE_MODEL,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_MODEL,
            trust_remote_code=True,
        ).to(FLORENCE_DEVICE)

        self.model.eval()

        logger.info("Florence-2 initialized.")

    def predict(self, image):

        return self.caption(image)

    ####################################################################
    # Image Conversion
    ####################################################################

    def _to_pil(self, image):

        if isinstance(image, Image.Image):
            return image

        if isinstance(image, np.ndarray):
            return Image.fromarray(image)

        raise TypeError(
            f"Unsupported image type: {type(image)}"
        )

    ####################################################################
    # Generic Florence Task Runner
    ####################################################################

    def _run_task(self, image, task):

        self.load()

        image = self._to_pil(image)

        inputs = self.processor(
            text=task,
            images=image,
            return_tensors="pt",
        ).to(FLORENCE_DEVICE)

        with torch.no_grad():

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
                do_sample=False,
            )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]

        parsed = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height),
        )

        return parsed

    ####################################################################
    # Image Captioning
    ####################################################################

    def caption(self, image):

        parsed = self._run_task(
            image,
            "<CAPTION>",
        )

        caption = parsed.get(
            "<CAPTION>",
            "",
        ).strip()

        if caption:
            caption = caption[0].upper() + caption[1:]

        return CaptionResult(
            caption=caption,
        )

    ####################################################################
    # Open Vocabulary Detection
    ####################################################################

    def detect_objects(self, image):

     parsed = self._run_task(
         image,
        "<OD>",
     )

     od = parsed.get("<OD>", {})

     labels = od.get("labels", [])

     # Remove duplicates while preserving order
     objects = list(dict.fromkeys(
        label.lower().strip()
        for label in labels
        if label.strip()
     ))

     return objects

    ####################################################################
    # Dense Region Captioning
    ####################################################################

    def dense_region_caption(self, image):

        parsed = self._run_task(
            image,
            "<DENSE_REGION_CAPTION>",
        )

        return parsed

    ####################################################################
    # OCR
    ####################################################################

    def ocr(self, image):

        parsed = self._run_task(
            image,
            "<OCR>",
        )

        return parsed

    ####################################################################
    # Region Proposal
    ####################################################################

    def region_proposal(self, image):

        parsed = self._run_task(
            image,
            "<REGION_PROPOSAL>",
        )

        return parsed