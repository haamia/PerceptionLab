"""
florence2.py

Microsoft Florence-2 wrapper.
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM

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

    def caption(self, image):

        self.load()

        prompt = "<CAPTION>"

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(FLORENCE_DEVICE)

        with torch.no_grad():

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                num_beams=3,
                do_sample=False,
            )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]

        parsed = self.processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=image.size,
        )

        caption = parsed.get("<CAPTION>", "").strip()

        return CaptionResult(
            caption=caption,
        )