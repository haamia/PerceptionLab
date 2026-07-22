"""
depth_anything.py

Depth Anything V2 wrapper.
"""

from transformers import AutoImageProcessor
from transformers import AutoModelForDepthEstimation

from config import DEPTH_MODEL_ID, DEPTH_DEVICE
from utils.logger import logger
import torch
from PIL import Image



from models.base_model import VisionModel
from core.results import DepthResult


class DepthAnything(VisionModel):

    def __init__(self):
        self.model = None
        self.processor = None

    def load(self):

     if self.model is not None:
        return

     self.processor = AutoImageProcessor.from_pretrained(
        DEPTH_MODEL_ID
     )

     self.model = AutoModelForDepthEstimation.from_pretrained(
        DEPTH_MODEL_ID
     )

     self.model.to(DEPTH_DEVICE)
     self.model.eval()

     logger.info("Depth Anything V2 initialized.")

    def predict(self, image):
        """
        Required by VisionModel.
        """
        return self.estimate(image)

    def estimate(self, image):

     self.load()

     if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

     inputs = self.processor(
        images=image,
        return_tensors="pt"
     )

     inputs = {
        k: v.to(DEPTH_DEVICE)
        for k, v in inputs.items()
     }

     with torch.no_grad():

        outputs = self.model(**inputs)

     predicted_depth = outputs.predicted_depth
     predicted_depth = torch.nn.functional.interpolate(
      predicted_depth.unsqueeze(1),
      size=image.size[::-1],
      mode="bicubic",
      align_corners=False,
     ).squeeze()

     depth = predicted_depth.cpu().numpy()
     depth = depth - depth.min()

     depth = depth / (depth.max() + 1e-8)
     return DepthResult(
     depth_map=depth
     )
     