from .base_manager import BaseManager

from models.grounding_dino import GroundingDINO
from models.sam2 import SAM2
from models.depth_anything import DepthAnything
from models.florence2 import Florence2

from utils.prompt import DEFAULT_DETECTION_PROMPT


class PipelineManager(BaseManager):

    def __init__(self):

        self.detector = GroundingDINO()
        self.segmenter = SAM2()
        self.depth = DepthAnything()
        self.florence = Florence2()

    def initialize(self):

        self.detector.load()
        self.segmenter.load()
        self.depth.load()
        self.florence.load()

    def detect(self, image):

        return self.detector.detect(
            image,
            DEFAULT_DETECTION_PROMPT,
        )

    def segment(self, image, detections):

        return self.segmenter.segment(
            image,
            detections,
        )
    
    def estimate_depth(self, image):
     return self.depth.estimate(image)

    
    def caption(self, image):
     return self.florence.caption(image)