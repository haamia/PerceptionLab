"""
Complete perception visualizer.

Combines segmentation masks and object detections.
"""

from utils.visualization.segmentation import draw_segmentation
from utils.visualization.detection import DetectionVisualizer
from collections import Counter
from PIL import ImageDraw, ImageFont


class PerceptionVisualizer:

    def __init__(self):

        self.detector = DetectionVisualizer()

    def draw(
        self,
        image,
        segmentation_result,
    ):

        image = draw_segmentation(
            image,
            segmentation_result,
        )

        image = self.detector.draw(
            image,
            segmentation_result.detections,
        )

        return image