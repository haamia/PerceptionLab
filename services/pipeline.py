"""
pipeline.py

Central perception pipeline.
"""

from models.segmentation import segment
from models.depth import estimate_depth
from models.caption import generate_caption
from models.scene_graph import generate_scene_graph
from models.vqa import answer_question
from utils.logger import logger
from models.grounding_dino import GroundingDINO
from utils.prompt import DEFAULT_DETECTION_PROMPT

detector = GroundingDINO()


def run_pipeline(image):

    logger.info("Pipeline started.")

    detections = detector.detect(
        image,
        DEFAULT_DETECTION_PROMPT,
    )

    labels = "\n".join(str(label) for label in detections["labels"])

    logger.info("Pipeline completed.")

    return (
        image,
        labels,
        None,
        None,
        None,
        None,
        None,
        "Detection completed.",
    )
    logger.info("Pipeline started.")
    """
    Executes the perception pipeline.

    Future pipeline:

    Image
        ↓
    Detection
        ↓
    Segmentation
        ↓
    Depth
        ↓
    Caption
        ↓
    Scene Graph
        ↓
    VQA
    """

    detections = detector.detect(
    image,
    DEFAULT_DETECTION_PROMPT,)

    labels = "\n".join(str(label) for label in detections["labels"])

    segmentation = segment(image)

    depth = estimate_depth(image)

    caption = generate_caption(image)

    scene = generate_scene_graph(image)

    vqa = answer_question(image)

    benchmark = "Pipeline executed successfully."
    
    logger.info("Pipeline completed.")

    return (
        image,
        labels,
        segmentation,
        depth,
        scene,
        caption,
        vqa,
        benchmark,
    )