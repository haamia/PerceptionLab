"""
pipeline.py

Central perception pipeline.
"""

from models.detector import detect
from models.segmentation import segment
from models.depth import estimate_depth
from models.caption import generate_caption
from models.scene_graph import generate_scene_graph
from models.vqa import answer_question


def run_pipeline(image):
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

    detection = detect(image)

    segmentation = segment(image)

    depth = estimate_depth(image)

    caption = generate_caption(image)

    scene = generate_scene_graph(image)

    vqa = answer_question(image)

    benchmark = "Pipeline executed successfully."

    return (
        image,
        detection,
        segmentation,
        depth,
        scene,
        caption,
        vqa,
        benchmark,
    )