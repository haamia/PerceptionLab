"""
pipeline.py

Central perception pipeline.
"""

from core.managers.pipeline_manager import PipelineManager



from models.caption import generate_caption
from models.scene_graph import generate_scene_graph
from models.vqa import answer_question
import time
from collections import Counter
from utils.visualization.perception import PerceptionVisualizer
from utils.visualization.depth import DepthVisualizer
from utils.logger import logger


manager = PipelineManager()
visualizer = PerceptionVisualizer()
depth_visualizer = DepthVisualizer()


def run_pipeline(image):
    
    logger.info("Pipeline started.")
    start_time = time.perf_counter()

    # ---------------- Detection ----------------

    detections = manager.detect(image)
    detection_time = time.perf_counter()
    labels = detections.summary()
    

    # ---------------- Segmentation ----------------

    segmentations = manager.segment(
        image,
        detections,
    )
    segmentation_time = time.perf_counter()

    depth = manager.estimate_depth(image)

    depth_time = time.perf_counter()
    depth_image = depth_visualizer.draw(depth)

    caption = manager.caption(image)
    caption_time = time.perf_counter()

    result_image = visualizer.draw(
      image,
     segmentations,
   )
    
    end_time = time.perf_counter() 
   
    # ---------------- Other Modules ----------------


  

    scene = generate_scene_graph(image)

    vqa = answer_question(image)
    #------------------------------------------------

    
    label_counts = Counter(detections.labels)

    object_summary = "\n".join(
    f"{label.title()} : {count}"
    for label, count in sorted(label_counts.items())
    )
    benchmark = f"""
    Objects Detected : {detections.num_objects}
    Object Summary
    --------------
    {object_summary}

    Masks Generated : {segmentations.num_masks}

    Detection Model :
    Grounding DINO

    Segmentation Model :
    SAM2.1 Hiera Base+
    
    Depth Model :
    Depth Anything V2 Base

    Caption Model :
    Florence-2 Base

    Detection Time :
    {detection_time - start_time:.2f} sec

    Segmentation Time :
    {segmentation_time - detection_time:.2f} sec

    Depth Time :
    {depth_time - segmentation_time:.2f} sec


    Caption Time :
    {caption_time - depth_time:.2f} sec



    Total Pipeline Time :
    {end_time - start_time:.2f} sec
    """
   
    logger.info("Pipeline completed.")
    

    return (
        result_image,
        labels,
        segmentations.summary(),
        depth_image,
        scene,
        caption.caption,
        vqa,
        benchmark,
    )