"""
Visualization utilities for segmentation.
"""

import random
import numpy as np
from PIL import Image


def draw_segmentation(image, segmentation_result, alpha=0.5):
    """
    Draw segmentation masks on an image.
    """

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    output = image.copy()

    if segmentation_result is None:
        return Image.fromarray(output)

    if len(segmentation_result.masks) == 0:
        return Image.fromarray(output)

    for mask in segmentation_result.masks:

        # SAM2 returns (1, H, W)
        if mask.ndim == 3:
            mask = mask[0]

        # Convert probability mask to boolean
        mask = mask > 0.5

        color = np.array(
            [
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            ],
            dtype=np.uint8,
        )

        output[mask] = (
            alpha * color
            + (1 - alpha) * output[mask]
        ).astype(np.uint8)

    return Image.fromarray(output)