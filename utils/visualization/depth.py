"""
depth.py

Utilities for visualizing depth maps.
"""

import numpy as np
import matplotlib

from PIL import Image


class DepthVisualizer:
    """
    Converts a normalized depth map into a colorful visualization.
    """

    def draw(self, depth_result):

        if depth_result is None or depth_result.is_empty:
            return None

        depth = depth_result.depth_map

        # Clamp values to [0, 1]
        depth = np.clip(depth, 0.0, 1.0)

        # Turbo colormap (Matplotlib 3.11)
        cmap = matplotlib.colormaps["turbo"]

        colored = cmap(depth)

        # Remove alpha channel
        colored = colored[:, :, :3]

        # Convert to uint8
        colored = (colored * 255).astype(np.uint8)

        return Image.fromarray(colored)