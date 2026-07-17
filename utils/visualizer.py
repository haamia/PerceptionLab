"""
Visualization utilities.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_boxes(image, detections):

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.imshow(image)

    for box, score, label in zip(
        detections["boxes"],
        detections["scores"],
        detections["labels"],
    ):

        x1, y1, x2, y2 = box.tolist()

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
        )

        ax.add_patch(rect)

        ax.text(
            x1,
            y1,
            f"{label} ({score:.2f})",
        )

    plt.axis("off")

    return fig