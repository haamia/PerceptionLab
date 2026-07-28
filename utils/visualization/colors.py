"""
Professional color palette for visualization.
Inspired by Ultralytics YOLO.
"""

COLORS = [
    (255, 56, 56),      # Red
    (255, 157, 0),      # Orange
    (255, 112, 31),     # Dark Orange
    (255, 178, 29),     # Yellow
    (207, 210, 49),     # Lime
    (72, 249, 10),      # Green
    (146, 204, 23),     # Olive
    (61, 219, 134),     # Emerald
    (26, 147, 52),      # Dark Green
    (0, 212, 187),      # Cyan
    (44, 153, 168),     # Teal
    (0, 194, 255),      # Sky Blue
    (52, 69, 147),      # Blue
    (100, 115, 255),    # Indigo
    (0, 24, 236),       # Deep Blue
    (132, 56, 255),     # Purple
    (203, 56, 255),     # Magenta
    (255, 149, 200),    # Pink
]


def get_color(label: str):
    """
    Returns a consistent color for each class.
    """

    idx = abs(hash(label.lower())) % len(COLORS)

    return COLORS[idx]