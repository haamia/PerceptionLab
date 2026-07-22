"""
Consistent colors for visualization.
"""

CLASS_COLORS = {
    "person": (0, 114, 189),
    "cat": (0, 180, 255),
    "dog": (255, 140, 0),
    "couch": (0, 200, 0),
    "chair": (50, 205, 50),
    "car": (255, 0, 0),
    "remote control": (255, 215, 0),
    "tv": (128, 0, 255),
    "laptop": (255, 0, 255),
}


DEFAULT_COLOR = (0, 255, 255)


def get_color(label):
    return CLASS_COLORS.get(label, DEFAULT_COLOR)