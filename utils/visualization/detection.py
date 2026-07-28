"""
Visualization utilities for object detection.
"""

from PIL import Image, ImageDraw, ImageFont

from utils.visualization.colors import get_color


class DetectionVisualizer:
    """
    Draw Grounding DINO detections.
    """

    def __init__(self):

        try:
            # Slightly larger font
            self.font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            self.font = ImageFont.load_default()

    def draw(self, image, detection_result):

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.copy()
        draw = ImageDraw.Draw(image)

        img_w, img_h = image.size

        for box, score, label in zip(
            detection_result.boxes,
            detection_result.scores,
            detection_result.labels,
        ):

            if hasattr(box, "cpu"):
                box = box.cpu().tolist()

            x1, y1, x2, y2 = map(int, box)

            color = get_color(label)

            # -----------------------------
            # Bounding Box
            # -----------------------------
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=color,
                width=3,
            )

            # Cleaner label
            text = f"{label.title()} {score:.0%}"

            # -----------------------------
            # Text Size
            # -----------------------------
            try:
                bbox = draw.textbbox((0, 0), text, font=self.font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except AttributeError:
                text_w, text_h = draw.textsize(
                    text,
                    font=self.font,
                )

            padding = 6

            # Draw above box if possible,
            # otherwise inside the box.
            label_y = y1 - text_h - padding * 2

            if label_y < 0:
                label_y = y1 + 2

            # Keep inside image width
            if x1 + text_w + padding * 2 > img_w:
                x1 = img_w - text_w - padding * 2

            # -----------------------------
            # Background
            # -----------------------------
            draw.rectangle(
                (
                    x1,
                    label_y,
                    x1 + text_w + padding * 2,
                    label_y + text_h + padding * 2,
                ),
                fill=color,
            )

            # -----------------------------
            # Text
            # -----------------------------
            draw.text(
                (
                    x1 + padding,
                    label_y + padding,
                ),
                text,
                fill="white",
                font=self.font,
            )

        return image