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
            # Smaller font for a cleaner appearance
            self.font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            self.font = ImageFont.load_default()

    def draw(self, image, detection_result):

        # Convert NumPy array to PIL Image if necessary
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.copy()
        draw = ImageDraw.Draw(image)

        for idx, (box, score, label) in enumerate(
         zip(
              detection_result.boxes,
             detection_result.scores,
             detection_result.labels,
        ),
        start=1,
       ):

            # Convert tensor → list if necessary
            if hasattr(box, "cpu"):
                box = box.cpu().tolist()

            x1, y1, x2, y2 = box

            color = get_color(label)

            # ----------------------------------
            # Draw Bounding Box
            # ----------------------------------
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=color,
                width=2,
            )

            text = f"#{idx} {label.title()} ({score:.2f})"

            # ----------------------------------
            # Calculate text size
            # ----------------------------------
            try:
                bbox = draw.textbbox((0, 0), text, font=self.font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                text_width, text_height = draw.textsize(
                    text,
                    font=self.font,
                )

            # ----------------------------------
            # Keep label inside image
            # ----------------------------------
            label_y = max(0, y1 - text_height - 4)

            # ----------------------------------
            # Draw label background
            # ----------------------------------
            draw.rectangle(
                [
                    (x1, label_y),
                    (x1 + text_width + 4, label_y + text_height + 4),
                ],
                fill=color,
            )

            # ----------------------------------
            # Draw label text
            # ----------------------------------
            draw.text(
                (x1 + 2, label_y + 2),
                text,
                fill="white",
                font=self.font,
            )

        return image