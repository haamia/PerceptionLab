"""
config.py

Global configuration for PerceptionLab.
"""

from pathlib import Path

# ==========================================================
# Project Directories
# ==========================================================

PROJECT_ROOT = Path(__file__).parent

ASSETS_DIR = PROJECT_ROOT / "assets"
IMAGES_DIR = PROJECT_ROOT / "images"
UPLOAD_DIR = IMAGES_DIR / "uploads"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ==========================================================
# Hugging Face Models
# ==========================================================

GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-base"

SEGMENTATION_MODEL = "SAM 2"

DEPTH_MODEL = "Depth Anything V2"

CAPTION_MODEL = "Florence-2"

VQA_MODEL = "Florence-2"

# ==========================================================
# Runtime
# ==========================================================

DEVICE = "cpu"

BOX_THRESHOLD = 0.35

TEXT_THRESHOLD = 0.25

# ==========================================================
# UI
# ==========================================================

APP_TITLE = "PerceptionLab"

APP_SUBTITLE = "A Modular Computer Vision Research Toolkit"