"""
logger.py

Logging utilities for PerceptionLab.
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("PerceptionLab")