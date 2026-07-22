"""
Base manager for all perception modules.
"""


class BaseManager:

    def initialize(self):
        """Initialize required models."""
        raise NotImplementedError