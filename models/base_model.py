"""
Base class for all vision models.
"""

from abc import ABC, abstractmethod


class VisionModel(ABC):

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, image):
        pass