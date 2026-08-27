"""Helper classes and methods for the handling of transformations in the nuScenes dataset."""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    
class Transform:
    def __init__(self, translation: np.ndarray | "torch.Tensor", rotation: np.ndarray | "torch.Tensor", timestamp: int):
        self.translation = translation
        self.rotation = rotation
        self.timestamp = timestamp

    def __call__(self, point: np.ndarray | "torch.Tensor") -> np.ndarray | "torch.Tensor":
        return self.translation + self.rotation @ point
        
    def __mul__(self, other: Transform) -> Transform:
        return Transform(self.translation + self.rotation @other.translation, self.rotation @ other.rotation, self.timestamp)

    def toMatrix(self) -> np.ndarray:
        return np.concatenate([np.concatenate([self.rotation, self.translation.reshape(-1, 1)], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

    def fromMatrix(matrix: np.ndarray, timestamp: int) -> Transform:
        return Transform(matrix[0:3, 3], matrix[0:3, 0:3], timestamp)
    
    def inverse(self) -> Transform:
        return Transform(-self.rotation.T @ self.translation, self.rotation.T, self.timestamp)

    def __truediv__(self, other: Transform) -> Transform:
        return self * other.inverse()