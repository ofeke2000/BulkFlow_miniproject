from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .vector3d import Vector3D


@dataclass
class BulkFlowResult:
    origin_id: int
    mask: str
    origin: Vector3D
    radii: np.ndarray
    u_x: np.ndarray
    u_y: np.ndarray
    u_z: np.ndarray
    U: np.ndarray

    def __post_init__(self):
        self.radii = np.asarray(self.radii, dtype=float)
        self.u_x = np.asarray(self.u_x, dtype=float)
        self.u_y = np.asarray(self.u_y, dtype=float)
        self.u_z = np.asarray(self.u_z, dtype=float)
        self.U = np.asarray(self.U, dtype=float)

        if self.origin is None:
            raise ValueError("BulkFlowResult requires a valid origin")
        if not self.radii.shape == self.u_x.shape == self.u_y.shape == self.u_z.shape == self.U.shape:
            raise ValueError("BulkFlowResult arrays must all have the same shape")
