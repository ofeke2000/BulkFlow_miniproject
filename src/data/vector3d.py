from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Vector3D:
    x: float
    y: float
    z: float

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __getitem__(self, index: int) -> float:
        return (self.x, self.y, self.z)[index]

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray((self.x, self.y, self.z), dtype=dtype)

    def to_array(self) -> np.ndarray:
        return np.array((self.x, self.y, self.z), dtype=float)

    def norm(self) -> float:
        return float(np.linalg.norm(self.to_array()))

    def __add__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self) -> Vector3D:
        return Vector3D(-self.x, -self.y, -self.z)

    @classmethod
    def from_sequence(cls, values) -> Vector3D:
        values = tuple(float(v) for v in values)
        if len(values) != 3:
            raise ValueError("Vector3D requires three components")
        return cls(*values)

    def periodic_delta(self, other: Vector3D, box_size: float) -> Vector3D:
        delta = np.array(self.to_array()) - np.array(other.to_array())
        delta -= box_size * np.round(delta / box_size)
        return Vector3D(*delta)

    def periodic_distance_to(self, other: Vector3D, box_size: float) -> float:
        return float(np.linalg.norm(self.periodic_delta(other, box_size).to_array()))
