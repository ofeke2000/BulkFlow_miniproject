import numpy as np
from typing import Optional


class BulkFlowResult:
    """
    Container for bulk flow results at a given origin and mask.
    """

    def __init__(
        self,
        origin_id: int,
        mask: str,
        origin: np.ndarray,
        radii: np.ndarray,
        u_x: np.ndarray,
        u_y: np.ndarray,
        u_z: np.ndarray,
        U: np.ndarray,
    ):
        self.origin_id = origin_id
        self.mask = mask
        self.origin = np.asarray(origin, dtype=float)

        self.radii = np.asarray(radii, dtype=float)
        self.u_x = np.asarray(u_x, dtype=float)
        self.u_y = np.asarray(u_y, dtype=float)
        self.u_z = np.asarray(u_z, dtype=float)
        self.U = np.asarray(U, dtype=float)

        self._validate_shapes()
