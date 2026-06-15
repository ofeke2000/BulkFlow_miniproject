from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MDPL2Config:
    """Fixed simulation parameters - not user-editable."""
    HubbleParameter: float = 0.6777
    box_size: float = 1000.0
