from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np


class BCType(Enum):
    DIRICHLET = auto()  # u = value
    NEUMANN = auto()  # du/dx = 0


@dataclass
class BoundaryCondition:
    bc_type: BCType
    applies_near: tuple[float | None, ...]
    value: float | None = field(default=None)

    @classmethod
    def from_str(
        cls,
        bc_str: str,
        applies_near: tuple[float | None, ...],
        value: float | None = None,
    ):
        if bc_str == "dirichlet":
            type_ = BCType.DIRICHLET
        elif bc_str == "neumann":
            type_ = BCType.NEUMANN
        else:
            raise ValueError(f"Invalid boundary condition type: {bc_str}")
        return cls(type_, applies_near, value)

    def applies_mask(self, mesh: np.ndarray[float]):
        mask = np.zeros(mesh.shape[1], dtype=bool)
        for idx, val_dim in enumerate(self.applies_near):
            if val_dim is None:
                continue
            mask = np.logical_or(mask, np.isclose(mesh[idx], val_dim))

        return mask


@dataclass
class EdgeBoundaryConditions:
    left: BoundaryCondition
    right: BoundaryCondition
    bottom: BoundaryCondition
    top: BoundaryCondition

    def all_bcs(self):
        return self.__dict__.items()
