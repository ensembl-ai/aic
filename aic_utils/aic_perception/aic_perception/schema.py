"""Versioned board schema: the stable mapping from model queries to target IDs.

The planner only ever sees target IDs such as ``nic_card_mount_3/sfp_port_1``.
Query indices are an implementation detail owned by this schema, so a new
board revision is a new schema name with a new target list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Target:
    """One named estimation target.

    Attributes:
        id: Stable semantic ID shared with the task planner.
        type: Target category, such as ``sfp_port``.
        frame: Ground-truth TF frame published by the simulator.
    """

    id: str
    type: str
    frame: str


@dataclass(frozen=True)
class BoardSchema:
    """Ordered targets and cameras for one board revision.

    Attributes:
        name: Schema version, such as ``aic_v1``.
        base_frame: Robot frame every pose is expressed in.
        cameras: Camera names in model input order.
        target_types: Target categories in type-index order.
        targets: Targets in model query order.
    """

    name: str
    base_frame: str
    cameras: tuple[str, ...]
    target_types: tuple[str, ...]
    targets: tuple[Target, ...]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BoardSchema:
        """Build the schema from a validated configuration.

        Args:
            config: Configuration containing the ``schema`` section.

        Returns:
            Immutable board schema.
        """

        schema = config["schema"]
        return cls(
            name=schema["name"],
            base_frame=schema["base_frame"],
            cameras=tuple(schema["cameras"]),
            target_types=tuple(schema["target_types"]),
            targets=tuple(Target(**target) for target in schema["targets"]),
        )

    @property
    def ids(self) -> list[str]:
        """Target IDs in query order."""

        return [target.id for target in self.targets]

    @property
    def type_indices(self) -> list[int]:
        """Target type index for each query."""

        return [self.target_types.index(target.type) for target in self.targets]
