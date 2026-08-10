"""Selected-world geometry visualization."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import warp as wp

from aic_mujoco.runtime import AICWarpRuntime
from aic_mujoco.utils.arrays import gather_mat33, gather_vec3


class RuntimeOutputs:
    """Bridge explicitly selected device worlds to Viser."""

    def __init__(self, config: dict[str, Any], runtime: AICWarpRuntime):
        """Initialize selected-world geometry visualization.

        Args:
            config: Strict merged runtime configuration.
            runtime: Initialized MJWarp runtime.
        """

        self.config = config
        self.server = None
        self.mesh_handles: dict[tuple[int, int], Any] = {}

        visualization_enabled = config["visualization"]["enabled"]
        visualization_selection = config["visualization"]["env_ids"]
        self.visual_envs = (
            list(range(runtime.num_envs))
            if visualization_enabled and visualization_selection == "all"
            else list(visualization_selection)
            if visualization_enabled
            else []
        )
        self.visual_ids = self.device_ids(self.visual_envs, runtime)
        self.visual_geom_ids = (
            np.flatnonzero(runtime.host_model.geom_matid >= 0).tolist()
            if visualization_enabled
            else []
        )
        self.geom_positions: Any = None
        self.geom_orientations: Any = None
        if visualization_enabled:
            self.allocate_visual_poses(runtime)

        if visualization_enabled:
            self.start_visualization(runtime)

    @staticmethod
    def device_ids(env_ids: list[int], runtime: AICWarpRuntime) -> Any:
        """Upload selected environment IDs to the runtime device.

        Args:
            env_ids: Ordered environment indices.
            runtime: Runtime whose device receives the array.

        Returns:
            Device index array, or ``None`` for an empty selection.
        """

        if not env_ids:
            return None
        return wp.array(env_ids, dtype=int, device=runtime.device)

    def allocate_visual_poses(self, runtime: AICWarpRuntime) -> None:
        """Allocate geometry pose buffers for visualized worlds.

        Args:
            runtime: Runtime providing geometry count and device ownership.
        """

        count = len(self.visual_envs)
        self.geom_positions = wp.empty(
            (count, runtime.host_model.ngeom), dtype=wp.vec3, device=runtime.device
        )
        self.geom_orientations = wp.empty(
            (count, runtime.host_model.ngeom), dtype=wp.mat33, device=runtime.device
        )

    def download_visual_poses(
        self, runtime: AICWarpRuntime
    ) -> tuple[np.ndarray, np.ndarray]:
        """Download geometry positions and rotations for visualized worlds.

        Args:
            runtime: Runtime containing current geometry transforms.

        Returns:
            Batched host geometry positions and rotation matrices.
        """

        wp.launch(
            gather_vec3,
            dim=(len(self.visual_envs), runtime.host_model.ngeom),
            inputs=[runtime.data.geom_xpos, self.visual_ids],
            outputs=[self.geom_positions],
            device=runtime.device,
        )
        wp.launch(
            gather_mat33,
            dim=(len(self.visual_envs), runtime.host_model.ngeom),
            inputs=[runtime.data.geom_xmat, self.visual_ids],
            outputs=[self.geom_orientations],
            device=runtime.device,
        )
        return (
            self.geom_positions.numpy(),
            self.geom_orientations.numpy(),
        )

    @staticmethod
    def mesh_color(
        model: Any, geom_id: int
    ) -> tuple[tuple[int, int, int], float | None]:
        """Convert one MuJoCo material color for Viser.

        Args:
            model: Compiled host MuJoCo model.
            geom_id: Geometry whose material is selected.

        Returns:
            Integer RGB color and optional opacity.

        Raises:
            ValueError: If the geometry has no material.
        """

        material_id = int(model.geom_matid[geom_id])
        if material_id < 0:
            raise ValueError(f"Viser geometry {geom_id} has no material")
        rgba = model.mat_rgba[material_id]
        color = (
            int(np.clip(round(float(rgba[0]) * 255.0), 0, 255)),
            int(np.clip(round(float(rgba[1]) * 255.0), 0, 255)),
            int(np.clip(round(float(rgba[2]) * 255.0), 0, 255)),
        )
        opacity = float(rgba[3]) if float(rgba[3]) < 1.0 else None
        return color, opacity

    @staticmethod
    def wxyz(rotation: np.ndarray) -> np.ndarray:
        """Convert a rotation matrix to Viser's required WXYZ boundary format.

        Args:
            rotation: Three-by-three rotation matrix.

        Returns:
            Four-component Viser WXYZ orientation.
        """

        from viser.transforms import SO3

        return SO3.from_matrix(rotation).wxyz

    def start_visualization(self, runtime: AICWarpRuntime) -> None:
        """Create the Viser geometry grid.

        Args:
            runtime: Runtime providing scene meshes and geometry poses.
        """

        import viser

        visual = self.config["visualization"]
        model = runtime.host_model
        geom_positions, geom_orientations = self.download_visual_poses(runtime)
        self.server = viser.ViserServer(
            host=visual["host"], port=visual["port"], label="AIC MuJoCo-Warp"
        )
        self.server.initial_camera.position = tuple(visual["initial_camera_position"])
        self.server.initial_camera.look_at = tuple(visual["initial_camera_look_at"])

        grid_spacing = np.asarray(visual["grid_spacing"], dtype=np.float64)
        grid_columns = math.ceil(math.sqrt(len(self.visual_envs)))
        for local_index, env_id in enumerate(self.visual_envs):
            env_root = f"/environments/{env_id}"
            column = local_index % grid_columns
            row = local_index // grid_columns
            self.server.scene.add_frame(
                env_root,
                show_axes=False,
                position=(
                    column * grid_spacing[0],
                    row * grid_spacing[1],
                    0.0,
                ),
            )
            for geom_id in self.visual_geom_ids:
                mesh_id = int(model.geom_dataid[geom_id])
                if mesh_id < 0:
                    raise ValueError(
                        f"Viser geometry {geom_id} is not a compiled mesh"
                    )
                vertex_start = int(model.mesh_vertadr[mesh_id])
                vertex_end = vertex_start + int(model.mesh_vertnum[mesh_id])
                face_start = int(model.mesh_faceadr[mesh_id])
                face_end = face_start + int(model.mesh_facenum[mesh_id])
                color, opacity = self.mesh_color(model, geom_id)
                self.mesh_handles[(env_id, geom_id)] = self.server.scene.add_mesh_simple(
                    f"{env_root}/geometry/{geom_id:04d}",
                    vertices=np.asarray(
                        model.mesh_vert[vertex_start:vertex_end], dtype=np.float32
                    ).copy(),
                    faces=np.asarray(
                        model.mesh_face[face_start:face_end], dtype=np.uint32
                    ).copy(),
                    color=color,
                    opacity=opacity,
                    side="double",
                    position=geom_positions[local_index, geom_id],
                    wxyz=self.wxyz(geom_orientations[local_index, geom_id]),
                )

        print(f"Viewer: http://{visual['host']}:{visual['port']}")

    def update_visual_poses(self, runtime: AICWarpRuntime) -> None:
        """Update every Viser mesh from current MJWarp transforms.

        Args:
            runtime: Runtime containing current geometry transforms.
        """

        geom_positions, geom_orientations = self.download_visual_poses(runtime)
        for local_index, env_id in enumerate(self.visual_envs):
            for geom_id in self.visual_geom_ids:
                handle = self.mesh_handles[(env_id, geom_id)]
                handle.position = geom_positions[local_index, geom_id]
                handle.wxyz = self.wxyz(geom_orientations[local_index, geom_id])

    def update(self, runtime: AICWarpRuntime) -> None:
        """Publish current geometry poses.

        Args:
            runtime: Runtime containing current device observations and poses.
        """

        if self.server is None:
            return
        self.update_visual_poses(runtime)

    def close(self) -> None:
        """Stop the Viser server."""

        if self.server is not None:
            self.server.stop()
