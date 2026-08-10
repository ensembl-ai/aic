"""Selected-world Viser visualization and RGB recording."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from aic_mujoco.runtime import AICWarpRuntime


@wp.kernel
def _gather_rgb(
    source: wp.array4d(dtype=wp.uint8),
    env_ids: wp.array(dtype=int),
    output: wp.array4d(dtype=wp.uint8),
):
    selected, row, column = wp.tid()
    source_env = env_ids[selected]
    for channel in range(3):
        output[selected, row, column, channel] = source[
            source_env, row, column, channel
        ]


@wp.kernel
def _gather_vec3(
    source: wp.array2d(dtype=wp.vec3),
    env_ids: wp.array(dtype=int),
    output: wp.array2d(dtype=wp.vec3),
):
    selected, item = wp.tid()
    output[selected, item] = source[env_ids[selected], item]


@wp.kernel
def _gather_mat33(
    source: wp.array2d(dtype=wp.mat33),
    env_ids: wp.array(dtype=int),
    output: wp.array2d(dtype=wp.mat33),
):
    selected, item = wp.tid()
    output[selected, item] = source[env_ids[selected], item]


class RuntimeOutputs:
    """Bridge explicitly selected device worlds to human-facing outputs."""

    def __init__(self, config: dict[str, Any], runtime: AICWarpRuntime):
        self._config = config
        self._server = None
        self._image_handles: dict[tuple[int, str], Any] = {}
        self._mesh_handles: dict[tuple[int, int], Any] = {}
        self._writers: dict[tuple[int, str], Any] = {}

        visualization_enabled = config["visualization"]["enabled"]
        visualization_selection = config["visualization"]["env_ids"]
        self._visual_envs = (
            list(range(runtime.num_envs))
            if visualization_enabled and visualization_selection == "all"
            else list(visualization_selection)
            if visualization_enabled
            else []
        )
        selected: set[int] = set()
        selected.update(self._visual_envs)
        if config["recording"]["enabled"]:
            selected.update(config["recording"]["env_ids"])
        self._selected = sorted(selected)
        self._selected_ids = self._device_ids(self._selected, runtime)
        self._gathered_rgb = self._allocate_rgb(runtime)

        self._visual_ids = self._device_ids(self._visual_envs, runtime)
        self._visual_geom_ids = (
            np.flatnonzero(runtime.host_model.geom_matid >= 0).tolist()
            if visualization_enabled
            else []
        )
        self._geom_positions: Any = None
        self._geom_orientations: Any = None
        if visualization_enabled:
            self._allocate_visual_poses(runtime)

        frames = self._download_selected(runtime)
        if visualization_enabled:
            self._start_visualization(runtime, frames)
        if config["recording"]["enabled"]:
            self._start_recording(runtime, frames)

    @staticmethod
    def _device_ids(env_ids: list[int], runtime: AICWarpRuntime) -> Any:
        if not env_ids:
            return None
        return wp.array(env_ids, dtype=int, device=runtime.device)

    def _allocate_rgb(self, runtime: AICWarpRuntime) -> dict[str, Any]:
        if not self._selected:
            return {}
        cameras = self._config["cameras"]
        return {
            camera_name: wp.empty(
                (
                    len(self._selected),
                    cameras["height"],
                    cameras["width"],
                    3,
                ),
                dtype=wp.uint8,
                device=runtime.device,
            )
            for camera_name in runtime.rgb
        }

    def _allocate_visual_poses(self, runtime: AICWarpRuntime) -> None:
        count = len(self._visual_envs)
        self._geom_positions = wp.empty(
            (count, runtime.host_model.ngeom), dtype=wp.vec3, device=runtime.device
        )
        self._geom_orientations = wp.empty(
            (count, runtime.host_model.ngeom), dtype=wp.mat33, device=runtime.device
        )

    def _download_selected(
        self, runtime: AICWarpRuntime
    ) -> dict[str, dict[int, np.ndarray]]:
        if not self._selected:
            return {}
        frames: dict[str, dict[int, np.ndarray]] = {}
        cameras = self._config["cameras"]
        for camera_name, tensor in runtime.rgb.items():
            gathered = self._gathered_rgb[camera_name]
            wp.launch(
                _gather_rgb,
                dim=(len(self._selected), cameras["height"], cameras["width"]),
                inputs=[tensor, self._selected_ids],
                outputs=[gathered],
                device=runtime.device,
            )
            host_tensor = gathered.numpy()
            frames[camera_name] = {
                env_id: host_tensor[local_index]
                for local_index, env_id in enumerate(self._selected)
            }
        return frames

    def _download_visual_poses(
        self, runtime: AICWarpRuntime
    ) -> tuple[np.ndarray, np.ndarray]:
        wp.launch(
            _gather_vec3,
            dim=(len(self._visual_envs), runtime.host_model.ngeom),
            inputs=[runtime.data.geom_xpos, self._visual_ids],
            outputs=[self._geom_positions],
            device=runtime.device,
        )
        wp.launch(
            _gather_mat33,
            dim=(len(self._visual_envs), runtime.host_model.ngeom),
            inputs=[runtime.data.geom_xmat, self._visual_ids],
            outputs=[self._geom_orientations],
            device=runtime.device,
        )
        return (
            self._geom_positions.numpy(),
            self._geom_orientations.numpy(),
        )

    @staticmethod
    def _mesh_color(
        model: Any, geom_id: int
    ) -> tuple[tuple[int, int, int], float | None]:
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
    def _wxyz(rotation: np.ndarray) -> np.ndarray:
        from viser.transforms import SO3

        return SO3.from_matrix(rotation).wxyz

    def _start_visualization(
        self,
        runtime: AICWarpRuntime,
        frames: dict[str, dict[int, np.ndarray]],
    ) -> None:
        import viser

        visual = self._config["visualization"]
        model = runtime.host_model
        geom_positions, geom_orientations = self._download_visual_poses(runtime)
        self._server = viser.ViserServer(
            host=visual["host"], port=visual["port"], label="AIC MuJoCo-Warp"
        )
        self._server.initial_camera.position = tuple(visual["initial_camera_position"])
        self._server.initial_camera.look_at = tuple(visual["initial_camera_look_at"])

        grid_spacing = np.asarray(visual["grid_spacing"], dtype=np.float64)
        grid_columns = visual["grid_columns"]
        for local_index, env_id in enumerate(self._visual_envs):
            env_root = f"/environments/{env_id}"
            column = local_index % grid_columns
            row = local_index // grid_columns
            self._server.scene.add_frame(
                env_root,
                show_axes=False,
                position=(
                    column * grid_spacing[0],
                    row * grid_spacing[1],
                    0.0,
                ),
            )
            for geom_id in self._visual_geom_ids:
                mesh_id = int(model.geom_dataid[geom_id])
                if mesh_id < 0:
                    raise ValueError(
                        f"Viser geometry {geom_id} is not a compiled mesh"
                    )
                vertex_start = int(model.mesh_vertadr[mesh_id])
                vertex_end = vertex_start + int(model.mesh_vertnum[mesh_id])
                face_start = int(model.mesh_faceadr[mesh_id])
                face_end = face_start + int(model.mesh_facenum[mesh_id])
                color, opacity = self._mesh_color(model, geom_id)
                self._mesh_handles[(env_id, geom_id)] = self._server.scene.add_mesh_simple(
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
                    wxyz=self._wxyz(geom_orientations[local_index, geom_id]),
                )

            for camera_name in runtime.robot.camera_ids:
                image = frames[camera_name][env_id]
                self._image_handles[(env_id, camera_name)] = self._server.gui.add_image(
                    image,
                    label=f"env {env_id}: {camera_name}",
                    format="jpeg",
                    jpeg_quality=visual["jpeg_quality"],
                )
        print(f"Viewer: http://{visual['host']}:{visual['port']}")

    def _start_recording(
        self,
        runtime: AICWarpRuntime,
        frames: dict[str, dict[int, np.ndarray]],
    ) -> None:
        import imageio.v2 as imageio

        recording = self._config["recording"]
        output_directory = Path(recording["output_directory"])
        output_directory.mkdir(parents=True, exist_ok=True)
        for env_id in recording["env_ids"]:
            for camera_name in runtime.rgb:
                path = output_directory / f"env_{env_id:04d}_{camera_name}.mp4"
                self._writers[(env_id, camera_name)] = imageio.get_writer(
                    path,
                    fps=self._config["cameras"]["fps"],
                    codec=recording["codec"],
                )
        self.update(runtime, frames)

    def _update_visual_poses(self, runtime: AICWarpRuntime) -> None:
        geom_positions, geom_orientations = self._download_visual_poses(runtime)
        for local_index, env_id in enumerate(self._visual_envs):
            for geom_id in self._visual_geom_ids:
                handle = self._mesh_handles[(env_id, geom_id)]
                handle.position = geom_positions[local_index, geom_id]
                handle.wxyz = self._wxyz(geom_orientations[local_index, geom_id])

    def update(
        self,
        runtime: AICWarpRuntime,
        frames: dict[str, dict[int, np.ndarray]] | None = None,
    ) -> None:
        if not self._selected:
            return
        current = self._download_selected(runtime) if frames is None else frames
        for (env_id, camera_name), handle in self._image_handles.items():
            handle.image = current[camera_name][env_id]
        if self._server is not None:
            self._update_visual_poses(runtime)
        for (env_id, camera_name), writer in self._writers.items():
            writer.append_data(current[camera_name][env_id])

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()
        if self._server is not None:
            self._server.stop()
