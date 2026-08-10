"""Selected-world Viser visualization and RGB recording."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from aic_mujoco.runtime import AICWarpRuntime


@wp.kernel
def gather_rgb(
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
def gather_vec3(
    source: wp.array2d(dtype=wp.vec3),
    env_ids: wp.array(dtype=int),
    output: wp.array2d(dtype=wp.vec3),
):
    selected, item = wp.tid()
    output[selected, item] = source[env_ids[selected], item]


@wp.kernel
def gather_mat33(
    source: wp.array2d(dtype=wp.mat33),
    env_ids: wp.array(dtype=int),
    output: wp.array2d(dtype=wp.mat33),
):
    selected, item = wp.tid()
    output[selected, item] = source[env_ids[selected], item]


class RuntimeOutputs:
    """Bridge explicitly selected device worlds to human-facing outputs."""

    def __init__(self, config: dict[str, Any], runtime: AICWarpRuntime):
        self.config = config
        self.server = None
        self.image_handles: dict[tuple[int, str], Any] = {}
        self.mesh_handles: dict[tuple[int, int], Any] = {}
        self.writers: dict[tuple[int, str], Any] = {}

        visualization_enabled = config["visualization"]["enabled"]
        visualization_selection = config["visualization"]["env_ids"]
        self.visual_envs = (
            list(range(runtime.num_envs))
            if visualization_enabled and visualization_selection == "all"
            else list(visualization_selection)
            if visualization_enabled
            else []
        )
        selected: set[int] = set()
        selected.update(self.visual_envs)
        if config["recording"]["enabled"]:
            selected.update(config["recording"]["env_ids"])
        self.selected = sorted(selected)
        self.selected_ids = self.device_ids(self.selected, runtime)
        self.gathered_rgb = self.allocate_rgb(runtime)

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

        frames = self.download_selected(runtime)
        if visualization_enabled:
            self.start_visualization(runtime, frames)
        if config["recording"]["enabled"]:
            self.start_recording(runtime, frames)

    @staticmethod
    def device_ids(env_ids: list[int], runtime: AICWarpRuntime) -> Any:
        if not env_ids:
            return None
        return wp.array(env_ids, dtype=int, device=runtime.device)

    def allocate_rgb(self, runtime: AICWarpRuntime) -> dict[str, Any]:
        if not self.selected:
            return {}
        cameras = self.config["cameras"]
        return {
            camera_name: wp.empty(
                (
                    len(self.selected),
                    cameras["height"],
                    cameras["width"],
                    3,
                ),
                dtype=wp.uint8,
                device=runtime.device,
            )
            for camera_name in runtime.rgb
        }

    def allocate_visual_poses(self, runtime: AICWarpRuntime) -> None:
        count = len(self.visual_envs)
        self.geom_positions = wp.empty(
            (count, runtime.host_model.ngeom), dtype=wp.vec3, device=runtime.device
        )
        self.geom_orientations = wp.empty(
            (count, runtime.host_model.ngeom), dtype=wp.mat33, device=runtime.device
        )

    def download_selected(
        self, runtime: AICWarpRuntime
    ) -> dict[str, dict[int, np.ndarray]]:
        if not self.selected:
            return {}
        frames: dict[str, dict[int, np.ndarray]] = {}
        cameras = self.config["cameras"]
        for camera_name, tensor in runtime.rgb.items():
            gathered = self.gathered_rgb[camera_name]
            wp.launch(
                gather_rgb,
                dim=(len(self.selected), cameras["height"], cameras["width"]),
                inputs=[tensor, self.selected_ids],
                outputs=[gathered],
                device=runtime.device,
            )
            host_tensor = gathered.numpy()
            frames[camera_name] = {
                env_id: host_tensor[local_index]
                for local_index, env_id in enumerate(self.selected)
            }
        return frames

    def download_visual_poses(
        self, runtime: AICWarpRuntime
    ) -> tuple[np.ndarray, np.ndarray]:
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
        from viser.transforms import SO3

        return SO3.from_matrix(rotation).wxyz

    def start_visualization(
        self,
        runtime: AICWarpRuntime,
        frames: dict[str, dict[int, np.ndarray]],
    ) -> None:
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
        grid_columns = visual["grid_columns"]
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

            for camera_name in runtime.robot.camera_ids:
                image = frames[camera_name][env_id]
                self.image_handles[(env_id, camera_name)] = self.server.gui.add_image(
                    image,
                    label=f"env {env_id}: {camera_name}",
                    format="jpeg",
                    jpeg_quality=visual["jpeg_quality"],
                )
        print(f"Viewer: http://{visual['host']}:{visual['port']}")

    def start_recording(
        self,
        runtime: AICWarpRuntime,
        frames: dict[str, dict[int, np.ndarray]],
    ) -> None:
        import imageio.v2 as imageio

        recording = self.config["recording"]
        output_directory = Path(recording["output_directory"])
        output_directory.mkdir(parents=True, exist_ok=True)
        for env_id in recording["env_ids"]:
            for camera_name in runtime.rgb:
                path = output_directory / f"env_{env_id:04d}_{camera_name}.mp4"
                self.writers[(env_id, camera_name)] = imageio.get_writer(
                    path,
                    fps=self.config["cameras"]["fps"],
                    codec=recording["codec"],
                )
        self.update(runtime, frames)

    def update_visual_poses(self, runtime: AICWarpRuntime) -> None:
        geom_positions, geom_orientations = self.download_visual_poses(runtime)
        for local_index, env_id in enumerate(self.visual_envs):
            for geom_id in self.visual_geom_ids:
                handle = self.mesh_handles[(env_id, geom_id)]
                handle.position = geom_positions[local_index, geom_id]
                handle.wxyz = self.wxyz(geom_orientations[local_index, geom_id])

    def update(
        self,
        runtime: AICWarpRuntime,
        frames: dict[str, dict[int, np.ndarray]] | None = None,
    ) -> None:
        if not self.selected:
            return
        current = self.download_selected(runtime) if frames is None else frames
        for (env_id, camera_name), handle in self.image_handles.items():
            handle.image = current[camera_name][env_id]
        if self.server is not None:
            self.update_visual_poses(runtime)
        for (env_id, camera_name), writer in self.writers.items():
            writer.append_data(current[camera_name][env_id])

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()
        if self.server is not None:
            self.server.stop()
