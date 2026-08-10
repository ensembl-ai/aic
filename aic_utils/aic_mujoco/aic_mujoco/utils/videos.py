"""Small video-decoding utilities for policy datasets."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import torch


def decode_rgb_video(path: str | Path) -> torch.Tensor:
    """Decode an entire MP4 into an ordered uint8 RGB tensor.

    Args:
        path: Video file to decode.

    Returns:
        RGB frames shaped ``(time, channels, height, width)``.

    Raises:
        FileNotFoundError: If the video does not exist.
        ValueError: If the video contains no decodable frames.
    """

    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    frames: list[torch.Tensor] = []
    with av.open(str(video_path), mode="r") as container:
        for frame in container.decode(video=0):
            rgb = np.asarray(frame.to_ndarray(format="rgb24")).copy()
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
    if not frames:
        raise ValueError(f"Video contains no RGB frames: {video_path}")
    return torch.stack(frames)
