"""Small Warp kernels for selecting batched environment tensors."""

from __future__ import annotations

import warp as wp


@wp.kernel
def gather_vec3(
    source: wp.array2d[wp.vec3],
    env_ids: wp.array[int],
    output: wp.array2d[wp.vec3],
):
    """Gather selected environments from a vec3 tensor.

    Args:
        source: Source tensor shaped ``(N, M)``.
        env_ids: Ordered source environment indices.
        output: Preallocated selected tensor.
    """

    selected, item = wp.tid()
    output[selected, item] = source[env_ids[selected], item]


@wp.kernel
def gather_mat33(
    source: wp.array2d[wp.mat33],
    env_ids: wp.array[int],
    output: wp.array2d[wp.mat33],
):
    """Gather selected environments from a mat33 tensor.

    Args:
        source: Source tensor shaped ``(N, M)``.
        env_ids: Ordered source environment indices.
        output: Preallocated selected tensor.
    """

    selected, item = wp.tid()
    output[selected, item] = source[env_ids[selected], item]
