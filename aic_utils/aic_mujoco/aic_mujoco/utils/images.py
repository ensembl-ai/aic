"""Device-side image transformations shared by dataset producers."""

from __future__ import annotations

import warp as wp


@wp.kernel
def unpack_rgb(
    packed: wp.array2d[wp.uint32],
    address: int,
    width: int,
    output: wp.array4d[wp.uint8],
):
    """Expose one camera from MJWarp's packed RGB renderer allocation.

    Args:
        packed: MJWarp packed RGB pixels for every world and camera.
        address: Pixel offset of the selected camera.
        width: Camera width in pixels.
        output: Named output tensor shaped ``(N, H, W, 3)``.
    """

    world, row, column = wp.tid()
    pixel = packed[world, address + row * width + column]
    output[world, row, column, 0] = wp.uint8((pixel >> wp.uint32(16)) & wp.uint32(255))
    output[world, row, column, 1] = wp.uint8((pixel >> wp.uint32(8)) & wp.uint32(255))
    output[world, row, column, 2] = wp.uint8(pixel & wp.uint32(255))


@wp.kernel
def resize_rgb_bilinear(
    source: wp.array4d[wp.uint8],
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    output: wp.array4d[wp.uint8],
):
    """Resize a batch of RGB images with bilinear interpolation.

    Args:
        source: Batched source images shaped ``(N, H, W, 3)``.
        source_width: Source width in pixels.
        source_height: Source height in pixels.
        output_width: Requested output width in pixels.
        output_height: Requested output height in pixels.
        output: Preallocated destination images shaped ``(N, h, w, 3)``.
    """

    world, output_row, output_column = wp.tid()
    source_x = (
        (float(output_column) + 0.5) * float(source_width) / float(output_width)
        - 0.5
    )
    source_y = (
        (float(output_row) + 0.5) * float(source_height) / float(output_height)
        - 0.5
    )
    x0 = int(wp.floor(source_x))
    y0 = int(wp.floor(source_y))
    x0 = wp.clamp(x0, 0, source_width - 1)
    y0 = wp.clamp(y0, 0, source_height - 1)
    x1 = wp.min(x0 + 1, source_width - 1)
    y1 = wp.min(y0 + 1, source_height - 1)
    x_weight = wp.clamp(source_x - float(x0), 0.0, 1.0)
    y_weight = wp.clamp(source_y - float(y0), 0.0, 1.0)

    for channel in range(3):
        top = (1.0 - x_weight) * float(source[world, y0, x0, channel])
        top += x_weight * float(source[world, y0, x1, channel])
        bottom = (1.0 - x_weight) * float(source[world, y1, x0, channel])
        bottom += x_weight * float(source[world, y1, x1, channel])
        value = (1.0 - y_weight) * top + y_weight * bottom
        output[world, output_row, output_column, channel] = wp.uint8(
            wp.clamp(value + 0.5, 0.0, 255.0)
        )
