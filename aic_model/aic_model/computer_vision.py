# Copyright (c) 2026 Vitalii Russinkovskii. All rights reserved.

# Algorithms of classic computer vision

# TODO: Needs unit tests

import cv2
import numpy as np


def index_colors(image, cluster_centers):
    """
    For each pixel, find the closest cluster center by L2 distance.
    
    Args:
        image: H x W x 3 uint8 RGB image
        cluster_centers: N x 3 float/uint8 matrix of cluster centers
    
    Returns:
        H x W uint8 numpy array with the index of the closest cluster center
    """
    h, w = image.shape[:2]

    # Flatten image to (H*W, 3)
    pixels = image.reshape(-1, 3).astype(np.float32)

    # Cluster centers as float: (N, 3)
    centers = cluster_centers.astype(np.float32)

    # Vectorized L2 squared distances: (H*W, N)
    # ||p - c||^2 = ||p||^2 + ||c||^2 - 2*(p @ c.T)
    p_sq = np.sum(pixels ** 2, axis=1, keepdims=True)   # (H*W, 1)
    c_sq = np.sum(centers ** 2, axis=1, keepdims=True).T # (1, N)
    cross = pixels @ centers.T                            # (H*W, N)

    dist_sq = p_sq + c_sq - 2 * cross                    # (H*W, N)

    # Index of closest center per pixel
    labels = np.argmin(dist_sq, axis=1)                  # (H*W,)

    return labels.reshape(h, w).astype(np.uint8)


def index_colors_with_limit(image, cluster_centers, d2):
    h, w = image.shape[:2]

    pixels = image.reshape(-1, 3).astype(np.float32)
    centers = cluster_centers.astype(np.float32)

    p_sq = np.sum(pixels ** 2, axis=1, keepdims=True)
    c_sq = np.sum(centers ** 2, axis=1, keepdims=True).T
    cross = pixels @ centers.T

    dist_sq = p_sq + c_sq - 2 * cross                   # (H*W, N)

    min_dist_sq = np.min(dist_sq, axis=1)                # (H*W,)
    labels = np.argmin(dist_sq, axis=1).astype(np.uint8) # (H*W,)

    labels[min_dist_sq > d2] = 255

    return labels.reshape(h, w)


def get_ray(camera_info, pixel_row, pixel_col):
    """
    Get a normalized 3D ray direction for a pixel in a unrectified image.
    
    Args:
        camera_info: ROS sensor_msgs/CameraInfo object
        pixel_row: row (v) coordinate of the pixel
        pixel_col: column (u) coordinate of the pixel
    
    Returns:
        (rx, ry, rz): normalized 3D ray direction tuple
    """
    
    pts = np.array([[[pixel_col, pixel_row]]], dtype=np.float32)
    K = np.array(camera_info.k).reshape(3, 3)
    D = np.array(camera_info.d)
    R = np.array(camera_info.r).reshape(3, 3)
    P = np.array(camera_info.p).reshape(3, 4)
    rectified = cv2.undistortPoints(pts, K, D, R, P[0:3,0:3])

    pixel_col_r, pixel_row_r = rectified[0, 0]

    fx = P[0, 0]
    fy = P[1, 1]
    cx = P[0, 2]
    cy = P[1, 2]

    # Back-project pixel to normalized camera coordinates
    # (inverse of: u = fx*X/Z + cx, v = fy*Y/Z + cy, with Z=1)
    x = (pixel_col_r - cx) / fx
    y = (pixel_row_r - cy) / fy
    z = 1.0

    # Normalize to unit length
    norm = np.sqrt(x**2 + y**2 + z**2)
    rx = x / norm
    ry = y / norm
    rz = z / norm

    return rx, ry, rz


# TODO: Unit test, maybe better name, documentation
def indexed_color_to_binary_mask(image, index):
    return np.where(image == index, np.uint8(0xFF), np.uint8(0))

# TODO: Unit test, maybe better name, documentation
def dilate_erode(mask, r):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    mask255 = (mask * 255).astype(np.uint8)
    dilated = cv2.dilate(mask255, kernel)
    eroded  = cv2.erode(dilated, kernel)
    return (eroded > 0).astype(np.uint8)


def blob_centroids_with_limit(binary_image, area_limit):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8, ltype=cv2.CV_32S)
    blob_centroids = []
    for i in range(1, num_labels):
        if stats[i][4] >= area_limit:
            blob_centroids.append((centroids[i][1], centroids[i][0]))
    return blob_centroids