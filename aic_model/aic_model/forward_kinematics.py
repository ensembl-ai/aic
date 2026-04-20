# Copyright (c) 2026 Vitalii Russinkovskii. All rights reserved.

# Kinematics of UR5e

# TODO: Needs unit tests

import math
import numpy as np

def _rpy_to_rotation_matrix(roll, pitch, yaw):
    """
    Computes rotation matrix for Euler angles provided
    Rotations are done around axes of parent frame
    :param roll: Rotation about axis X of parent frame (first rotation)
    :type roll: float, radian
    :param pitch: Rotation about axis Y of parent frame (second rotation)
    :type pitch: float, radian
    :param yaw: Rotation about axis Z of parent frame (third rotation)
    :type yaw: float, radian
    :return: Rotation matrix
    :rtype: np.array, shape (3, 3)
    """
    cg = math.cos(roll)
    sg = math.sin(roll)
    cb = math.cos(pitch)
    sb = math.sin(pitch)
    ca = math.cos(yaw)
    sa = math.sin(yaw)

    R = np.array(
        [
            [ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg],
            [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg],
            [-sb, cb * sg, cb * cg],
        ],
        dtype=float,
    )

    return R


def _rotation_matrix_to_rpy(R):
    """
    Computes Euler angles from rotation matrix
    Rotations are done around axes of parent frame
    :param R: Rotation matrix
    :type R: np.array, shape is (3, 3)
    :return: Euler angles roll, pitch, yaw
    :rtype: float, float, float
    """
    yaw = math.atan2(R[1, 0], R[0, 0])
    pitch = math.atan2(-R[2, 0], math.sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2]))
    roll = math.atan2(R[2, 1], R[2, 2])
    return roll, pitch, yaw


def _transformation_matrix_to_xyzrpy(T):
    """
    Computes offset and Euler angles from transformation matrix
    :param T: Transformation matrix
    :type T: np.array, shape is (4, 4)
    :return: x, y, z, roll, pitch, yaw
    :rtype: float, float, float, float, float, float
    """
    roll, pitch, yaw = _rotation_matrix_to_rpy(T[0:3, 0:3])
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]
    return x, y, z, roll, pitch, yaw


def _xyzrpy_to_transformation_matrix(x, y, z, roll, pitch, yaw):
    """
    Computes transformation matrix from offset and Euler angles
    :param x: offset x
    :type x: float
    :param y: offset y
    :type y: float
    :param z: offset z
    :type z: float
    :param roll: Rotation about parent frame axis X
    :type roll: float
    :param pitch: Rotation about parent frame axis Y
    :type pitch: float
    :param yaw: Rotation about parent frame axis Z
    :type yaw: float
    :return: Transformation matrix
    :rtype: np.array, shape is (4, 4)
    """
    T = np.zeros((4, 4,), dtype=float)
    T[0:3, 0:3] = _rpy_to_rotation_matrix(roll, pitch, yaw)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    T[3, 3] = 1
    return T


# Please do not delete this method
# We do not use this method in this class but it will be useful for debugging
# and maintenance of this code, so it is recommended to leave it here
# For rotation RZ RY RX
def _quaternion_to_rpy(qx, qy, qz, qw):
    """
    Computes Euler angle from quaternion
    :param qx: qx of quaternion
    :type qx: float
    :param qy: qy of quaternion
    :type qy: float
    :param qz: qz of quaternion
    :type qz: float
    :param qw: qw of quaternion
    :type qw: float
    :return: Euler angles roll, pitch, yaw
    :rtype: float, float, float
    """
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def _rpy_to_quaternion(roll, pitch, yaw):
    """
    Computes quaternion from Euler angles
    :param roll: Rotation about axis X of parent frame (first rotation)
    :type roll: float, radian
    :param pitch: Rotation about axis Y of parent frame (second rotation)
    :type pitch: float, radian
    :param yaw: Rotation about axis Z of parent frame (third rotation)
    :type yaw: float, radian
    :return: quaternion in form qx, qy, qz, qw
    :rtype: float, float, float, float
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # Quaternion
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qx, qy, qz, qw

################################################################################

def denavit_hartenberg_matrix(theta, d, a, alpha):
    """
    Create the Denavit-Hartenberg transformation matrix.

    Parameters:
    theta : float : Joint angle (in radians)
    d : float : Link offset
    a : float : Link length
    alpha : float : Link twist (in radians)

    Returns:
    numpy.ndarray : The D-H transformation matrix
    """
    # Create the transformation matrix
    transformation_matrix = np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])

    return transformation_matrix

# CORRECT TRANSFORMATION
def T1_0(thetha1):
    d1 = 0.16250
    a1 = 0.0
    alpha1 = 0.5 * np.pi
    t1 = np.pi
    return denavit_hartenberg_matrix(t1 + thetha1, d1, a1, alpha1)

# CORRECT TRANSFORMATION
def T2_1(thetha2):
    d2 = 0.0
    a2 = -0.42500
    alpha2 = 0.0
    return denavit_hartenberg_matrix(thetha2, d2, a2, alpha2)

# CORRECT TRANSFORMATION
def T3_2(thetha3):
    d3 = 0.0
    a3 = -0.3922
    alpha3 = 0.0
    return denavit_hartenberg_matrix(thetha3, d3, a3, alpha3)

# CORRECT TRANSFORMATION
def T4_3(thetha4):
    d4 = 0.1333
    a4 = 0
    alpha4 = 0.5 * np.pi
    return denavit_hartenberg_matrix(thetha4, d4, a4, alpha4)

# CORRECT TRANSFORMTAION
def T5_4(thetha5):
    d5 = 0.0997
    a5 = 0
    alpha5 = -0.5 * np.pi
    return denavit_hartenberg_matrix(thetha5, d5, a5, alpha5)

# CORRECT TRANSFORM
def T6_5(thetha6):
    d6 = 0.0996
    a6 = 0.0
    alpha6 = 0.0
    return denavit_hartenberg_matrix(thetha6, d6, a6, alpha6)

def T_gripper_tcp_6():
    x, y, z = 0.0, 0.0, 0.1965
    qx, qy, qz, qw = 0.0, 0.0, 0.0, -1.0
    roll, pitch, yaw = _quaternion_to_rpy(qx, qy, qz, qw)
    transformation_matrix = _xyzrpy_to_transformation_matrix(x, y, z, roll, pitch, yaw)
    return transformation_matrix

def T_center_camera_6():
    x, y, z = 0.0, -0.11607918, -0.00893789
    qx, qy, qz, qw = 0.13052833, -0.00000183, 0.00000029, -0.99144458
    roll, pitch, yaw = _quaternion_to_rpy(qx, qy, qz, qw)
    transformation_matrix = _xyzrpy_to_transformation_matrix(x, y, z, roll, pitch, yaw)
    return transformation_matrix

def T_left_camera_6():
    x, y, z = -0.10051658, -0.05803259, -0.00893589
    qx, qy, qz, qw = 0.11303995, -0.06526573, 0.49572239, -0.85861614
    roll, pitch, yaw = _quaternion_to_rpy(qx, qy, qz, qw)
    transformation_matrix = _xyzrpy_to_transformation_matrix(x, y, z, roll, pitch, yaw)
    return transformation_matrix

def T_right_camera_6():
    x, y, z = 0.10051658, -0.05803259, -0.00893589
    qx, qy, qz, qw = 0.11304177, 0.06526256, -0.49572189, -0.85861642
    roll, pitch, yaw = _quaternion_to_rpy(qx, qy, qz, qw)
    transformation_matrix = _xyzrpy_to_transformation_matrix(x, y, z, roll, pitch, yaw)
    return transformation_matrix

def T_base_right(base_xyzrpy):
    T_base = _xyzrpy_to_transformation_matrix(base_xyzrpy[0], base_xyzrpy[1], base_xyzrpy[2], base_xyzrpy[3], base_xyzrpy[4], base_xyzrpy[5])
    return T_base

def T1(base_xyzrpy, joints):
    T1 = np.dot(T_base_right(base_xyzrpy), T1_0(joints[0]))
    return T1

def T2(base_xyzrpy, joints):
    T2 = np.dot(T1(base_xyzrpy, joints), T2_1(joints[1]))
    return T2

def T3(base_xyzrpy, joints):
    T3 = np.dot(T2(base_xyzrpy, joints), T3_2(joints[2]))
    return T3

def T4(base_xyzrpy, joints):
    T4 = np.dot(T3(base_xyzrpy, joints), T4_3(joints[3]))
    return T4

def T5(base_xyzrpy, joints):
    T5 = np.dot(T4(base_xyzrpy, joints), T5_4(joints[4]))
    return T5

# T6 matches wrist_3_link
def T6(base_xyzrpy, joints):
    T6 = np.dot(T5(base_xyzrpy, joints), T6_5(joints[5]))
    return T6

def T_gripper_tcp(base_xyzrpy, joints):
    T_gripper_tcp = np.dot(T6(base_xyzrpy, joints), T_gripper_tcp_6())
    return T_gripper_tcp

def T_center_camera(base_xyzrpy, joints):
    T_center_camera = np.dot(T6(base_xyzrpy, joints), T_center_camera_6())
    return T_center_camera

def T_left_camera(base_xyzrpy, joints):
    T_left_camera = np.dot(T6(base_xyzrpy, joints), T_left_camera_6())
    return T_left_camera

def T_right_camera(base_xyzrpy, joints):
    T_right_camera = np.dot(T6(base_xyzrpy, joints), T_right_camera_6())
    return T_right_camera


class Test:
    def __init__(self):
        pass

