# Copyright (c) 2026 Vitalii Russinkovskii. All rights reserved.

from aic_control_interfaces.msg import JointMotionUpdate
from aic_control_interfaces.msg import TrajectoryGenerationMode


######## MARG LIBRARIES ##############
import cv2
import aic_model.computer_vision as computer_vision
import aic_model.forward_kinematics as forward_kinematics
import random

######################################

import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_control_interfaces.msg import (
    MotionUpdate,
    TrajectoryGenerationMode,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from rclpy.duration import Duration


class MargPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.get_logger().info("MargPolicy.__init__()")

    def search_color_spot(self, obs_msg: Observation, camera_name: str, cluster_index: int, area_limit: int, expected_height_z: float):
        
        image = {"left_camera": obs_msg.left_image,
                 "right_camera": obs_msg.right_image,
                 "center_camera": obs_msg.center_image}[camera_name]
        
        
        camera_info = {"left_camera": obs_msg.left_camera_info,
                      "right_camera": obs_msg.right_camera_info,
                      "center_camera": obs_msg.center_camera_info}[camera_name]
        
        img = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, 3)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 0: Taksboard 1: Table 2: Magenta logo 3: Blue of conntectors 4: Green of PCB schema
        cluster_centers = np.array([[78, 78, 78], [218, 218, 218], [192, 4, 192], [224, 161, 62], [46, 60, 36]])
        color_deviation_d2 = 2500
        color_indexes = computer_vision.index_colors_with_limit(img_bgr, cluster_centers, color_deviation_d2)
        binary_mask = computer_vision.indexed_color_to_binary_mask(color_indexes, cluster_index)
        morph_radius = 5
        binary_mask = computer_vision.dilate_erode(binary_mask, morph_radius)
        blob_centroids = computer_vision.blob_centroids_with_limit(binary_mask, area_limit)

        if len(blob_centroids) == 0:
            return None
        
        # There can be several blobs. Randomizing for change
        random.shuffle(blob_centroids)
        centroid_row, centroid_col = blob_centroids[0]
        rx, ry, rz = computer_vision.get_ray(camera_info, centroid_row, centroid_col)

        base_xyzrpy = np.array([0, 0, 0, 0, 0, 0])
        joints = self.get_arm_joint_angles(obs_msg)

        # Initial position
        T0 = forward_kinematics.T_gripper_tcp(base_xyzrpy, joints)
        x0, y0, z0, _, _, _ = forward_kinematics._transformation_matrix_to_xyzrpy(T0)
        

        T = forward_kinematics.T_center_camera(base_xyzrpy, joints)
        camera_origin = T.dot(np.array([0,0,0,1]))[0:3]
        ray = T[0:3,0:3].dot(np.array([rx, ry, rz]))

        self.get_logger().info(f"###### RAY: {ray}") 

        length = (camera_origin[2] - expected_height_z) / np.abs(ray[2])
        projection = camera_origin + length * ray
        
        destination = projection
        connector_dim_z = 0.05
        destination[2] += connector_dim_z

        self.get_logger().info(f"### DESTINATION: {destination}") 

        x1, y1, z1 = destination[0], destination[1], destination[2]

        trajectory = []
        for i in range(11):
            factor = i / 10
            x = x1 * factor + x0 * (1 - factor)
            y = y1 * factor + y0 * (1 - factor)
            z = z1 * factor + z0 * (1 - factor) 

            trajectory.append([x, y, z])            

        return trajectory


    # Very silly solution of inverse kinematics
    def find_approach(self, obs_msg: Observation, goal):
        xg, yg, zg = goal
        base_xyzrpy = np.array([0, 0, 0, 0, 0, 0])
        joints = self.get_arm_joint_angles(obs_msg)
        bj = [joints[0], joints[1], joints[2], joints[3], joints[4], joints[5]]
        best_crit = None
        for _ in range(5000):
            jc = np.zeros(6, dtype=float)
            for i in range(6):
                jc[i] = np.random.uniform(joints[i] - 0.15, joints[i] + 0.15)
            T = forward_kinematics.T_gripper_tcp(base_xyzrpy, jc)
            x, y, z, roll, pitch, yaw = forward_kinematics._transformation_matrix_to_xyzrpy(T)
            qx, qy, qz, qw = forward_kinematics._rpy_to_quaternion(roll, pitch, yaw)
            #error = np.abs(xg - x) + np.abs(yg - y) + np.abs(zg - z) + np.abs(1.0 - qx) + np.abs(qy) + np.abs(qz) + np.abs(qw)
            #error = (xg - x) * (xg - x) + (yg - y) * (yg - y) + (zg - z) * (zg - z) + np.abs(1.0 - qx) + np.abs(qy) + np.abs(qz) + np.abs(qw)
            error = max(np.abs(xg - x), np.abs(yg - y), np.abs(zg - z), np.abs(1.0 - qx), np.abs(qy), np.abs(qz), np.abs(qw))
            if best_crit is None or error < best_crit:
                best_crit = error
                bj = jc
        return [bj[0], bj[1], bj[2], bj[3], bj[4], bj[5]]
    

    def get_arm_joint_angles(self, observation: Observation) -> np.ndarray:
        """
        Decodes arm joint angles from observation
        """
        joint_map = {"shoulder_pan_joint": 0, "shoulder_lift_joint": 1, "elbow_joint": 2, "wrist_1_joint": 3, "wrist_2_joint": 4, "wrist_3_joint": 5}
        joints = np.zeros(6, dtype=float)
        for index, joint_name in enumerate(observation.joint_states.name):
            if joint_name in joint_map:
                joints[joint_map[joint_name]] = observation.joint_states.position[index]
        return joints


    ######## ZOOM OUT IS NOT USED 
    def zoom_out(self, observation: Observation) -> Pose:
        """
        Attempt to zoom out to see the task board
        """
        base_xyzrpy = np.array([0, 0, 0, 0, 0, 0])
        joints = self.get_arm_joint_angles(observation)

        self.get_logger().info(f"Joint angles: {joints}")

        T = forward_kinematics.T_gripper_tcp(base_xyzrpy, joints)
        x, y, z, roll, pitch, yaw = forward_kinematics._transformation_matrix_to_xyzrpy(T)
        qx, qy, qz, qw = forward_kinematics._rpy_to_quaternion(roll, pitch, yaw)

        self.get_logger().info(f"### XYZ: {x},{y},{z}")
        self.get_logger().info(f"### ORIENT: {qx},{qy},{qz},{qw}")

        target_height = 0.45
        pose = Pose(
            position=Point(x=x, y=y, z=target_height),
            orientation=Quaternion(x=qx, y=qy, z=qz, w=qw)
        )
        return pose
    
    ######### LOOK AROUND IS NOT USED
    def look_around(self, observation: Observation) -> list[list[float]]:
        look_around_poses = []
        j = self.get_arm_joint_angles(observation)
        look_around_poses.append([j[0], j[1], j[2], j[3] + 0.1, j[4], j[5]])
        look_around_poses.append([j[0], j[1], j[2], j[3] - 0.1, j[4], j[5]])
        look_around_poses.append([j[0], j[1], j[2], j[3], j[4] + 0.1, j[5]])
        look_around_poses.append([j[0], j[1], j[2], j[3], j[4] - 0.1, j[5]])
        return look_around_poses


    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"MargPolicy.insert_cable() enter. Task: {task}")
        start_time = self.time_now()
        timeout = Duration(seconds=10.0)
        send_feedback("Daleks conquer and destroy!")

        search_completed = False

        self.get_logger().info(f"################ VERSION 2 ########################")

        while (self.time_now() - start_time) < timeout:
            self.sleep_for(0.25)
            observation = get_observation()

            if observation is None:
                self.get_logger().info("No observation received.")
                continue

            if search_completed:
                self.get_logger().info("Search completed.")
                continue

            self.get_logger().info(f"TASK: {task}")
            # We are coming here only once
            search_completed = True
            joint_motion_update = JointMotionUpdate(
                        target_stiffness=[50.0, 50.0, 50.0, 20.0, 20.0, 20.0],
                        target_damping=[40.0, 40.0, 40.0, 20.0, 20.0, 20.0],
                        trajectory_generation_mode=TrajectoryGenerationMode(
                            mode=TrajectoryGenerationMode.MODE_POSITION
                        ),
                    )
            if task.port_type == 'sfp':
                trajectory = self.search_color_spot(observation, "center_camera", 4, 200, 0.19)
                if trajectory is not None:
                    for p in trajectory:
                        observation = get_observation()
                        j = self.find_approach(observation, p)
                        joint_motion_update.target_state.positions = j
                        move_robot(joint_motion_update=joint_motion_update)
                        self.sleep_for(1.0)
                continue

            if task.port_type == 'sc':
                trajectory = self.search_color_spot(observation, "center_camera", 3, 500, 0.03)
                if trajectory is not None:
                    for p in trajectory:
                        observation = get_observation()
                        j = self.find_approach(observation, p)
                        joint_motion_update.target_state.positions = j
                        move_robot(joint_motion_update=joint_motion_update)
                        self.sleep_for(1.0)
                continue


        self.get_logger().info("MargPolicy.insert_cable() exiting...")
        return True