import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import GraspPhaseBase, ReachPhaseBase


class ReachPhase1(ReachPhaseBase):
    def set_target(self):
        target_rot = pin.rpy.rpyToMatrix(-np.pi, 0.0, 0.0)
        target_pos = np.array([0.0, 0.45, 0.45])
        self.target_se3 = pin.SE3(target_rot, target_pos)
        self.duration = 0.7  # [s]


class ReachPhase2(ReachPhaseBase):
    def set_target(self):
        target_rot = pin.rpy.rpyToMatrix(-np.pi, 0.0, 0.0)
        target_pos = np.array([0.0, 0.45, 0.36])
        self.target_se3 = pin.SE3(target_rot, target_pos)
        self.duration = 0.7  # [s]


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        self.gripper_joint_pos = np.array([0.0])
        self.duration = 0.5  # [s]


class OperationRealUR5eDemo:
    def __init__(
        self,
        robot_ip,
        camera_ids=None,
        gelsight_ids=None,
        pointcloud_camera_ids=None,
        sanwa_keyboard_ids=None,
    ):
        self.robot_ip = robot_ip
        self.camera_ids = camera_ids
        self.gelsight_ids = gelsight_ids
        self.pointcloud_camera_ids = pointcloud_camera_ids
        self.sanwa_keyboard_ids = sanwa_keyboard_ids
        super().__init__()

    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/RealUR5eDemoEnv-v0",
            robot_ip=self.robot_ip,
            camera_ids=self.camera_ids,
            gelsight_ids=self.gelsight_ids,
            pointcloud_camera_ids=self.pointcloud_camera_ids,
            sanwa_keyboard_ids=self.sanwa_keyboard_ids,
        )

    def get_pre_motion_phases(self):
        return [ReachPhase1(self), ReachPhase2(self), GraspPhase(self)]
