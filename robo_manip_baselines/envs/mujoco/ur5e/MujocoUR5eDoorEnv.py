from os import path

import numpy as np

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase


class MujocoUR5eDoorEnv(MujocoUR5eEnvBase):
    default_camera_config = {
        "azimuth": 45.0,
        "elevation": -45.0,
        "distance": 1.8,
        "lookat": [-0.2, -0.2, 0.8],
    }

    def __init__(
        self,
        **kwargs,
    ):
        MujocoUR5eEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/ur5e/env_ur5e_door.xml",
            ),
            np.array(
                [
                    np.pi,
                    -0.4 * np.pi,
                    -0.65 * np.pi,
                    -0.25 * np.pi,
                    np.pi / 2,
                    np.pi / 2,
                    *np.zeros(8),
                ]
            ),
            **kwargs,
        )

        self.original_door_pos = self.model.body("door").pos.copy()
        self.door_pos_offsets = np.array(
            [
                [0.0, -0.09, 0.0],
                [0.03, 0.0, 0.0],
                [-0.03, 0.0, 0.0],
                [0.03, 0.03, 0.0],
                [-0.03, 0.03, 0.0],
                [0.0, 0.12, 0.0],
            ]
        )  # [m]

    def _get_reward(self):
        gripper_pos = self.data.site("pinch").xpos.copy()
        handle_pos = self.data.geom("door_handle").xpos.copy()
        gripper_handle_dist = np.linalg.norm(gripper_pos - handle_pos)
        gripper_handle_dist_margin = 0.08  # [m]
        reaching_reward = np.exp(
            -10.0 * np.max([gripper_handle_dist - gripper_handle_dist_margin, 0.0])
        )

        door_angle = self.data.joint("door").qpos[0]
        door_angle_target = np.deg2rad(-45.0)
        opening_reward = np.clip(door_angle / door_angle_target, 0.0, 1.0)
        if opening_reward >= 1.0:
            reaching_reward = 1.0

        return 0.5 * (reaching_reward + opening_reward)

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.door_pos_offsets)

        door_pos = self.original_door_pos + self.door_pos_offsets[world_idx]
        if self.world_random_scale is not None:
            door_pos += np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )
        self.model.body("door").pos = door_pos

        return world_idx
