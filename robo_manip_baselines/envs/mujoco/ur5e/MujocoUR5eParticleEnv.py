from os import path

import mujoco
import numpy as np

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase


class MujocoUR5eParticleEnv(MujocoUR5eEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoUR5eEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/ur5e/env_ur5e_particle.xml",
            ),
            np.array(
                [
                    np.pi,
                    -np.pi / 2,
                    -0.75 * np.pi,
                    -0.25 * np.pi,
                    np.pi / 2,
                    np.pi,
                    *np.zeros(8),
                ]
            ),
            **kwargs,
        )

        self.original_source_pos = self.model.body("source_case").pos.copy()
        self.original_particle_pos = self.model.body("particle").pos.copy()
        self.pos_offsets = np.array(
            [
                [0.0, -0.04, 0.0],
                [0.04, 0.08, 0.0],
                [-0.04, 0.08, 0.0],
                [0.04, 0.12, 0.0],
                [-0.04, 0.12, 0.0],
                [0.0, 0.24, 0.0],
            ]
        )  # [m]

    def get_input_device_kwargs(self, input_device_name):
        if input_device_name == "spacemouse":
            return {"rpy_scale": 1e-2}
        else:
            return super().get_input_device_kwargs(input_device_name)

    def _get_reward(self):
        # Get particle position list
        particle_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "particle"
        )
        particle_pos_list = []
        for body_id in range(self.model.nbody):
            if self.model.body_parentid[body_id] == particle_body_id:
                particle_pos_list.append(self.data.xpos[body_id].copy())

        # Get goal case position and size
        goal_center = self.data.body("goal_case").xpos.copy()
        goal_half_extents = np.array([0.08, 0.08, 0.08])  # [m]

        # Check count
        count = sum(
            np.all(np.abs(particle_pos - goal_center) <= goal_half_extents)
            for particle_pos in particle_pos_list
        )
        count_thre = 3
        if count >= count_thre:
            return 1.0
        else:
            return 0.0

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.pos_offsets)

        source_pos = self.original_source_pos + self.pos_offsets[world_idx]
        particle_pos = self.original_particle_pos + self.pos_offsets[world_idx]
        if self.world_random_scale is not None:
            delta_pos = np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )
            source_pos += delta_pos
            particle_pos += delta_pos
        self.model.body("source_case").pos = source_pos
        self.model.body("particle").pos = particle_pos

        return world_idx
