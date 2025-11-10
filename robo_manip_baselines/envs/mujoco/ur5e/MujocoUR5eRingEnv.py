from os import path

import mujoco
import numpy as np
from matplotlib.path import Path

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase


class MujocoUR5eRingEnv(MujocoUR5eEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoUR5eEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/ur5e/env_ur5e_ring.xml",
            ),
            np.array(
                [
                    np.pi,
                    -np.pi / 2,
                    -0.75 * np.pi,
                    -0.75 * np.pi,
                    -0.5 * np.pi,
                    0.0,
                    *np.zeros(8),
                ]
            ),
            **kwargs,
        )

        self.original_pole_pos = self.model.body("pole").pos.copy()
        self.pole_pos_offsets = np.array(
            [
                [0.0, -0.04, 0.0],
                [0.04, 0.08, 0.0],
                [-0.04, 0.08, 0.0],
                [0.04, 0.12, 0.0],
                [-0.04, 0.12, 0.0],
                [0.0, 0.24, 0.0],
            ]
        )  # [m]

        self.ring_body_ids = None

    def _get_reward(self):
        # Get grid position list of ring
        if self.ring_body_ids is None:
            self.ring_body_ids = []
            for body_id in range(self.model.nbody):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if name is not None and name.startswith("ring_B"):
                    self.ring_body_ids.append(body_id)
        ring_grid_pos_list = np.array(
            [self.data.xpos[body_id] for body_id in self.ring_body_ids]
        )

        # Get position of pole
        pole_pos = self.data.body("pole").xpos.copy()

        # Check z position
        z_thre = pole_pos[2] + 0.08  # [m]
        if ring_grid_pos_list[:, 2].max() > z_thre:
            return 0.0

        # Check if the ring hangs on the pole
        ring_grid_xy_list = ring_grid_pos_list[:, :2]
        ring_grid_xy_list = np.vstack([ring_grid_xy_list, ring_grid_xy_list[0]])
        ring_path = Path(ring_grid_xy_list)
        if ring_path.contains_point(pole_pos[:2]):
            return 1.0
        else:
            return 0.0

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.pole_pos_offsets)

        pole_pos = self.original_pole_pos + self.pole_pos_offsets[world_idx]
        if self.world_random_scale is not None:
            pole_pos += np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )
        self.model.body("pole").pos = pole_pos

        return world_idx
