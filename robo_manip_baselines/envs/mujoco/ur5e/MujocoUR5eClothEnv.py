from os import path

import mujoco
import numpy as np

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase


class MujocoUR5eClothEnv(MujocoUR5eEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoUR5eEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/ur5e/env_ur5e_cloth.xml",
            ),
            np.array(
                [
                    1.2 * np.pi,
                    -np.pi / 2,
                    -0.85 * np.pi,
                    -0.65 * np.pi,
                    -0.5 * np.pi,
                    0.0,
                    *np.zeros(8),
                ]
            ),
            **kwargs,
        )

        self.original_cloth_pos = self.model.body("cloth").pos.copy()
        self.original_board_pos = self.model.body("board").pos.copy()
        self.pos_cloth_offsets = np.array(
            [
                [0.0, -0.16, 0.0],
                [0.04, -0.04, 0.0],
                [-0.04, -0.04, 0.0],
                [0.04, 0.0, 0.0],
                [-0.04, 0.0, 0.0],
                [0.0, 0.12, 0.0],
            ]
        )  # [m]

    def _get_reward(self):
        # Get grid position list of cloth
        cloth_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cloth")
        cloth_grid_pos_list = []
        for body_id in range(self.model.nbody):
            if self.model.body_parentid[body_id] == cloth_body_id:
                cloth_grid_pos_list.append(self.data.xpos[body_id].copy())
        cloth_grid_pos_list = np.array(cloth_grid_pos_list)

        # Get vertex position list of board
        board_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "board")
        board_center = self.data.geom_xpos[board_geom_id]
        board_rotation = self.data.geom_xmat[board_geom_id].reshape(3, 3)
        board_size = self.model.geom_size[board_geom_id]
        signs = np.array(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ]
        )
        board_local_vertices = signs * board_size
        board_world_vertices = (
            board_rotation @ board_local_vertices.T
        ).T + board_center

        # Check position
        x_thre = board_world_vertices[:, 0].min() - 0.001  # [m]
        z_thre = board_world_vertices[:, 2].min() - 0.092  # [m]
        if (cloth_grid_pos_list[:, 0].min() > x_thre) and (
            cloth_grid_pos_list[:, 2].min() > z_thre
        ):
            return 1.0
        else:
            return 0.0

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.pos_cloth_offsets)

        cloth_pos = self.original_cloth_pos + self.pos_cloth_offsets[world_idx]
        board_pos = self.original_board_pos + self.pos_cloth_offsets[world_idx]
        if self.world_random_scale is not None:
            delta_pos = np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )
            cloth_pos += delta_pos
            board_pos += delta_pos
        self.model.body("cloth").pos = cloth_pos
        self.model.body("board").pos = board_pos

        return world_idx
