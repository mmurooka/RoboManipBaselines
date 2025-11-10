from os import path

import mujoco
import numpy as np

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase


class MujocoUR5eCableEnv(MujocoUR5eEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoUR5eEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/ur5e/env_ur5e_cable.xml",
            ),
            np.array(
                [
                    np.pi,
                    -np.pi / 2,
                    -0.75 * np.pi,
                    -0.25 * np.pi,
                    np.pi / 2,
                    np.pi / 2,
                    *np.zeros(8),
                ]
            ),
            **kwargs,
        )

        self.original_pole_pos = self.model.body("poles").pos.copy()
        self.pole_pos_offsets = np.array(
            [
                [-0.06, 0.0, 0.0],
                [0.03, 0.03, 0.0],
                [0.03, -0.03, 0.0],
                [0.06, 0.03, 0.0],
                [0.06, -0.03, 0.0],
                [0.15, 0.0, 0.0],
            ]
        )  # [m]

        self.cable_body_ids = None

    def _get_reward(self):
        # Get grid position list of cable
        if self.cable_body_ids is None:
            self.cable_body_ids = []
            for body_id in range(self.model.nbody):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if name is not None and name.startswith("cable_B"):
                    self.cable_body_ids.append(body_id)
        cable_grid_pos_list = np.array(
            [self.data.xpos[body_id] for body_id in self.cable_body_ids]
        )

        # Get position of poles
        pole1_pos = self.data.geom("pole1").xpos.copy()
        pole2_pos = self.data.geom("pole2").xpos.copy()

        # Check cable height
        z_thre = pole1_pos[2] + 0.01  # [m]
        if cable_grid_pos_list[:, 2].max() > z_thre:
            return 0.0

        # Check cable end
        cable_end_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cable_end"
        )
        cable_end_pos = self.data.xpos[cable_end_body_id].copy()
        x_thre = pole2_pos[0]
        y_thre = pole1_pos[1] - 0.05
        if cable_end_pos[0] < x_thre or cable_end_pos[1] > y_thre:
            return 0.0

        # Check if the cable passes through the poles
        cable_grid_xy_list = cable_grid_pos_list[:, :2]
        pole1_xy = pole1_pos[:2]
        pole2_xy = pole2_pos[:2]
        pole_dir = pole2_xy - pole1_xy

        def check_ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        for i in range(len(cable_grid_xy_list) - 1):
            cable_grid1_pos = cable_grid_xy_list[i]
            cable_grid2_pos = cable_grid_xy_list[i + 1]
            if (
                check_ccw(cable_grid1_pos, pole1_xy, pole2_xy)
                != check_ccw(cable_grid2_pos, pole1_xy, pole2_xy)
            ) and (
                check_ccw(cable_grid1_pos, cable_grid2_pos, pole1_xy)
                != check_ccw(cable_grid1_pos, cable_grid2_pos, pole2_xy)
            ):
                cable_dir = cable_grid2_pos - cable_grid1_pos
                cable_pole_cross = (
                    pole_dir[0] * cable_dir[1] - pole_dir[1] * cable_dir[0]
                )
                if cable_pole_cross > 0:
                    return 1.0

        return 0.0

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.pole_pos_offsets)

        pole_pos = self.original_pole_pos + self.pole_pos_offsets[world_idx]
        if self.world_random_scale is not None:
            pole_pos += np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )
        self.model.body("poles").pos = pole_pos

        return world_idx
