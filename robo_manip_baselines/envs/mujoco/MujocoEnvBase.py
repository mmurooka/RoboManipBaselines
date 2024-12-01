from abc import ABCMeta, abstractmethod
import numpy as np
import mujoco

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer


class MujocoEnvBase(MujocoEnv, metaclass=ABCMeta):
    sim_timestep = 0.004
    frame_skip = 8
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": int(1 / (sim_timestep * frame_skip)),
    }

    def __init__(
        self,
        xml_file,
        init_qpos,
        **kwargs,
    ):
        MujocoEnv.__init__(
            self,
            model_path=xml_file,
            frame_skip=self.frame_skip,
            observation_space=self.observation_space,
            width=640,
            height=480,
            default_camera_config=self.default_camera_config,
            **kwargs,
        )
        self.mujoco_renderer.width = None
        self.mujoco_renderer.height = None

        self.setup_robot(init_qpos)
        self.setup_camera()

    @abstractmethod
    def setup_robot(self, init_qpos):
        pass

    def setup_camera(self):
        self.cameras = {}
        for camera_id in range(self.model.ncam):
            camera = {}
            camera_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id
            )
            camera["name"] = camera_name
            camera["id"] = camera_id
            camera["viewer"] = OffScreenViewer(
                self.model, self.data, width=640, height=480
            )
            self.cameras[camera_name] = camera

        # This is required to automatically switch context to free camera in render()
        # https://github.com/Farama-Foundation/Gymnasium/blob/81b87efb9f011e975f3b646bab6b7871c522e15e/gymnasium/envs/mujoco/mujoco_rendering.py#L695-L697
        self.mujoco_renderer._viewers["dummy"] = None

        self._first_render = True

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()
        reward = 0.0
        terminated = False
        info = self._get_info()

        if self.render_mode == "human":
            if self._first_render:
                self._first_render = False
                self.mujoco_renderer.viewer._hide_menu = True
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return obs, reward, terminated, False, info

    @abstractmethod
    def _get_obs(self):
        pass

    def _get_info(self):
        info = {}

        if len(self.camera_names) == 0:
            return info

        info["rgb_images"] = {}
        info["depth_images"] = {}
        for camera in self.cameras.values():
            camera["viewer"].make_context_current()
            rgb_image = camera["viewer"].render(
                render_mode="rgb_array", camera_id=camera["id"]
            )
            info["rgb_images"][camera["name"]] = rgb_image
            depth_image = camera["viewer"].render(
                render_mode="depth_array", camera_id=camera["id"]
            )
            # See https://github.com/google-deepmind/mujoco/blob/631b16e7ad192df936195658fe79f2ada85f755c/python/mujoco/renderer.py#L170-L178
            extent = self.model.stat.extent
            near = self.model.vis.map.znear * extent
            far = self.model.vis.map.zfar * extent
            depth_image = near / (1 - depth_image * (1 - near / far))
            info["depth_images"][camera["name"]] = depth_image

        return info

    def _get_reset_info(self):
        return self._get_info()

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def close(self):
        for camera in self.cameras.values():
            camera["viewer"].close()
        MujocoEnv.close(self)

    def get_joint_pos_from_obs(self, obs, exclude_gripper=False):
        """Get joint position from observation."""
        if exclude_gripper:
            return obs["joint_pos"][self.arm_action_idxes]
        else:
            return obs["joint_pos"]

    def get_joint_vel_from_obs(self, obs, exclude_gripper=False):
        """Get joint velocity from observation."""
        if exclude_gripper:
            return obs["joint_vel"][self.arm_action_idxes]
        else:
            return obs["joint_vel"]

    def get_eef_wrench_from_obs(self, obs):
        """Get end-effector wrench (fx, fy, fz, nx, ny, nz) from observation."""
        return obs["wrench"]

    def get_sim_time(self):
        """Get simulation time. [s]"""
        return self.data.time

    def get_body_pose(self, body_name):
        """Get body pose (tx, ty, tz, qw, qx, qy, qz)."""
        body = self.data.body(body_name)
        return np.concatenate((body.xpos, body.xquat))

    def get_geom_pose(self, geom_name):
        """Get geom pose (tx, ty, tz, qw, qx, qy, qz)."""
        geom = self.data.geom(geom_name)
        xquat = np.zeros(4)
        mujoco.mju_mat2Quat(xquat, geom.xmat.flatten())
        return np.concatenate((geom.xpos, xquat))

    @property
    def camera_names(self):
        """Camera names being measured."""
        return self.cameras.keys()

    def get_camera_fovy(self, camera_name):
        """Get vertical field-of-view of the camera."""
        return self.model.cam(camera_name).fovy[0]

    @abstractmethod
    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        pass

    def draw_box_marker(self, pos, mat, size, rgba):
        """Draw box marker."""
        self.mujoco_renderer.viewer.add_marker(
            pos=pos,
            mat=mat,
            label="",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            rgba=rgba,
        )