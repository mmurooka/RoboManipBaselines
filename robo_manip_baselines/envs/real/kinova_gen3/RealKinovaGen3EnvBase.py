import time
from os import path

import numpy as np
from gymnasium.spaces import Box, Dict
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Session_pb2
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport

from robo_manip_baselines.common import ArmConfig
from robo_manip_baselines.teleop import (
    GelloInputDevice,
    KeyboardInputDevice,
    SpacemouseInputDevice,
)

from ..RealEnvBase import RealEnvBase


def norm_180(angle):
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle


def joint_step(c_joint, m_joint, time, time_now):
    joint = m_joint + (c_joint - m_joint) * time_now / time

    return joint


class RealKinovaGen3EnvBase(RealEnvBase):
    action_space = Box(
        low=np.array(
            [
                -2 * np.pi,
                -2 * np.pi,
                -2 * np.pi,
                -2 * np.pi,
                -2 * np.pi,
                -2 * np.pi,
                -2 * np.pi,
                0.0,
            ],
            dtype=np.float32,
        ),
        high=np.array(
            [
                2 * np.pi,
                2 * np.pi,
                2 * np.pi,
                2 * np.pi,
                2 * np.pi,
                2 * np.pi,
                2 * np.pi,
                1.0,
            ],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64),
            "wrench": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
        }
    )

    def __init__(
        self,
        robot_ip,
        camera_ids,
        gelsight_ids,
        init_qpos,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Setup robot
        self.init_qpos = init_qpos
        self.joint_vel_limit = np.deg2rad(80)  # [rad/s]
        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__),
                    "../../assets/common/robots/kinovagen3/kinovagen3.urdf",
                ),
                arm_root_pose=None,
                ik_eef_joint_id=7,
                arm_joint_idxes=np.arange(7),
                gripper_joint_idxes=np.array([7]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
                eef_idx=0,
                init_arm_joint_pos=self.init_qpos[0:7],
                init_gripper_joint_pos=np.zeros(1),
            )
        ]

        # Connect to KinovaGen3
        print(f"[{self.__class__.__name__}] Start connecting the KinovaGen3.")
        self.robot_ip = robot_ip
        self.tcp_transport = TCPTransport()
        tcp_port = 10000
        self.tcp_transport.connect(self.robot_ip, tcp_port)
        self.router = RouterClient(self.tcp_transport, RouterClient.basicErrorCallback)

        self.session_info = Session_pb2.CreateSessionInfo()
        self.session_info.username = "admin"
        self.session_info.password = "admin"
        self.session_info.session_inactivity_timeout = 10000
        self.session_info.connection_inactivity_timeout = 2000
        self.session_manager = SessionManager(self.router)
        self.session_manager.CreateSession(self.session_info)

        self.base = BaseClient(self.router)

        self.udp_transport = UDPTransport()
        udp_port = 10001
        self.udp_transport.connect(self.robot_ip, udp_port)
        self.router_realtime = RouterClient(
            self.udp_transport, RouterClient.basicErrorCallback
        )

        self.session_manager_realtime = SessionManager(self.router_realtime)
        self.session_manager_realtime.CreateSession(self.session_info)

        self.base_cyclic = BaseCyclicClient(self.router_realtime)

        angles = self.base.GetMeasuredJointAngles()
        self.arm_joint_pos_actual = np.array(
            [norm_180(np.deg2rad(angles.joint_angles[i].value)) for i in range(7)]
        )  # \todo
        print(f"[{self.__class__.__name__}] Finish connecting the KinovaGen3.")

        # Connect to RealSense
        self.setup_realsense(camera_ids)
        self.setup_gelsight(gelsight_ids)

    def setup_input_device(self, input_device_name, motion_manager, overwrite_kwargs):
        if input_device_name == "spacemouse":
            InputDeviceClass = SpacemouseInputDevice
        elif input_device_name == "gello":
            InputDeviceClass = GelloInputDevice
        elif input_device_name == "keyboard":
            InputDeviceClass = KeyboardInputDevice
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid input device key: {input_device_name}"
            )

        default_kwargs = self.get_input_device_kwargs(input_device_name)

        return [
            InputDeviceClass(
                motion_manager.body_manager_list[0],
                **{**default_kwargs, **overwrite_kwargs},
            )
        ]

    def get_input_device_kwargs(self, input_device_name):
        return {}

    def _reset_robot(self):
        print(
            f"[{self.__class__.__name__}] Start moving the robot to the reset position."
        )
        self._set_action_high_level(
            self.init_qpos, duration=None, joint_vel_limit_scale=0.2, wait=True
        )
        print(
            f"[{self.__class__.__name__}] Finish moving the robot to the reset position."
        )

    def _set_action(self, action, duration=None, joint_vel_limit_scale=0.5, wait=False):
        start_time = time.time()

        # Overwrite duration or joint_pos for safety
        arm_joint_pos_command = action[self.body_config_list[0].arm_joint_idxes]
        arm_joint_pos_command = self.arm_joint_pos_actual + norm_180(
            arm_joint_pos_command - self.arm_joint_pos_actual
        )
        action[self.body_config_list[0].arm_joint_idxes] = arm_joint_pos_command
        action, duration = self.overwrite_command_for_safety(
            action, duration, joint_vel_limit_scale
        )

        # Send command to Robotiq gripper
        # mode = Base_pb2.ServoingModeInformation()
        # mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        # self.base.SetServoingMode(mode)

        # gripper_pos = action[self.body_config_list[0].gripper_joint_idxes][0]
        # gripper_command = Base_pb2.GripperCommand()
        # finger = gripper_command.gripper.finger.add()

        # gripper_command.mode = Base_pb2.GRIPPER_POSITION
        # finger.finger_identifier = 1
        # finger.value = gripper_pos / 255.0
        # self.base.SendGripperCommand(gripper_command)

        # Send command to KinovaGen3
        arm_joint_pos_command = action[self.body_config_list[0].arm_joint_idxes]
        while time.time() - start_time < duration:
            cmd = BaseCyclic_pb2.Command()
            t = time.time()
            for joint_idx in range(len(arm_joint_pos_command)):
                actuator = cmd.actuators.add()
                actuator.position = np.rad2deg(
                    joint_step(
                        arm_joint_pos_command[joint_idx],
                        self.arm_joint_pos_actual[joint_idx],
                        duration,
                        t - start_time,
                    )
                )
                actuator.flags = 1

            self.base_cyclic.Refresh(cmd, 0)

            cyclic_duration = 1e-3
            time.sleep(cyclic_duration)

    def _set_action_high_level(
        self, action, duration=None, joint_vel_limit_scale=0.5, wait=False
    ):
        start_time = time.time()

        # Set high-level mode
        mode = Base_pb2.ServoingModeInformation()
        mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(mode)
        time.sleep(0.2)

        # Overwrite duration or joint_pos for safety
        arm_joint_pos_command = action[self.body_config_list[0].arm_joint_idxes]
        arm_joint_pos_command = self.arm_joint_pos_actual + norm_180(
            arm_joint_pos_command - self.arm_joint_pos_actual
        )
        action[self.body_config_list[0].arm_joint_idxes] = arm_joint_pos_command
        action, duration = self.overwrite_command_for_safety(
            action, duration, joint_vel_limit_scale
        )

        # Send command to KinovaGen3
        arm_joint_pos_command = action[self.body_config_list[0].arm_joint_idxes]
        action_kinova = Base_pb2.Action()
        action_kinova.reach_joint_angles.constraint.type = (
            Base_pb2.JOINT_CONSTRAINT_DURATION
        )
        action_kinova.reach_joint_angles.constraint.value = duration
        for joint_idx in range(len(arm_joint_pos_command)):
            joint_angle = (
                action_kinova.reach_joint_angles.joint_angles.joint_angles.add()
            )
            joint_angle.joint_identifier = joint_idx
            joint_angle.value = np.rad2deg(arm_joint_pos_command[joint_idx])

        self.base.ExecuteAction(action_kinova)

        # Send command to Robotiq gripper
        gripper_pos = action[self.body_config_list[0].gripper_joint_idxes][0]
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger.finger_identifier = 1
        finger.value = gripper_pos / 255.0
        self.base.SendGripperCommand(gripper_command)

        # Wait
        elapsed_duration = time.time() - start_time
        if wait and elapsed_duration < duration:
            time.sleep(duration - elapsed_duration)

        # Set low-level mode
        mode = Base_pb2.ServoingModeInformation()
        mode.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
        self.base.SetServoingMode(mode)
        time.sleep(0.2)

    def _get_obs(self):
        # Get state from KinovaGen3
        feedback = self.base_cyclic.RefreshFeedback()
        arm_joint_pos_list = []
        arm_joint_vel_list = []
        for actuator in feedback.actuators:
            arm_joint_pos_list.append(norm_180(np.deg2rad(actuator.position)))  # \todo
            arm_joint_vel_list.append(np.deg2rad(actuator.velocity))
        arm_joint_pos = np.array(arm_joint_pos_list)
        arm_joint_vel = np.array(arm_joint_vel_list)
        self.arm_joint_pos_actual = arm_joint_pos.copy()

        # Get state from Robotiq gripper
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        gripper_joint_pos = np.array(
            [self.base.GetMeasuredGripperMovement(gripper_request).finger[0].value],
            dtype=np.float64,
        )
        gripper_joint_vel = np.zeros(1)

        # Get wrench from force sensor
        feedback_base = feedback.base
        fx = feedback_base.tool_external_wrench_force_x
        fy = feedback_base.tool_external_wrench_force_y
        fz = feedback_base.tool_external_wrench_force_z
        tx = feedback_base.tool_external_wrench_torque_x
        ty = feedback_base.tool_external_wrench_torque_y
        tz = feedback_base.tool_external_wrench_torque_z
        wrench = np.array([fx, fy, fz, tx, ty, tz], dtype=np.float64)

        return {
            "joint_pos": np.concatenate(
                (arm_joint_pos, gripper_joint_pos), dtype=np.float64
            ),
            "joint_vel": np.concatenate(
                (arm_joint_vel, gripper_joint_vel), dtype=np.float64
            ),
            "wrench": wrench,
        }
