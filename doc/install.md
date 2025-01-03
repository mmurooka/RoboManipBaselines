# Install

## Common installation
Install RoboManipBaselines:
```console
$ git clone git@github.com:isri-aist/RoboManipBaselines.git --recursive
$ cd RoboManipBaselines
$ pip install -e .
```

**Note**: If you have problems installing the Pinocchio library (`pin` module) from `pip` in certain environments (e.g. Ubuntu 20.04), you can also install it via `apt`. See [here](https://stack-of-tasks.github.io/pinocchio/download.html#Install) for details.

This common installation enables data collection by teleoperation in the MuJoCo environments.

## Installation of each policy
Complete [the common installation](#common-installation) first.

### [SARNN](../robo_manip_baselines/sarnn)
Install dependent libraries including [EIPL](https://github.com/ogata-lab/eipl):
```console
# Go to the top directory of this repository
$ pip install -e .[sarnn]

# Go to the top directory of this repository
$ cd third_party/eipl
$ pip install -e .
```

### [ACT](../robo_manip_baselines/act)
Install dependent libraries including [ACT](https://github.com/tonyzhaozh/act):
```console
# Go to the top directory of this repository
$ pip install -e .[act]

# Go to the top directory of this repository
$ cd third_party/act/detr
$ pip install -e .
```

### [Diffusion policy](../robo_manip_baselines/diffusion_policy)
Install dependent libraries including [r3m](https://github.com/facebookresearch/r3m) and [diffusion policy](https://github.com/real-stanford/diffusion_policy):
```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

# Go to the top directory of this repository
$ pip install -e .[diffusion-policy]

# Go to the top directory of this repository
$ cd third_party/r3m
$ pip install -e .

# Go to the top directory of this repository
$ cd third_party/diffusion_policy
$ pip install -e .
```

**Note**: If you encounter the following error,
```python
pip._vendor.packaging.requirements.InvalidRequirement: Expected end or semicolon (after version specifier)
    opencv-python>=3.
```
replace all `opencv-python>=3.` with `opencv-python>=3.0` in `<venv_directory>/lib/python3.8/site-packages/gym-0.21.0-py3.8.egg-info/requires.txt`.

## Installation of each environment
Complete [the common installation](#common-installation) first.

### [MuJoCo environments](../robo_manip_baselines/envs/mujoco)
The MuJoCo environment can be used with only a common installation.

### [Isaac environments](../robo_manip_baselines/envs/isaac)
Isaac Gym supports only Python 3.6, 3.7 and 3.8.
In Ubuntu 22.04, use Python 3.8 with [pyenv](https://github.com/pyenv/pyenv).

Download and unpack the Isaac Gym package from [here](https://developer.nvidia.com/isaac-gym).

Install Isaac Gym according to `IsaacGym_Preview_4_Package/isaacgym/doc/install.html`:
```console
$ cd IsaacGym_Preview_4_Package/isaacgym/python
$ pip install -e .
```

Confirm that the sample program can be executed.
```console
$ cd IsaacGym_Preview_4_Package/isaacgym/python/examples
$ python joint_monkey.py
```

Isaac Gym and MuJoCo version 3 are conflicted by a file of the same name, `libsdf.so`, which triggers the following error: `undefined symbol: _ZN32pxrInternal_v0_19__pxrReserved__17 SdfValueTypeNamesE`.
Downgrade MuJoCo:
```console
$ pip install mujoco==2.3.7
```

### [Real UR5e environments](../robo_manip_baselines/envs/real/ur5e)
Install dependent libraries including [gello_software](https://github.com/wuphilipp/gello_software):
```console
# Go to the top directory of this repository
$ pip install -e .[real-ur5e]

# Go to the top directory of this repository
$ cd third_party/gello_software
$ pip install -e .
```

See [here](./real_ur5e.md) for instructions on how to operate real robot.

### [Real xArm7 environments](../robo_manip_baselines/envs/real/xarm7)
Install dependent libraries including [gello_software](https://github.com/wuphilipp/gello_software):
```console
$ sudo apt-get install libboost-all-dev

# Go to the top directory of this repository
$ pip install -e .[real-xarm7]

# Go to the top directory of this repository
$ cd third_party/gello_software
$ pip install -e .
```
