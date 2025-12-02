# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shadow Hand environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


### Tactile
gym.register(
    id="Isaac-Shadow-Hand-Tactile-Direct-v0",
    entry_point=f"{__name__}.shadow_hand_tactile_env:ShadowHandTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_tactile_env_cfg:ShadowHandTactileEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandTactileFFPPORunnerCfg",
    },
)
