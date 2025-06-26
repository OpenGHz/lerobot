#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
import numpy as np

from lerobot.common.robots.airbot import AIRBOTPlayFollower, AIRBOTPlayFollowerConfig
from airbot_data_collection.common.utils.transformations import (
    quaternion_inverse,
    quaternion_multiply,
)

from ..teleoperator import Teleoperator
from .config_play_leader import AIRBOTPlayLeaderConfig

logger = logging.getLogger(__name__)


class AIRBOTPlayLeader(Teleoperator):
    config_class = AIRBOTPlayLeaderConfig
    name = "airbot_play_leader"

    def __init__(self, config: AIRBOTPlayLeaderConfig):
        super().__init__(config)
        self.config = config
        self.interface = AIRBOTPlayFollower(
            AIRBOTPlayFollowerConfig(port=config.port, use_pose=config.use_pose)
        )

    @property
    def action_features(self) -> dict[str, type]:
        return self.interface.action_features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.interface.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.interface.connect(calibrate)
        init_state = np.array(list(self.interface.get_observation().values()))
        self._init_pose = (init_state[:3], init_state[3:7])
        self._init_joint = init_state[7:]

    @property
    def is_calibrated(self) -> bool:
        return self.interface.is_calibrated

    def calibrate(self) -> None:
        return self.interface.calibrate()

    def configure(self) -> None:
        return self.interface.configure()

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.interface.get_observation()
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        if self.config.relative:
            action = self._get_rela_action(action)
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        return self.interface.disconnect()

    def _get_rela_action(self, action: dict[str, float]) -> dict[str, float]:
        values = np.array(list(action.values()))
        values[:3] = values[:3] - self._init_pose[0]
        # quaternion multiplication to get the relative quaternion
        values[3:7] = quaternion_multiply(
            values[3:7], quaternion_inverse(self._init_pose[1])
        )
        values[7:] = values[7:] - self._init_joint
        rela_action = dict(zip(action.keys(), values, strict=True))
        return rela_action
