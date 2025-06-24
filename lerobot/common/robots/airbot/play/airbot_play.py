from airbot_data_collection.airbot.robots.airbot_play_mock import (
    AIRBOTPlay,
    AIRBOTPlayConfig,
)

# from airbot_data_collection.airbot.robots.airbot_play import AIRBOTPlay, AIRBOTPlayConfig
from .config_airbot_play import AIRBOTPlayFollowerConfig


import logging
import time
from functools import cached_property
from typing import Any

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ...robot import Robot

logger = logging.getLogger(__name__)


class AIRBOTPlayFollower(Robot):

    config_class = AIRBOTPlayFollowerConfig
    name = "airbot_play"

    def __init__(self, config: AIRBOTPlayFollowerConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.interface = AIRBOTPlay(AIRBOTPlayConfig(port=config.port))
        self._is_connected = False

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"joint{motor}.pos": float for motor in range(1, 8)}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """
        Features of the observations returned by the robot.
        """
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate=True):
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        for cam in self.cameras.values():
            cam.connect()

        if not self.interface.configure():
            self._is_connected = False
        else:
            self._is_connected = True
            logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self):
        # logger.info(f"\nRunning calibration of {self}")
        pass

    def configure(self):
        logger.info(f"Configuring {self.name} with {self.config}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        low_dim = self.interface.capture_observation()
        obs_dict = {
            f"joint{motor + 1}.pos": val
            for motor, val in enumerate(
                low_dim["arm/joint_state"]["data"]["position"]
                + low_dim["eef/joint_state"]["data"]["position"]
            )
        }
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action (dict[str, float]): The goal positions for the motors.

        Returns:
            dict[str, float]: The action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.interface.send_action([action[key] for key in self._motors_ft])
        return action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.interface.shutdown()
        for cam in self.cameras.values():
            cam.disconnect()

        self._is_connected = False
        logger.info(f"{self} disconnected.")
