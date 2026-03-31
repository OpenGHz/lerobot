#!/usr/bin/env python

import logging
from typing import Any

import numpy as np

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_mujoco_env import MujocoEnvRobotConfig

logger = logging.getLogger(__name__)


class MujocoEnvRobot(Robot):
    """Wrap an auto_atom UnifiedMujocoEnv with the LeRobot Robot interface."""

    config_class = MujocoEnvRobotConfig
    name = "mujoco_env"

    def __init__(self, config: MujocoEnvRobotConfig):
        super().__init__(config)
        self.config = config
        self.env = None
        self._prefetched_env = None
        self._is_connected = False
        self._metadata_initialized = False
        self._observation_features: dict[str, type | tuple] = {}
        self._action_features: dict[str, type] = {key: float for key in config.action_keys}

    @property
    def cameras(self) -> list[str]:
        return self.config.image_keys

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        self._ensure_feature_metadata()
        return self._observation_features

    @property
    def action_features(self) -> dict[str, type]:
        self._ensure_feature_metadata()
        return self._action_features

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self._prefetched_env is not None:
            self.env = self._prefetched_env
            self._prefetched_env = None
        else:
            self.env = self._build_env()
        self._is_connected = True

        if self.config.reset_on_connect:
            self.env.reset()

        initial_observation = self.get_observation()
        self._observation_features = self._infer_observation_features(initial_observation)
        if not self._action_features:
            self._action_features = self._infer_action_features()
        self._metadata_initialized = True

        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        raw = self.env.capture_observation()
        self.env.update()
        if not self.env.is_updated():
            raw["skip"] = True
        return self._normalize_observation(raw)

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        env_action = self._normalize_action(action)
        self.env.step(env_action)

        if not self._action_features and isinstance(action, dict):
            self._action_features = {key: float for key in action}

        if isinstance(action, dict):
            return action
        if isinstance(env_action, np.ndarray):
            return {f"action_{i}": float(value) for i, value in enumerate(env_action.tolist())}
        return {"action": env_action}

    @check_if_not_connected
    def disconnect(self) -> None:
        self.env.close()
        self.env = None
        self._is_connected = False
        logger.info("%s disconnected.", self)

    def _ensure_feature_metadata(self) -> None:
        if self._metadata_initialized:
            return

        if self.is_connected:
            observation = self.get_observation()
            if not self._observation_features:
                self._observation_features = self._infer_observation_features(observation)
            if not self._action_features:
                self._action_features = self._infer_action_features()
            self._metadata_initialized = True
            return

        env = self._build_env()
        if self.config.reset_on_connect:
            env.reset()

        observation = self._normalize_observation(env.capture_observation())
        env.update()
        if not env.is_updated():
            observation["skip"] = 1.0

        if not self._observation_features:
            self._observation_features = self._infer_observation_features(observation)
        if not self._action_features:
            self._action_features = self._infer_action_features_from_env(env)

        self._prefetched_env = env
        self._metadata_initialized = True

    def _build_env(self):
        try:
            from auto_atom.basis.mjc.mujoco_env import EnvConfig, UnifiedMujocoEnv
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Failed to import auto_atom. Install the auto_atom dependency before using "
                "the 'mujoco_env' LeRobot robot."
            ) from exc

        env_config = EnvConfig(**self.config.env)
        env = UnifiedMujocoEnv(env_config)
        interest_objects, interest_operations = self.config.interests
        if interest_objects or interest_operations:
            env.set_interest_objects_and_operations(interest_objects, interest_operations)
        return env

    def _infer_observation_features(self, observation: RobotObservation) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray) and value.ndim == 3:
                features[key] = tuple(int(dim) for dim in value.shape)
            elif isinstance(value, (bool, int, float, np.number)):
                features[key] = float
        return features

    def _infer_action_features(self) -> dict[str, type]:
        return self._infer_action_features_from_env(self.env)

    def _infer_action_features_from_env(self, env) -> dict[str, type]:
        if self.config.action_keys:
            return {key: float for key in self.config.action_keys}

        action_space = getattr(env, "action_space", None)
        shape = getattr(action_space, "shape", None)
        if shape and len(shape) == 1:
            return {f"action_{i}": float for i in range(int(shape[0]))}

        return {}

    def _normalize_observation(self, observation: dict[str, Any]) -> RobotObservation:
        normalized: RobotObservation = {}
        for key, value in observation.items():
            self._flatten_observation_value(normalized, key, value)
        result: RobotObservation = {}
        state_keys = set(self.config.state_keys)
        for key, value in normalized.items():
            sanitized = key.replace("/", "_")
            if isinstance(value, np.ndarray) and value.ndim == 3:
                matched = next(
                    (
                        ik
                        for ik in self.config.image_keys
                        if sanitized == ik
                        or sanitized.startswith(ik + "_")
                        or sanitized.startswith(ik + ".")
                    ),
                    None,
                )
                result[matched if matched else sanitized] = value
            elif not state_keys or sanitized in state_keys:
                result[sanitized] = value
        return result

    def _flatten_observation_value(self, output: RobotObservation, key: str, value: Any) -> None:
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                self._flatten_observation_value(output, f"{key}.{nested_key}", nested_value)
            return

        if value is None:
            return

        if isinstance(value, np.ndarray):
            if self._is_image_value(key, value):
                output[key] = value
                return
            for index, item in enumerate(value.reshape(-1)):
                output[f"{key}_{index}"] = float(item)
            return

        if isinstance(value, (list, tuple)):
            array_value = np.asarray(value)
            if array_value.ndim == 0:
                output[key] = float(array_value.item())
                return
            if array_value.dtype == object:
                for index, item in enumerate(value):
                    self._flatten_observation_value(output, f"{key}_{index}", item)
                return
            if self._is_image_value(key, array_value):
                output[key] = array_value
                return
            for index, item in enumerate(array_value.reshape(-1)):
                output[f"{key}_{index}"] = float(item)
            return

        if isinstance(value, (bool, int, float, np.number)):
            output[key] = float(value)
            return

        output[key] = value

    def _is_image_value(self, key: str, value: np.ndarray) -> bool:
        if key in self.config.image_keys:
            return True
        return value.ndim == 3 and value.shape[-1] in (1, 3, 4) and np.issubdtype(value.dtype, np.integer)

    def _normalize_action(self, action: Any) -> Any:
        if not isinstance(action, dict):
            return action

        if "action" in action and len(action) == 1:
            return action["action"]

        if self._action_features:
            ordered_keys = list(self._action_features)
            if set(action) == set(ordered_keys):
                return np.array([action[key] for key in ordered_keys], dtype=np.float32)

        return action
