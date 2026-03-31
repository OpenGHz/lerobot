#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import Any

from ..config import RobotConfig


@RobotConfig.register_subclass("mujoco_env")
@dataclass
class MujocoEnvRobotConfig(RobotConfig):
    """Configuration for wrapping a UnifiedMujocoEnv as a LeRobot robot."""

    env: dict[str, Any] = field(default_factory=dict)
    interests: tuple[list[str], list[str]] = field(default_factory=lambda: ([], []))
    action_keys: list[str] = field(default_factory=list)
    image_keys: list[str] = field(default_factory=list)
    state_keys: list[str] = field(default_factory=list)
    reset_on_connect: bool = True
