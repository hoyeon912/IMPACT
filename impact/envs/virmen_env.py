"""Gymnasium environment for VirMEn virtual-reality experiments."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
import json

from impact.communication.base_comm import BaseCommunication
from impact.communication.mmap_comm import MmapCommunication


class VirMEnEnv(gymnasium.Env):
    """Gymnasium-style RL environment wrapping a VirMEn communication backend.

    Observations can be either the rendered image or the position vector,
    selected via ``obs_type``.  Regardless of the chosen observation type,
    the position is **always** read from VirMEn and included in the ``info``
    dict returned by :meth:`step` and :meth:`reset`.

    Args:
        comm: A :class:`BaseCommunication` instance (mmap or shm backend).
        obs_type: ``"image"`` for pixel observations or ``"position"`` for
            the state vector.
        render_mode: ``"rgb_array"`` to return rendered frames, or ``None``.
        max_steps: Maximum number of steps before the episode is truncated.
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"]}
    TRIAL_ON = 0
    STIMULUS_ON = 1
    REWARD_ON = 2
    SHOCK_ON = 3
    TRIAL_END = 4

    def __init__(
        self,
        comm: BaseCommunication,
        obs_type: str = "image",
        render_mode: str | None = None,
        max_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.comm = comm
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.max_steps = max_steps

        if obs_type == "image":
            self.observation_space = gymnasium.spaces.Box(
                low=0,
                high=255,
                shape=comm.image_shape,
                dtype=np.uint8,
            )
        elif obs_type == "position":
            self.observation_space = gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(comm.position_dim,),
                dtype=np.float64,
            )
        else:
            raise ValueError(
                f"obs_type must be 'image' or 'position', got '{obs_type}'"
            )

        self.action_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(comm.action_dim,),
            dtype=np.float64,
        )

        self._step_count = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.comm.open()
        self._step_count = 0
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.comm.write_action(np.asarray(action, dtype=np.uint8))
        self._step_count += 1

        obs = self._get_obs()
        info = self._get_info()

        reward = self._compute_reward(action, info["event"])
        terminated = self._check_terminated(info["event"])
        truncated = self._step_count >= self.max_steps

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self.comm.read_image()
        return None

    def close(self) -> None:
        self.comm.close()
        super().close()

    # ------------------------------------------------------------------
    # Helpers — override in subclasses for task-specific logic
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        if self.obs_type == "image":
            return self.comm.read_image()
        return self.comm.read_position()

    def _get_info(self) -> dict[str, Any]:
        return {
            "position": self.comm.read_position(),
            "event": self.comm.read_event(),
        }

    def _compute_reward(
            self,
            action: int | np.ndarray,
            event: np.ndarray) -> float:
        """Compute the scalar reward. Override in subclasses."""
        return 0.0

    def _check_terminated(self, event: np.ndarray) -> bool:
        """Check whether the episode has terminated. Override in subclasses."""
        return False


class OpenLoop1D(VirMEnEnv):
    """1D open-loop VirMEn environment with stop/lick actions."""

    STOP_ACTION = 0
    LICK_ACTION = 1

    def __init__(
        self,
        comm: MmapCommunication,
        obs_type: str = "image",
        render_mode: str | None = None,
        max_steps: int = 1000,
        setting_path: str = "setting.json",
    ) -> None:
        super().__init__(
            comm=comm,
            obs_type=obs_type,
            render_mode=render_mode,
            max_steps=max_steps,
        )
        self.action_space = gymnasium.spaces.Discrete(2)
        self._read_setting(setting_path)

    def _compute_reward(
            self,
            action: int | np.ndarray,
            event: np.ndarray
            ) -> float:
        if event[0] == self.REWARD_ON and action == self.LICK_ACTION:
            return self._reward_value
        else:
            return self._action_cost[action]

    def _read_setting(self, fpath: str) -> None:
        with open(fpath, "r") as f:
            settings = json.load(f)
            self._reward_value = settings.get("reward_value", 0.0)
            self._action_cost = settings.get("action_cost", [0.0, 0.0])


# ------------------------------------------------------------------
# Gymnasium registration
# ------------------------------------------------------------------
gymnasium.register(
    id="impact/OpenLoop1D-v0",
    entry_point="impact.envs.virmen_env:OpenLoop1D",
)
