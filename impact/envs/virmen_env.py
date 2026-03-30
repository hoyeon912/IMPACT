"""Gymnasium environment for VirMEn virtual-reality experiments."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from impact.communication.base_comm import BaseCommunication


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
        self.comm.write_action(np.asarray(action, dtype=np.float64))
        self._step_count += 1

        obs = self._get_obs()
        position = self.comm.read_position()
        event = self.comm.read_event()

        reward = self._compute_reward(obs, position, event, action)
        terminated = self._check_terminated(event)
        truncated = self._step_count >= self.max_steps
        info = {"position": position, "event": event}

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
        obs: np.ndarray,
        position: np.ndarray,
        event: np.ndarray,
        action: np.ndarray,
    ) -> float:
        """Compute the scalar reward. Override in subclasses."""
        return 0.0

    def _check_terminated(self, event: np.ndarray) -> bool:
        """Check whether the episode has terminated. Override in subclasses."""
        return False

    def _process_event(self, event: np.ndarray) -> None:
        """Process the event array by dispatching each event code.

        Override individual ``_on_*`` handlers in subclasses to implement
        task-specific event logic.

        Args:
            event: float64 array of shape ``(event_dim,)`` from VirMEn.

        Returns:
            Dict mapping event names to their processed results.
        """
        code = int(event[0]) if len(event) > 0 else -1

        if code == 0:
            self._on_trial_start(event)
        elif code == 1:
            self._on_trial_end(event)
        elif code == 2:
            self._on_reward(event)
        elif code == 3:
            self._on_shock(event)
        elif code == 4:
            self._on_cue_onset(event)

    def _on_trial_start(self, event: np.ndarray) -> Any:
        """Handle trial start event (code 1). Override in subclasses."""
        pass

    def _on_trial_end(self, event: np.ndarray) -> Any:
        """Handle trial end event (code 2). Override in subclasses."""
        pass

    def _on_reward(self, event: np.ndarray) -> Any:
        """Handle reward event (code 3). Override in subclasses."""
        pass

    def _on_shock(self, event: np.ndarray) -> Any:
        """Handle shock event (code 4). Override in subclasses."""
        pass

    def _on_cue_onset(self, event: np.ndarray) -> Any:
        """Handle cue onset event (code 5). Override in subclasses."""
        pass


# ------------------------------------------------------------------
# Gymnasium registration
# ------------------------------------------------------------------
gymnasium.register(
    id="impact/VirMEn-v0",
    entry_point="impact.envs.virmen_env:VirMEnEnv",
)
