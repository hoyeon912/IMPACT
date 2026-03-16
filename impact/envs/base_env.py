"""Base environment class following the Gymnasium API."""

from __future__ import annotations

import abc
from typing import Any, SupportsFloat

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from impact.communication.base_comm import BaseCommunication


ObsType = dict[str, np.ndarray] | np.ndarray
ActType = Any
RenderFrame = np.ndarray


class ImpactEnv(gym.Env, abc.ABC):
    """Partially-concrete Gymnasium environment that reads observations from VirMEn.

    This class wires a :class:`~impact.communication.base_comm.BaseCommunication`
    backend into the Gymnasium ``reset`` / ``step`` loop and assembles the
    standard RL dataset tuple::

        (observation, reward, terminated, truncated, info)

    Concrete subclasses must implement three domain-specific methods:

    - :meth:`_make_action_space` – define the action space.
    - :meth:`_compute_reward` – compute a scalar reward from the current
      observation and the VirMEn event array.
    - :meth:`_is_terminated` – decide whether the episode has ended.

    Parameters
    ----------
    comm : BaseCommunication
        IPC backend used to read observations and events from VirMEn (e.g.
        :class:`~impact.communication.mmap_comm.MmapCommunication`).
    obs_type : {"vector", "image", "mixed"}
        Type of observation to expose.

        - ``"vector"`` – 1-D float32 array of shape ``(obs_dim,)``.
        - ``"image"``  – uint8 array of shape ``(image_height, image_width,
          image_channels)``.
        - ``"mixed"``  – :class:`gymnasium.spaces.Dict` containing both
          ``"image"`` and ``"position"`` keys.

    obs_dim : int
        Dimensionality of the vector observation component. Ignored when
        ``obs_type`` is ``"image"``.
    image_height : int
        Height of the image observation in pixels. Ignored when ``obs_type``
        is ``"vector"``.
    image_width : int
        Width of the image observation in pixels. Ignored when ``obs_type``
        is ``"vector"``.
    image_channels : int
        Number of colour channels (e.g. 1 for grayscale, 3 for RGB). Ignored
        when ``obs_type`` is ``"vector"``.
    max_steps : int or None
        Maximum number of steps per episode. When the step count reaches this
        value, ``step`` sets ``truncated=True``. ``None`` disables truncation.
    render_mode : str or None
        Rendering mode. Must be one of the values declared in
        :attr:`metadata` ``["render_modes"]``, or ``None``.
    """

    metadata: dict[str, Any] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        comm: BaseCommunication,
        obs_type: str = "vector",
        obs_dim: int = 8,
        image_height: int = 64,
        image_width: int = 64,
        image_channels: int = 3,
        max_steps: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        if obs_type not in {"vector", "image", "mixed"}:
            raise ValueError(
                f"obs_type must be 'vector', 'image', or 'mixed', got {obs_type!r}"
            )
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']!r}, "
                f"got {render_mode!r}"
            )

        self._comm = comm
        self.obs_type = obs_type
        self._max_steps = max_steps
        self._step_count: int = 0
        self.render_mode = render_mode

        # Build observation space ------------------------------------------
        image_space = spaces.Box(
            low=0,
            high=255,
            shape=(image_height, image_width, image_channels),
            dtype=np.uint8,
        )
        vector_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        if obs_type == "image":
            self.observation_space: spaces.Space = image_space
        elif obs_type == "vector":
            self.observation_space = vector_space
        else:  # "mixed"
            self.observation_space = spaces.Dict(
                {
                    "image": image_space,
                    "position": vector_space,
                }
            )

        # Build action space (delegated to subclass) -----------------------
        self.action_space: spaces.Space = self._make_action_space()

    # ------------------------------------------------------------------
    # Abstract methods – subclasses must implement these
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _make_action_space(self) -> spaces.Space:
        """Return the action space for this environment.

        Called once during ``__init__``. Subclasses define the concrete action
        space here (e.g. :class:`gymnasium.spaces.Discrete`,
        :class:`gymnasium.spaces.Box`).

        Returns
        -------
        gymnasium.spaces.Space
            The action space.
        """

    @abc.abstractmethod
    def _compute_reward(self, obs: ObsType, event: np.ndarray) -> float:
        """Compute a scalar reward from the current observation and VirMEn event.

        Called on every :meth:`step`.

        Parameters
        ----------
        obs : ObsType
            Current observation returned by :meth:`_get_obs`.
        event : numpy.ndarray
            Event array read from the communication channel, shape
            ``(event_dim,)``, dtype ``float64``.

        Returns
        -------
        float
            Scalar reward for the current timestep.
        """

    @abc.abstractmethod
    def _is_terminated(self, obs: ObsType, event: np.ndarray) -> bool:
        """Determine whether the episode has reached a terminal state.

        Called on every :meth:`step`.

        Parameters
        ----------
        obs : ObsType
            Current observation returned by :meth:`_get_obs`.
        event : numpy.ndarray
            Event array read from the communication channel, shape
            ``(event_dim,)``, dtype ``float64``.

        Returns
        -------
        bool
            ``True`` if the episode should end (MDP terminal state).
        """

    # ------------------------------------------------------------------
    # Concrete Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and return the initial observation.

        Opens the communication channel (idempotent), resets the step counter,
        reads the initial observation and event from VirMEn.

        Parameters
        ----------
        seed : int or None
            Seed for ``self.np_random``.
        options : dict or None
            Unused; reserved for subclass extensions.

        Returns
        -------
        observation : ObsType
            Initial observation from VirMEn, consistent with
            :attr:`observation_space`.
        info : dict
            Auxiliary info containing ``"event"`` (the current VirMEn event
            array).
        """
        super().reset(seed=seed)
        self._comm.open()
        self._step_count = 0
        obs = self._get_obs()
        event = self._comm.read_event()
        return obs, {"event": event}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics.

        Reads the next observation and event from VirMEn, computes reward and
        termination via the abstract helpers, and checks the step-count
        truncation limit.

        Parameters
        ----------
        action : ActType
            An action provided by the agent, element of :attr:`action_space`.
            Delivery of the action to VirMEn is handled by the concrete
            subclass (e.g. by writing to the communication channel).

        Returns
        -------
        observation : ObsType
            Observation of the environment after the action.
        reward : float
            Scalar reward from :meth:`_compute_reward`.
        terminated : bool
            ``True`` if :meth:`_is_terminated` signals a terminal state.
        truncated : bool
            ``True`` if the step count has reached ``max_steps``.
        info : dict
            Contains ``"event"`` (VirMEn event array) and ``"step"`` (current
            step count).
        """
        self._step_count += 1
        obs = self._get_obs()
        event = self._comm.read_event()
        reward = self._compute_reward(obs, event)
        terminated = self._is_terminated(obs, event)
        truncated = (
            self._max_steps is not None and self._step_count >= self._max_steps
        )
        info: dict[str, Any] = {"event": event, "step": self._step_count, "position": obs.vector}
        return obs, reward, terminated, truncated, info

    def render(self) -> RenderFrame | None:
        """Return a render frame from the current image observation.

        Returns
        -------
        numpy.ndarray or None
            An ``(H, W, C)`` uint8 RGB array when ``render_mode`` is
            ``"rgb_array"`` and ``obs_type`` includes an image channel.
            ``None`` otherwise.
        """
        if self.render_mode == "rgb_array" and self.obs_type in ("image", "mixed"):
            return self._comm.read_image()
        return None

    def close(self) -> None:
        """Close the communication channel and release resources."""
        self._comm.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> ObsType:
        """Read and return the current observation from the comm channel.

        Returns
        -------
        ObsType
            Consistent with :attr:`observation_space`:

            - ``"vector"`` → ``(obs_dim,)`` float32 array.
            - ``"image"``  → ``(H, W, C)`` uint8 array.
            - ``"mixed"``  → dict with ``"image"`` and ``"position"`` keys.
        """
        if self.obs_type == "image":
            return self._comm.read_image()
        elif self.obs_type == "vector":
            return self._comm.read_vector().astype(np.float32)
        else:  # "mixed"
            return {
                "image": self._comm.read_image(),
                "position": self._comm.read_vector().astype(np.float32),
            }
