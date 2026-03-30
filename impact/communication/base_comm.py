"""Abstract base class for VirMEn communication backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseCommunication(ABC):
    """Unified interface for VirMEn <-> Python inter-process communication.

    All backends (memory-mapped files, shared memory) expose identical
    variable names so that downstream code is backend-agnostic.

    Data channels (VirMEn -> Python):
        image    : uint8   array, shape ``(H, W, C)``
        position : float64 array, shape ``(position_dim,)``
        event    : float64 array, shape ``(event_dim,)``

    Data channel (Python -> VirMEn):
        action   : float64 array, shape ``(action_dim,)``

    Synchronisation flag (shared, single uint8):
        0 — Python has written; VirMEn may proceed.
        1 — VirMEn has written; Python may proceed.
    """

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        position_dim: int,
        event_dim: int,
        action_dim: int,
    ) -> None:
        self.image_shape = image_shape
        self.position_dim = position_dim
        self.event_dim = event_dim
        self.action_dim = action_dim
        self._is_open = False

    # ------------------------------------------------------------------
    # Flag
    # ------------------------------------------------------------------

    @abstractmethod
    def read_flag(self) -> int:
        """Read the synchronisation flag.

        Returns:
            0 if Python has written (VirMEn's turn),
            1 if VirMEn has written (Python's turn).
        """

    @abstractmethod
    def write_flag(self, value: int) -> None:
        """Set the synchronisation flag to *value* (0 or 1)."""

    # ------------------------------------------------------------------
    # Read: VirMEn -> Python
    # ------------------------------------------------------------------

    @abstractmethod
    def read_image(self) -> np.ndarray:
        """
        Read the current image frame.
        Returns a copy, shape ``(H, W, C)``, dtype uint8.
        """

    @abstractmethod
    def read_position(self) -> np.ndarray:
        """
        Read the current position vector.
        Returns a copy, shape ``(position_dim,)``, dtype float64.
        """

    @abstractmethod
    def read_event(self) -> np.ndarray:
        """
        Read the current event flags.
        Returns a copy, shape ``(event_dim,)``, dtype float64.
        """

    def read_all(self) -> dict[str, np.ndarray]:
        """Read all channels at once.

        Returns:
            dict with keys ``"image"``, ``"position"``, ``"event"``.
        """
        return {
            "image": self.read_image(),
            "position": self.read_position(),
            "event": self.read_event(),
        }

    # ------------------------------------------------------------------
    # Write: Python -> VirMEn
    # ------------------------------------------------------------------

    @abstractmethod
    def write_action(self, action: np.ndarray) -> None:
        """
        Write an action array back to VirMEn.
        Shape ``(action_dim,)``, dtype float64.
        """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def open(self) -> None:
        """Open / attach to the underlying IPC resources."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying IPC resources."""

    def __enter__(self) -> BaseCommunication:
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
