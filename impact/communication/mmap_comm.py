"""Memory-mapped file communication backend for VirMEn."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from impact.communication.base_comm import BaseCommunication


class MmapCommunication(BaseCommunication):
    """Read/write VirMEn data through memory-mapped binary files.

    MATLAB writes binary files in column-major (Fortran) order.
    This backend opens them with ``order='F'`` so that the array
    layout matches MATLAB's native format.

    Each channel maps to a separate file on disk:
        - ``flag_path``     : uint8,   shape ``(1,)``  — synchronisation flag
        - ``image_path``    : uint8,   shape ``(H, W, C)``
        - ``position_path`` : float64, shape ``(position_dim,)``
        - ``event_path``    : float64, shape ``(event_dim,)``
        - ``action_path``   : float64, shape ``(action_dim,)``
    """

    def __init__(
        self,
        flag_path: str | Path,
        image_path: str | Path,
        position_path: str | Path,
        event_path: str | Path,
        action_path: str | Path,
        image_shape: tuple[int, int, int],
        position_dim: int,
        event_dim: int,
        action_dim: int,
        mode: str = "r+",
    ) -> None:
        super().__init__(image_shape, position_dim, event_dim, action_dim)
        self._flag_path = Path(flag_path)
        self._image_path = Path(image_path)
        self._position_path = Path(position_path)
        self._event_path = Path(event_path)
        self._action_path = Path(action_path)
        self._mode = mode

        self._flag_mmap: np.memmap | None = None
        self._image_mmap: np.memmap | None = None
        self._position_mmap: np.memmap | None = None
        self._event_mmap: np.memmap | None = None
        self._action_mmap: np.memmap | None = None

    # ------------------------------------------------------------------
    # Flag
    # ------------------------------------------------------------------

    def read_flag(self) -> int:
        return int(self._flag_mmap[0])

    def write_flag(self) -> None:
        self._flag_mmap[0] = np.uint8(0)
        self._flag_mmap.flush()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_image(self) -> np.ndarray:
        return np.array(self._image_mmap, copy=True)

    def read_position(self) -> np.ndarray:
        return np.array(self._position_mmap, copy=True)

    def read_event(self) -> np.ndarray:
        return np.array(self._event_mmap, copy=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_action(self, action: np.ndarray) -> None:
        self._action_mmap[:] = np.asarray(action, dtype=np.float64)
        self._action_mmap.flush()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._is_open:
            return
        self._flag_mmap = np.memmap(
            self._flag_path,
            dtype=np.uint8,
            mode="r+",
            shape=(1,),
        )
        self._image_mmap = np.memmap(
            self._image_path,
            dtype=np.uint8,
            mode=self._mode,
            shape=self.image_shape,
            order="F",
        )
        self._position_mmap = np.memmap(
            self._position_path,
            dtype=np.float64,
            mode=self._mode,
            shape=(self.position_dim,),
            order="F",
        )
        self._event_mmap = np.memmap(
            self._event_path,
            dtype=np.float64,
            mode=self._mode,
            shape=(self.event_dim,),
            order="F",
        )
        self._action_mmap = np.memmap(
            self._action_path,
            dtype=np.float64,
            mode="r+",
            shape=(self.action_dim,),
            order="F",
        )
        self._is_open = True

    def close(self) -> None:
        attrs = (
            "_flag_mmap",
            "_image_mmap",
            "_position_mmap",
            "_event_mmap",
            "_action_mmap",
            )
        if not self._is_open:
            return
        for attr in attrs:
            mmap = getattr(self, attr, None)
            if mmap is not None:
                del mmap
            setattr(self, attr, None)
        self._is_open = False
