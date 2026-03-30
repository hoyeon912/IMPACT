"""POSIX shared memory communication backend for VirMEn."""

from __future__ import annotations

from multiprocessing.shared_memory import SharedMemory

import numpy as np

from impact.communication.base_comm import BaseCommunication


class ShmCommunication(BaseCommunication):
    """Read/write VirMEn data through POSIX shared memory blocks.

    MATLAB creates and owns the shared memory blocks.  Python attaches
    as a secondary consumer (``create=False``) and **never** unlinks
    them — only closes its own handles.

    Arrays are constructed with ``order='F'`` to match MATLAB's
    column-major memory layout.

    Each channel maps to a named shared memory block:
        - ``flag_name``     : uint8,   shape ``(1,)``  — synchronisation flag
        - ``image_name``    : uint8,   shape ``(H, W, C)``
        - ``position_name`` : float64, shape ``(position_dim,)``
        - ``event_name``    : float64, shape ``(event_dim,)``
        - ``action_name``   : float64, shape ``(action_dim,)``
    """

    def __init__(
        self,
        flag_name: str,
        image_name: str,
        position_name: str,
        event_name: str,
        action_name: str,
        image_shape: tuple[int, int, int],
        position_dim: int,
        event_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__(image_shape, position_dim, event_dim, action_dim)
        self._flag_name = flag_name
        self._image_name = image_name
        self._position_name = position_name
        self._event_name = event_name
        self._action_name = action_name

        self._flag_shm: SharedMemory | None = None
        self._image_shm: SharedMemory | None = None
        self._position_shm: SharedMemory | None = None
        self._event_shm: SharedMemory | None = None
        self._action_shm: SharedMemory | None = None

        self._flag_array: np.ndarray | None = None
        self._image_array: np.ndarray | None = None
        self._position_array: np.ndarray | None = None
        self._event_array: np.ndarray | None = None
        self._action_array: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Flag
    # ------------------------------------------------------------------

    def read_flag(self) -> int:
        return int(self._flag_array[0])

    def write_flag(self) -> None:
        self._flag_array[0] = np.uint8(0)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_image(self) -> np.ndarray:
        return np.array(self._image_array, copy=True)

    def read_position(self) -> np.ndarray:
        return np.array(self._position_array, copy=True)

    def read_event(self) -> np.ndarray:
        return np.array(self._event_array, copy=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_action(self, action: np.ndarray) -> None:
        self._action_array[:] = np.asarray(action, dtype=np.float64)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._is_open:
            return

        self._flag_shm = SharedMemory(name=self._flag_name, create=False)
        self._flag_array = np.ndarray(
            (1,),
            dtype=np.uint8,
            buffer=self._flag_shm.buf,
        )

        self._image_shm = SharedMemory(name=self._image_name, create=False)
        self._image_array = np.ndarray(
            self.image_shape,
            dtype=np.uint8,
            buffer=self._image_shm.buf,
            order="F",
        )

        self._position_shm = SharedMemory(
                name=self._position_name,
                create=False
                )
        self._position_array = np.ndarray(
            (self.position_dim,),
            dtype=np.float64,
            buffer=self._position_shm.buf,
            order="F",
        )

        self._event_shm = SharedMemory(name=self._event_name, create=False)
        self._event_array = np.ndarray(
            (self.event_dim,),
            dtype=np.float64,
            buffer=self._event_shm.buf,
            order="F",
        )

        self._action_shm = SharedMemory(name=self._action_name, create=False)
        self._action_array = np.ndarray(
            (self.action_dim,),
            dtype=np.float64,
            buffer=self._action_shm.buf,
            order="F",
        )

        self._is_open = True

    def close(self) -> None:
        attrs = ("_flag_shm", "_image_shm", "_position_shm", "_event_shm", "_action_shm")
        if not self._is_open:
            return
        for attr in attrs:
            shm = getattr(self, attr, None)
            if shm is not None:
                shm.close()
            setattr(self, attr, None)
        self._flag_array = None
        self._image_array = None
        self._position_array = None
        self._event_array = None
        self._action_array = None
        self._is_open = False
