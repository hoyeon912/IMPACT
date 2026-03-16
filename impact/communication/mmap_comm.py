"""Memory-mapped file communication backend for VirMEn observations.

MATLAB writes each channel to a raw binary file via ``fwrite``.  Python opens
the same files as ``numpy.memmap`` arrays in read-only mode.

Expected MATLAB write order
---------------------------
All arrays must be written in **column-major (Fortran) order**, which is
MATLAB's native layout.  ``numpy.memmap`` is opened with ``order='F'`` to
match.

File layout
-----------
Each channel lives in its own file:

==============================  ========  ===========================
File                            dtype     shape
==============================  ========  ===========================
``<image_path>``                uint8     ``(H, W, C)``
``<vector_path>``               float64   ``(obs_dim,)``
``<event_path>``                float64   ``(event_dim,)``
==============================  ========  ===========================
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np

from impact.communication.base_comm import BaseCommunication

PathLike = Union[str, os.PathLike]


class MmapCommunication(BaseCommunication):
    """Read VirMEn observations from memory-mapped binary files.

    Parameters
    ----------
    image_path : path-like
        Path to the memory-mapped file for the image channel.
    vector_path : path-like
        Path to the memory-mapped file for the vector channel.
    event_path : path-like
        Path to the memory-mapped file for the event channel.
    image_shape : tuple of int
        Shape of the image array as ``(height, width, channels)``.
    obs_dim : int
        Number of elements in the vector observation.
    event_dim : int
        Number of elements in the event array.
    mode : {"r", "r+", "c"}
        ``numpy.memmap`` mode.  Use ``"r"`` (read-only, default) when Python
        only reads and MATLAB writes.  ``"r+"`` allows Python to write back
        (e.g. for handshake flags).  ``"c"`` is copy-on-write.

    Examples
    --------
    >>> comm = MmapCommunication(
    ...     image_path="virmen_image.bin",
    ...     vector_path="virmen_vector.bin",
    ...     event_path="virmen_event.bin",
    ...     image_shape=(128, 128, 3),
    ...     obs_dim=16,
    ...     event_dim=2,
    ... )
    >>> with comm:
    ...     obs = comm.read_all()
    """

    def __init__(
        self,
        image_path: PathLike = "virmen_image.bin",
        vector_path: PathLike = "virmen_vector.bin",
        event_path: PathLike = "virmen_event.bin",
        image_shape: tuple[int, int, int] = (64, 64, 3),
        obs_dim: int = 8,
        event_dim: int = 1,
        mode: str = "r",
    ) -> None:
        self._image_path = Path(image_path)
        self._vector_path = Path(vector_path)
        self._event_path = Path(event_path)
        self._image_shape = image_shape
        self._obs_dim = obs_dim
        self._event_dim = event_dim
        self._mode = mode

        self._mmap_image: np.memmap | None = None
        self._mmap_vector: np.memmap | None = None
        self._mmap_event: np.memmap | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open memory-mapped views for all three channels.

        The binary files must already exist and have the correct size.
        """
        if self._mmap_image is not None:
            return  # already open

        self._mmap_image = np.memmap(
            self._image_path,
            dtype=np.uint8,
            mode=self._mode,
            shape=self._image_shape,
            order="F",
        )
        self._mmap_vector = np.memmap(
            self._vector_path,
            dtype=np.float64,
            mode=self._mode,
            shape=(self._obs_dim,),
        )
        self._mmap_event = np.memmap(
            self._event_path,
            dtype=np.float64,
            mode=self._mode,
            shape=(self._event_dim,),
        )

    def close(self) -> None:
        """Delete the memmap handles and release OS file references."""
        if self._mmap_image is None:
            return  # already closed

        del self._mmap_image
        del self._mmap_vector
        del self._mmap_event

        self._mmap_image = None
        self._mmap_vector = None
        self._mmap_event = None

    # ------------------------------------------------------------------
    # Channel readers
    # ------------------------------------------------------------------

    def read_image(self) -> np.ndarray:
        """Read the image channel.

        Returns
        -------
        numpy.ndarray
            Shape ``(H, W, C)``, dtype ``uint8``.  A copy of the current
            memmap view is returned so that subsequent MATLAB writes do not
            mutate the returned array.

        Raises
        ------
        RuntimeError
            If :meth:`open` has not been called.
        """
        self._require_open()
        return np.array(self._mmap_image)

    def read_vector(self) -> np.ndarray:
        """Read the vector observation channel.

        Returns
        -------
        numpy.ndarray
            Shape ``(obs_dim,)``, dtype ``float64``.

        Raises
        ------
        RuntimeError
            If :meth:`open` has not been called.
        """
        self._require_open()
        return np.array(self._mmap_vector)

    def read_event(self) -> np.ndarray:
        """Read the event channel.

        Returns
        -------
        numpy.ndarray
            Shape ``(event_dim,)``, dtype ``float64``.

        Raises
        ------
        RuntimeError
            If :meth:`open` has not been called.
        """
        self._require_open()
        return np.array(self._mmap_event)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_open(self) -> None:
        if self._mmap_image is None:
            raise RuntimeError(
                "MmapCommunication is not open. Call open() or use it as a "
                "context manager before reading."
            )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "open" if self._mmap_image is not None else "closed"
        return (
            f"{type(self).__name__}("
            f"image_path={str(self._image_path)!r}, "
            f"vector_path={str(self._vector_path)!r}, "
            f"event_path={str(self._event_path)!r}, "
            f"image_shape={self._image_shape}, "
            f"obs_dim={self._obs_dim}, "
            f"event_dim={self._event_dim}, "
            f"status={status!r})"
        )
