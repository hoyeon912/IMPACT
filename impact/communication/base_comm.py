"""Abstract base class for VirMEn ↔ Python communication backends."""

from __future__ import annotations

import abc
from typing import Any

import numpy as np


class BaseCommunication(abc.ABC):
    """Abstract interface for reading VirMEn observations over an IPC channel.

    All backends (memory-mapped files, shared memory, sockets, …) must
    implement this interface.  Three observation channels are defined:

    - **image** – a raw pixel frame, shape ``(H, W, C)``, dtype ``uint8``.
    - **vector** – a 1-D array of behavioural/state features, dtype ``float64``.
    - **event** – a 1-D array of trial/event flags, dtype ``float64``.

    Lifecycle
    ---------
    Call :meth:`open` before reading and :meth:`close` when done.  The class
    also supports the context-manager protocol::

        with MmapCommunication(...) as comm:
            obs = comm.read_all()

    Subclasses must implement :meth:`open`, :meth:`close`, :meth:`read_image`,
    :meth:`read_vector`, and :meth:`read_event`.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def open(self) -> None:
        """Open / attach to the IPC channel.

        Must be called before any ``read_*`` method.  Idempotent
        implementations are encouraged.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources associated with the IPC channel.

        After calling this method, calling ``read_*`` methods has undefined
        behaviour.  Idempotent implementations are encouraged.
        """

    def __enter__(self) -> BaseCommunication:
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Channel readers
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def read_image(self) -> np.ndarray:
        """Read the current image observation from VirMEn.

        Returns
        -------
        numpy.ndarray
            Shape ``(H, W, C)``, dtype ``uint8``.  A fresh copy is returned
            so that the caller owns the data independently of the IPC buffer.
        """

    @abc.abstractmethod
    def read_vector(self) -> np.ndarray:
        """Read the current vector observation from VirMEn.

        Returns
        -------
        numpy.ndarray
            Shape ``(obs_dim,)``, dtype ``float64``.
        """

    @abc.abstractmethod
    def read_event(self) -> np.ndarray:
        """Read the current event flags from VirMEn.

        Returns
        -------
        numpy.ndarray
            Shape ``(event_dim,)``, dtype ``float64``.
        """

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def read_all(self) -> dict[str, np.ndarray]:
        """Read all channels and return them in a single dict.

        Returns
        -------
        dict
            Keys: ``"image"``, ``"vector"``, ``"event"``.
        """
        return {
            "image": self.read_image(),
            "vector": self.read_vector(),
            "event": self.read_event(),
        }
