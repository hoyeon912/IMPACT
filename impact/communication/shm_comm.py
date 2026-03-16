"""Abstract base class for POSIX shared-memory communication backends.

This module defines :class:`ShmCommunication`, an intermediate abstract class
between :class:`~impact.communication.base_comm.BaseCommunication` and a
concrete POSIX / named shared-memory implementation.

Motivation
----------
Unlike memory-mapped files, POSIX shared-memory blocks reside entirely in RAM
and are not backed by a file on disk.  They are identified by a name string
(e.g. ``"/virmen_image"``) and can be accessed from any process—including
MATLAB MEX functions—using the standard ``shm_open`` / ``mmap`` POSIX API.

This makes shared memory faster than file-based mmap for high-frequency,
low-latency communication (e.g. per-frame image transfer at >60 Hz).

Subclass responsibilities
-------------------------
Concrete subclasses must implement **five** abstract methods:

1. :meth:`_attach_blocks` – attach to pre-existing named SHM blocks created
   by VirMEn/MATLAB.
2. :meth:`_detach_blocks` – detach without destroying the blocks (MATLAB owns
   them).
3. :meth:`read_image` – copy the image frame out of the SHM block.
4. :meth:`read_vector` – copy the vector observation out of the SHM block.
5. :meth:`read_event` – copy the event array out of the SHM block.

Subclasses may also add synchronisation primitives (POSIX semaphores, etc.)
to coordinate read/write access with MATLAB.

Example skeleton
----------------
.. code-block:: python

    from multiprocessing.shared_memory import SharedMemory
    import numpy as np
    from impact.communication.shm_comm import ShmCommunication

    class PosixShmCommunication(ShmCommunication):
        def _attach_blocks(self):
            self._shm_image  = SharedMemory(name=self._image_name)
            self._shm_vector = SharedMemory(name=self._vector_name)
            self._shm_event  = SharedMemory(name=self._event_name)

        def _detach_blocks(self):
            self._shm_image.close()
            self._shm_vector.close()
            self._shm_event.close()

        def read_image(self) -> np.ndarray:
            buf = np.ndarray(self._image_shape, dtype=np.uint8,
                             buffer=self._shm_image.buf, order="F")
            return buf.copy()

        def read_vector(self) -> np.ndarray:
            buf = np.ndarray((self._obs_dim,), dtype=np.float64,
                             buffer=self._shm_vector.buf)
            return buf.copy()

        def read_event(self) -> np.ndarray:
            buf = np.ndarray((self._event_dim,), dtype=np.float64,
                             buffer=self._shm_event.buf)
            return buf.copy()
"""

from __future__ import annotations

import abc

from impact.communication.base_comm import BaseCommunication


class ShmCommunication(BaseCommunication, abc.ABC):
    """Abstract base for shared-memory IPC backends.

    Provides a concrete lifecycle (``open`` / ``close``) delegating to the
    abstract :meth:`_attach_blocks` and :meth:`_detach_blocks` hooks.
    Channel readers (:meth:`read_image`, :meth:`read_vector`,
    :meth:`read_event`) remain abstract and must be implemented by the
    concrete subclass.

    Notes
    -----
    - Python's :mod:`multiprocessing.shared_memory` (Python ≥ 3.8) only
      manages the lifecycle on the Python side.  For cross-language SHM
      (MATLAB ↔ Python), MATLAB must create the block first (via a MEX file
      calling ``shm_open`` + ``mmap``), and Python attaches as a secondary
      owner.
    - Synchronisation (POSIX semaphores, spin-locks, double-buffering) is the
      responsibility of the concrete subclass.
    - Do **not** call ``SharedMemory(..., create=True)`` from the Python side;
      MATLAB owns the block lifetime.
    """

    # ------------------------------------------------------------------
    # Lifecycle (concrete)
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Attach to existing shared-memory blocks created by VirMEn."""
        self._attach_blocks()

    def close(self) -> None:
        """Detach from shared-memory blocks without destroying them."""
        self._detach_blocks()

    # ------------------------------------------------------------------
    # SHM lifecycle hooks (abstract)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _attach_blocks(self) -> None:
        """Attach to pre-existing named shared-memory blocks.

        Called by :meth:`open`.  The blocks are assumed to have been created
        by VirMEn/MATLAB before this method is invoked.

        Implementations should store references to the attached blocks as
        instance attributes for use in the ``read_*`` methods.
        """

    @abc.abstractmethod
    def _detach_blocks(self) -> None:
        """Detach from shared-memory blocks.

        Called by :meth:`close`.  Must **not** unlink or destroy the blocks—
        that is MATLAB's responsibility.  Only close the Python-side handles.
        """

    # ------------------------------------------------------------------
    # Channel readers – remain abstract; implemented by concrete subclass
    # ------------------------------------------------------------------
    # (Inherited from BaseCommunication via abc.abstractmethod)
    #
    # @abc.abstractmethod
    # def read_image(self) -> np.ndarray: ...
    #
    # @abc.abstractmethod
    # def read_vector(self) -> np.ndarray: ...
    #
    # @abc.abstractmethod
    # def read_event(self) -> np.ndarray: ...
