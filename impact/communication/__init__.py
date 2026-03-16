"""VirMEn ↔ Python communication backends.

Backends
--------
:class:`MmapCommunication`
    Reads VirMEn observations from memory-mapped binary files (current method).
:class:`ShmCommunication`
    Abstract base for future POSIX shared-memory backends.
:class:`BaseCommunication`
    Abstract interface implemented by all backends.
"""

from impact.communication.base_comm import BaseCommunication
from impact.communication.mmap_comm import MmapCommunication
from impact.communication.shm_comm import ShmCommunication

__all__ = [
    "BaseCommunication",
    "MmapCommunication",
    "ShmCommunication",
]
