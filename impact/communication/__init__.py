"""Communication backends for VirMEn <-> Python IPC."""

from impact.communication.base_comm import BaseCommunication
from impact.communication.mmap_comm import MmapCommunication
from impact.communication.shm_comm import ShmCommunication

__all__ = [
    "BaseCommunication",
    "MmapCommunication",
    "ShmCommunication",
]
