# IMPACT Architecture

## Purpose

IMPACT is a small Python package that exposes a Gymnasium-compatible environment backed by VirMEn inter-process communication.

## Package Structure

- `impact/communication/base_comm.py`
  - abstract communication contract shared by all backends
- `impact/communication/mmap_comm.py`
  - memory-mapped file backend
- `impact/communication/shm_comm.py`
  - POSIX shared-memory backend
- `impact/envs/virmen_env.py`
  - Gymnasium environment that depends on `BaseCommunication`, not on a specific backend
- `impact/__init__.py`
  - public package exports

## Ownership Boundaries

- Communication backends own transport details:
  - resource names and paths
  - array construction
  - dtype and shape handling
  - memory-order compatibility
  - resource lifecycle
- `VirMEnEnv` owns environment-facing behavior:
  - observation selection
  - action submission
  - reward and termination hooks
  - Gymnasium API surface
- Public exports own the package API surface seen by downstream users.

## External Contract Boundary

- MATLAB and VirMEn are external systems.
- Python must preserve the documented IPC contract in `docs/interfaces.md`.
- Changes that alter the external contract must update both communication backends and the docs in the same change.

## Current Architectural Risks

- Synchronization is documented in the communication layer but is not enforced by `VirMEnEnv` during `reset()` or `step()`.
- Event dispatch helpers exist in `VirMEnEnv`, but event processing is not currently invoked from `step()`.
- The abstract `write_flag(value)` signature and concrete backend implementations are currently inconsistent.

