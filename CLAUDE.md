# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IMPACT is a Python package providing Gymnasium-compatible RL environments for VirMEn (Virtual Multi-Sensory Environment) experiments. It bridges MATLAB-based VirMEn and Python RL agents through pluggable IPC backends.

## Build & Install

```bash
pip install -e .          # Development install
pip install .             # Production install
```

Dependencies: `numpy`, `gymnasium` (Python >= 3.9)

No tests exist yet. No linter/formatter configured.

## Architecture

Two layers, both under `impact/`:

### Communication Layer (`impact/communication/`)

Abstract `BaseCommunication` defines four data channels for VirMEn <-> Python IPC:
- **Read** (VirMEn -> Python): `image` (uint8 HxWxC), `position` (float64), `event` (float64)
- **Write** (Python -> VirMEn): `action` (float64)

Two backends implement this interface:
- `MmapCommunication` — memory-mapped files on disk. Each channel is a separate binary file.
- `ShmCommunication` — POSIX shared memory. MATLAB owns the memory blocks; Python only attaches (`create=False`), never creates or unlinks them.

### Environment Layer (`impact/envs/`)

`VirMEnEnv` wraps any `BaseCommunication` in the Gymnasium `Env` API. Registered as `"impact/VirMEn-v0"`. Supports `obs_type="image"` or `"position"`. Subclass and override `_compute_reward()` and `_check_terminated()` for task-specific logic.

## Critical Conventions

- **Fortran memory order**: All arrays use `order='F'` to match MATLAB's column-major layout. This is essential for correct data interpretation.
- **Read operations return copies**: Backends never expose direct references to shared memory/mmap.
- **Context manager protocol**: Communication backends support `with` statements for safe resource management.
- **`from __future__ import annotations`**: Used throughout for modern type hint syntax.
