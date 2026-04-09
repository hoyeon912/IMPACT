# IMPACT Interfaces

## Communication Channels

These are the authoritative channel contracts for IMPACT.

### VirMEn -> Python

- `image`
  - dtype: `uint8`
  - shape: `(H, W, C)`
  - semantics: current rendered frame
- `position`
  - dtype: `float64`
  - shape: `(position_dim,)`
  - semantics: position/state vector
- `event`
  - dtype: `float64`
  - shape: `(event_dim,)`
  - semantics: event payload from VirMEn

### Python -> VirMEn

- `action`
  - dtype: `float64`
  - shape: `(action_dim,)`
  - semantics: action chosen by the Python agent

## Memory Layout

- Communication buffers must preserve MATLAB-compatible column-major layout.
- `MmapCommunication` and `ShmCommunication` must continue to treat shared arrays as Fortran-order data where documented.
- Any change to shape, dtype, or memory order must be documented in the plan and validated in both backends.

## Synchronization Flag

- dtype: `uint8`
- shape: `(1,)`
- meanings:
  - `0`: Python has written; VirMEn may proceed
  - `1`: VirMEn has written; Python may proceed

## Intended Interface Contract

- `BaseCommunication.read_flag() -> int`
- `BaseCommunication.write_flag(value: int) -> None`
- `BaseCommunication.read_image() -> np.ndarray`
- `BaseCommunication.read_position() -> np.ndarray`
- `BaseCommunication.read_event() -> np.ndarray`
- `BaseCommunication.write_action(action: np.ndarray) -> None`
- `BaseCommunication.open() -> None`
- `BaseCommunication.close() -> None`

## Current Known Contract Drift

- The abstract interface declares `write_flag(value: int)`.
- The current concrete backends implement `write_flag()` without a `value` parameter and hardcode the written value.
- Any future synchronization change must reconcile this drift explicitly before claiming interface consistency.

## Environment Contract

- `VirMEnEnv` supports:
  - `obs_type="image"`
  - `obs_type="position"`
- `reset()` opens the communication backend and returns `(obs, info)`.
- `step(action)` writes the action, reads observation and info, computes reward, and returns `(obs, reward, terminated, truncated, info)`.
- `render()` returns an RGB array only when `render_mode == "rgb_array"`.

## Authoritative Event Codes

Use the following mapping as the source of truth for this repo:

- `0`: trial start
- `1`: trial end
- `2`: reward
- `3`: shock
- `4`: cue onset

## Event Handling Rule

- Any change touching event behavior must update:
  - the mapping above
  - the relevant code paths in `VirMEnEnv`
  - the validation plan in `docs/testing.md`
- Reviewer must reject plans that leave event-code meaning ambiguous.

## Current Event Caveat

- `VirMEnEnv._process_event()` contains dispatch logic for the mapping above.
- The handler docstrings in the current code do not fully match that mapping.
- `_process_event()` is not currently called from `step()`.
- Future changes must state whether they are documenting the current implementation gap or fixing it.

