# IMPACT Testing

## Validation Model

Every change must classify validation as:

- `offline validated`
- `requires live MATLAB/VirMEn`

Do not merge those categories into a single claim.

## Required Offline Coverage

- Prefer implementations that keep smaller units independently testable offline before they are assembled into larger flows.

### For Environment Changes

- Use a fake `BaseCommunication` implementation.
- Validate:
  - observation space creation
  - action space creation
  - `reset()` behavior
  - `step()` return shape and ordering
  - `max_steps` truncation
  - `render()` behavior
  - `close()` behavior

### For Communication Backend Changes

- Validate both concrete backends when shared interface behavior changes.
- Use deterministic fixtures with known data.

For `MmapCommunication`:
- create temporary memmap files
- validate dtype, shape, and Fortran-order assumptions
- validate read fidelity
- validate action writes
- validate repeated `open()` and `close()` behavior

For `ShmCommunication`:
- create temporary shared-memory blocks
- validate dtype, shape, and Fortran-order assumptions
- validate read fidelity
- validate action writes
- validate repeated `open()` and `close()` behavior

### For Interface Changes

- Validate that `BaseCommunication`, `MmapCommunication`, and `ShmCommunication` stay aligned.
- Explicitly test method signatures when synchronization or lifecycle behavior changes.

## Required Smoke Checks

- Syntax smoke:
```bash
PYTHONPYCACHEPREFIX=/tmp/impact-pyc python3 -m compileall impact
```

- Import smoke after dependency bootstrap:
```bash
python3 -c "import impact; print(impact.__version__)"
```

## Live MATLAB/VirMEn Boundary

Offline validation is not enough when a change depends on:

- true producer-consumer timing
- MATLAB-created shared-memory ownership or cleanup
- real VirMEn event generation
- end-to-end synchronization semantics

Mark these as `requires live MATLAB/VirMEn`.

## Governance-Only Changes

For changes limited to `AGENTS.md`, `HOOKS.md`, or `docs/`:

- verify the Rule Maker team was used for the document edit
- verify all required governance files exist
- verify cross-document consistency
- verify commands in `HOOKS.md` match the current repo reality
- do not claim runtime behavior was validated unless code or dependencies were actually exercised

## Current Repo Constraints

- The repo does not currently contain a unit test suite.
- Runtime import checks depend on installed package dependencies such as `numpy` and `gymnasium`.
- If bootstrap dependencies are missing, report that boundary instead of treating the hook as passed.
