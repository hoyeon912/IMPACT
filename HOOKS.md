# IMPACT Hooks

This file defines the command conventions and validation hooks for this repo.

## Command Conventions

- Use `python3`. Do not assume `python` exists.
- Run commands from the repo root: `/Users/hoyeon/Codes/IMPACT`.
- Do not claim a hook passed if required dependencies are not installed.

## Bootstrap

- Install editable package and runtime dependencies:
```bash
python3 -m pip install -e .
```

## Baseline Smoke Checks

- Syntax smoke without writing bytecode into restricted cache locations:
```bash
PYTHONPYCACHEPREFIX=/tmp/impact-pyc python3 -m compileall impact
```

- Import smoke after bootstrap:
```bash
python3 -c "import impact; print(impact.__version__)"
```

## Validation Hooks By Change Type

- Docs-only governance changes:
  - call the Rule Maker team
  - confirm required files exist
  - confirm cross-document consistency
  - run syntax smoke only if code files changed
- For code changes, prefer plans that introduce small focused functions before larger orchestration code.

- `impact/envs/virmen_env.py` changes:
  - validate Gymnasium API behavior with a fake communication backend
  - validate observation space, action space, `reset`, `step`, `render`, `close`, and `max_steps` truncation
  - validate offline first, then note live MATLAB/VirMEn gaps

- `impact/communication/base_comm.py` changes:
  - validate interface compatibility for both concrete backends
  - check lifecycle expectations and method signatures

- `impact/communication/mmap_comm.py` changes:
  - validate memmap fixtures with known dtype, shape, and Fortran-order layout
  - validate read/write fidelity and repeated `open()` or `close()` behavior

- `impact/communication/shm_comm.py` changes:
  - validate shared-memory fixtures with known dtype, shape, and Fortran-order layout
  - validate read/write fidelity and repeated `open()` or `close()` behavior

## Optional Static Checks

- Run lint, type, or test commands only if the corresponding config or test suite exists in the repo.
- Do not document a command here as required unless the repo can actually support it.

## Live Integration Boundary

- These hooks do not replace live MATLAB/VirMEn integration testing.
- Mark validation as incomplete when a change depends on:
  - real producer-consumer synchronization timing
  - MATLAB-owned shared-memory lifecycle
  - VirMEn-generated event semantics
