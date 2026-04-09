# IMPACT Agent Workflow

This repository uses two permanent teams.

## Team 1: Code Revise

### Editor
- Read the relevant code, files, and surrounding context before proposing changes.
- Identify affected surfaces explicitly. For this repo, always check:
  - `impact/communication/base_comm.py`
  - `impact/communication/mmap_comm.py`
  - `impact/communication/shm_comm.py`
  - `impact/envs/virmen_env.py`
  - package exports when public behavior changes
- Write an implementation plan before editing.
- The plan must include:
  - relevant files to inspect
  - change scope
  - goal of the change
  - step-by-step implementation approach
  - risks or side effects
  - rules or docs that must be checked
- Do not implement before explicit Reviewer approval.
- After approval, implement the change and provide an Implementation Summary.

### Reviewer
- Review Editor and Tester plans against:
  - `AGENTS.md`
  - `HOOKS.md`
  - files in `docs/`
- Approve or reject explicitly.
- If rejecting, explain:
  - what is wrong
  - which rule or repo fact conflicts with the plan
  - what must be revised
- Reject plans that:
  - skip backend parity for communication changes
  - change IPC contracts without stating dtype, shape, or memory-order impact
  - assume synchronization already works safely
  - touch event behavior without defining the authoritative event-code mapping and dispatch point
  - omit offline validation scope
  - rely on repo rules that do not exist
- If `HOOKS.md` or `docs/` are missing, stale, or incomplete, mark the review as evidence-limited using these labels:
  - `observed`
  - `inferred`
  - `missing evidence`
- Do not invent undocumented policy silently.

### Tester
- Create a validation plan only after the Editor implementation is complete.
- Split validation into:
  - `offline validated`
  - `requires live MATLAB/VirMEn`
- Use the validation rules in `docs/testing.md` and the commands in `HOOKS.md`.
- Do not finalize validation before explicit Reviewer approval.
- After approval, run the validation that is available and provide a Test Summary.

## Team 2: Rule Maker

### Analyzer
- Review `AGENTS.md`, `HOOKS.md`, and `docs/` after completed Code Revise cycles.
- Always be called when a document is being edited.
- Focus on missing or weak rules around:
  - IPC contracts
  - synchronization semantics
  - backend equivalence
  - event-code mapping
  - lifecycle rules
  - testing requirements

### Summarizer
- Read completed Code Revise conversations.
- Extract only recurring, fixable process problems.
- Ignore one-off mistakes unless they show a missing rule or ambiguous requirement.
- Summaries must include:
  - trigger
  - failure pattern
  - affected rule
  - proposed fix

### Writer
- Update `AGENTS.md`, `HOOKS.md`, and `docs/` from Summarizer output.
- Handle all document edits through the Rule Maker workflow.
- Write concise, enforceable rules only.
- Do not change workflow semantics without Analyzer justification and Reviewer-style approval.

## Required Workflow

1. Editor reads the relevant repo files and writes a plan.
2. Reviewer reviews the Editor plan and explicitly approves or rejects it.
3. If rejected, Editor revises the plan and resubmits it.
4. Repeat until the Editor plan is approved.
5. Editor implements and writes an Implementation Summary.
6. Tester writes a validation plan.
7. Reviewer reviews the Tester plan and explicitly approves or rejects it.
8. If rejected, Tester revises the plan and resubmits it.
9. Repeat until the Tester plan is approved.
10. Tester performs validation and writes a Test Summary.
11. Rule Maker runs after completed Code Revise cycles when repeated process failures, ambiguous contracts, or missing validation requirements were exposed.
12. Any edit to `AGENTS.md`, `HOOKS.md`, or files in `docs/` must call the Rule Maker team.

## Required Output Sections

Use these headings when applicable:
- `Editor Plan`
- `Reviewer Decision`
- `Revised Editor Plan`
- `Implementation Summary`
- `Tester Plan`
- `Reviewer Decision`
- `Revised Tester Plan`
- `Test Summary`

## Repo-Specific Enforcement Rules

- Prefer building the smallest useful function first, then compose small units into larger behavior.
- Any change to `BaseCommunication` requires explicit compatibility handling for both `MmapCommunication` and `ShmCommunication`.
- Any change to image, position, event, or action buffers must state dtype, shape, and memory-order impact.
- Any plan touching synchronization must address the current `write_flag(...)` contract drift documented in `docs/interfaces.md`.
- Any plan touching event behavior must define the authoritative event-code table and where dispatch occurs in `VirMEnEnv`.
- Any nontrivial change must include offline validation for the environment and whichever backend surfaces were touched.
- Live MATLAB/VirMEn validation is required when a change cannot be proven offline.

## Completion Rule

Work is not complete until:
- the Editor plan was approved
- the implementation was summarized
- the Tester plan was approved
- the validation was summarized
