# Specification Files

This directory contains structured specification documents for the TabPFN Activation Function Ablation Study project. Specs drive the development process: they start as **aspirational goals** defining desired behavior, guide the implementation, and ultimately serve as **living snapshots** of what the committed code currently does. Code and specs must always remain in strict sync upon commit.

## Spec Format

Every spec file uses YAML frontmatter followed by Markdown content:

```markdown
---
id: PREFIX-NNN
title: "Title"
status: implemented
module: filename.py
last_synced: YYYY-MM-DD
---

# Title

## User Story

As a [role], I want [goal] so that [benefit].

## Acceptance Criteria

- [ ] AC-1: Description
- [ ] AC-2: Description
...

## Notes

Optional implementation notes.
```

## Frontmatter Fields

| Field         | Description                                                      |
|---------------|------------------------------------------------------------------|
| `id`          | Unique identifier. Prefix indicates category (`CORE`, `EVAL`, `EXP`). |
| `title`       | Human-readable title for the spec.                               |
| `status`      | One of: `new`, `implemented`, `edited`.                          |
| `module`      | Primary source file(s) the spec covers.                          |
| `last_synced` | Date when the spec was last verified against the source code.    |

## Status Values

| Status        | Meaning                                                                    |
|---------------|----------------------------------------------------------------------------|
| `new`         | Spec written but not yet implemented.                                      |
| `implemented` | Spec documents functionality that exists and has been verified in the code. |
| `edited`      | Code has changed since the spec was last synced; spec needs review.        |

## Acceptance Criteria Format

- Each criterion uses the checkbox format: `- [ ] AC-N: Description` (for unmet) or `- [x] AC-N: Description` (for met).
- During active development (`status: new` or `status: edited`), check the boxes (`[x]`) for criteria that are currently met by the code, and leave them unchecked (`[ ]`) for those that are not.
- **CRITICAL:** Acceptance criteria can **only** be checked off if they have been explicitly verified against the codebase (e.g., via automated tests or manual code inspection).
- When `status: implemented`, **all** checkboxes must be checked (`[x]`).
- Criteria must be **testable** — each one should be verifiable by an automated test or manual inspection.
- Criteria must be **atomic** — one behavior per criterion.
- Criteria are initially derived from **desired behavior** for new features, and may evolve during implementation. Once committed, they must strictly match the **actual source code**.

## Directory Structure

```
specs/
├── README.md                          # This file
├── core/
│   ├── model_architecture.md          # CORE-001
│   ├── training_pipeline.md           # CORE-002
│   └── seed_determinism.md            # CORE-003
├── evaluation/
│   ├── local_eval.md                  # EVAL-001
│   ├── tabarena_eval.md               # EVAL-002
│   ├── plotting.md                    # EVAL-003
│   └── variant_comparison.md          # EVAL-004
└── experiment/
    └── activation_ablation.md         # EXP-001
```

## Sync Workflow

1. **Read the code** — understand the current implementation.
2. **Write/update the spec** — derive acceptance criteria from actual behavior.
3. **Set `last_synced`** — to today's date after verifying against the code.
4. **When code changes** — set status to `edited` until the spec is re-verified.
5. **When re-verified** — update acceptance criteria if needed, reset status to `implemented`, update `last_synced`.
