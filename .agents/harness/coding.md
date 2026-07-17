# Python Coding Guidelines

## Linting & Formatting

- **Linter:** Ruff with rule sets: `E, F, W, I, UP, B, SIM, RUF, N, D`
- **Docstring convention:** Google style (enforced by `D` / pydocstyle rules)
- **Line length:** 120 characters
- **Type checker:** Pyright in basic mode — type hints required on all function signatures
- **Type Ignores:** Type warnings or errors (`# type: ignore`) should only be ignored if there is a strong, justifiable reason for it (e.g., known limitations in library stubs like pandas).
- **Excluded from linting:** `prior.py` (upstream nanoTabPFN code, do not modify)

## Imports

- Sorted by isort (Ruff `I` rules)
- No unused imports (`F401`)
- No star imports (`F403`)
- Group order: stdlib → third-party → local

## Naming Conventions

- `snake_case` for functions, methods, and variables
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for module-level constants
- Follow PEP 8 naming (`N` rules)

## Code Style

- **f-strings** preferred over `.format()` or `%` formatting (`UP032`, `RUF010`)
- **`pathlib.Path`** over `os.path` where practical (`PTH` if enabled, general preference)
- No bare `except:` — always catch specific exceptions (`B001`, `E722`)
- No mutable default arguments (`B006`)
- No shadowing of Python builtins (`A001`, `A002`)
- Use modern Python syntax: `dict | None` over `Optional[Dict]` (`UP` rules)

## Docstrings

- All public modules, classes, and functions must have Google-style docstrings
- Include `Args:`, `Returns:`, and `Raises:` sections where applicable
- One-line docstrings for trivial functions are acceptable

## General

- Keep functions focused and under ~50 lines where possible
- Prefer early returns over deep nesting (`SIM` rules)
- Use `contextlib` utilities where appropriate
- All new code must pass `ruff check` and `pyright` before committing
