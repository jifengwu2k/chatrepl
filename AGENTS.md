---
title: Python Coding Guidelines
date: 2025-08-29
categories:
  - "Programming"
tags:
  - "reference"
  - "python"
  - "coding-guidelines"
---

## General Principles

- **One file = one module = one purpose**: Each file must be explicitly importable as a module and serve exactly one main purpose.
- Do **not** default to generic Python ecosystem "best practices" when they conflict with the user's stated project shape or these repository rules.
- Prefer the simplest implementation that directly matches the user's wording over reusable abstraction layers added speculatively.

## Python Version Policy

- **All code must run and be tested on both Python 2 and Python 3.**
- **Version-conditional logic**: Only use `sys.version_info` for version checks. Do *not* use `except ImportError`, `sys.version`, or similar.

### String Types

- Encoding/decoding UTF-8, URI, stdin/out, file names etc. can use our [`textcompat`](https://github.com/jifengwu2k/textcompat) package.
- Only use the legacy `%` formatting syntax. No `.format()` or f-strings.

### Path Handling

- **Do not use `pathlib`.** Handle paths of all kinds as plain strings.
- **Clearly separate user-facing paths from internal paths.**
  - **User-facing paths** are arbitrary strings accepted from external inputs.
  - **Internal paths** are always-valid, canonicalized, domain-specific strings used by internal machinery.
- **API boundary rule**:
  - User-facing API functions should accept user-facing paths.
  - Internal machinery should store and use internal paths.
  - Convert user-facing paths to internal paths at the API boundary, then use internal paths as-is. Do not repeatedly canonicalize, normalize, decode, encode, or reparse internal paths.
- **Path syntax and semantics are context-dependent.** There is no single universal path model for all code. The meaning of a path string depends on the surrounding system and must be defined per use case.
- **Use [`fspathverbs`](https://pypi.org/project/fspathverbs/) to convert user-facing paths into internal paths.**
  - Basic conversion pipeline: **user-facing path -> verbs -> internal path**
    - Handle all path verbs explicitly. 
  - The exact compilation and interpretation steps vary by context.
- **Handle links transparently.**

#### User-facing vs internal paths

- A user-facing path may be:
  - a local filesystem path supplied by a user,
  - a URL/HTTP path,
  - an archive member path,
  - or any other external path-like string defined by the application.
- An internal path should be a canonicalized string representation suitable for stable internal use.
- Do not mix the two concepts in naming, storage, or APIs.

#### Web server example

In an `http.server`-style web server, there are at least two distinct user-facing path domains:

1. **Filesystem path provided when starting the server**
   - Example user-facing path: `../../root/of/web/server`
   - Compile to verbs:
     ```python
     [
         Current(),
         Parent(),
         Parent(),
         Child(child='root'),
         Child(child='of'),
         Child(child='web'),
         Child(child='server')
     ]
     ```
   - Interpret those verbs relative to `os.getcwd()` such as `/home/user`
   - Resulting internal path: `/root/of/web/server`

2. **HTTP request path**
   - Example user-facing path: `/path/./to/../other-resource/../resource`
   - Compile to verbs:
     ```python
     [
         Root(root='/'),
         Child(child='path'),
         Current(),
         Child(child='to'),
         Parent(),
         Child(child='other-resource'),
         Parent(),
         Child(child='resource')
     ]
     ```
   - Interpret those verbs relative to the internal web server root `/root/of/web/server`
   - Resulting internal path: `/root/of/web/server/path/to/resource`

### Input Handling

- For interactive input, use our [`unicode-raw-input`](https://github.com/jifengwu2k/unicode-raw-input) package.
- Use `open(...)` for binary files; `codecs.open(...)` for Unicode text files.

### Print

- Use `from __future__ import print_function` if using `print()`.

### Typing

- All APIs fully typed with type **comments only** (`# type: ...`)
    - No inline type annotations, no `AnnAssign`.
- Use only typing features as in Python 3.5 / PEP 484.
- Absolutely no dependent typing. The return type of a function must not vary with different parameter types and/or values.
- No use of `@overload` permitted.

### Classes

- No `attrs`, `dataclasses`, or `namedtuple`.
- All classes must have:
    - Declared `__slots__`
    - Explicit `object` base class 
    - Use `six.with_metaclass(meta, *bases)` for metaclasses
    - Mutable: `__init__`; Immutable: `__new__`

### Enums

- Only manual/explicit values; do not use `auto()` even with `enum34`.

### Restricted Language Features

- Never use: `async`, `await`, `yield from`, walrus (`:=`), structural pattern matching (`match/case`).

### Exceptions

- Exceptions should propagate by default.
- Do **not** catch exceptions unless there is a clear, documented reason to transform them, add essential context, or implement a specific recovery strategy.
- Never silence exceptions or convert them into fallback behavior by default.

## Platform Policy

- **All code must run and be tested on both NT and POSIX.**
- **Determine platform with [`posix-or-nt`](https://github.com/jifengwu2k/posix-or-nt)** (returns `'nt'` or `'posix'`).
- **Never use** the `subprocess`, or `platform` modules. Overly complicated and convoluted codebases, and large parts are not available or behave differently on NT, Android, and iOS, etc.
    - *Process launching*: Use our [`ctypes-unicode-proclaunch`](https://github.com/jifengwu2k/ctypes-unicode-proclaunch).
    - *Env vars*: Use our [`read-unicode-environment-variables-dictionary`](https://github.com/jifengwu2k/read-unicode-environment-variables-dictionary).
- Use [`sys.platform`](https://docs.python.org/3/library/sys.html#sys.platform) if you absolutely need an OS name.

### System API Access

- The `multiprocessing` module is **not allowed**.
- Direct system calls via `ctypes` are strictly limited as follows:
  - On NT:
    - Only functions in **MSVCRT** (Microsoft C Runtime) and the standard **Win32 API** may be called.
  - On POSIX:
    - Only standardized POSIX functions (from libc or standard headers) may be called.
  - Should be factored as a standalone, well-tested PyPI package similar to our other infra tools.
- If code needs capabilities beyond these, use **PyQt/PySide** and the Qt API as your system abstraction layer.
  - Examples: advanced file I/O, process control, IPC, networking, device enumeration, drag-and-drop, clipboard, graphics, etc.
  - Use our [`detect-qt-binding`](https://github.com/jifengwu2k/detect-qt-binding) to automatically detect which Qt binding is available in your environment.

## File and Folder Structure

- Import all files (modules) via absolute import. No relative import, no `sys.path` manipulation.
- All files (modules) must only have public functions and classes - no private/internal APIs.
- All directories must include an explicit `__init__.py` within them.
- **Preserve the requested application shape**: if the user describes the project as a single application / single-file program, keep it as exactly one project `.py` file, one module, and one entrypoint unless the user explicitly asks for further splitting.
- Do **not** proactively refactor a single-file application into a package with multiple local modules just for organization.

### Utility Code: No "utils.py" - Publish Generalized Tools

- No local "utils.py" files: Do not keep grab-bag or miscellaneous utility functions/classes in a project-private `utils.py` file.
- **General-purpose tools must be published as standalone packages to PyPI**, each focusing on one well-defined function, class, or concept.
    - Move helpers that would otherwise go into "utils.py" into their own properly documented, versioned packages.
    - These utility packages should be:
        - Named after their actual purpose;
        - Well-tested and actively maintained;
        - Equipped with a README, proper docstrings, testable usage examples, semantic versioning, and a clear license.
- All projects share utilities via explicit dependencies rather than duplicating or copying helpers.
    - When a utility is improved or a bug is fixed, updating the package ensures all dependent projects benefit automatically - "write once, run everywhere."
    - This approach avoids hidden technical debt and promotes code quality, documentation, reuse, and maintainability across your entire codebase.
    - It also contributes to the wider Python ecosystem.

## Regular Expressions

- Regular expressions **only for simple parsing**.
- Use **Unix-style/basic** regex features:
  - `.`: any single character
  - `[ ]`: character set/class
  - `[^ ]`: negated class
  - `^, $`: line start/end
  - `( )`: grouping/subexpression
  - `*`: zero or more
- For **complicated input**:
  - Use a context-free grammar parser (e.g. [Lark](https://github.com/lark-parser/lark)), hand-written parser, or an LLM.

## Argument Parsing

- Use `argparse`.
- Use ambiguous `str` for all argument values (default API behavior).
- All arguments should have a `help=...` string.
- By default, parse arguments directly where the program starts instead of adding speculative wrappers such as `run(argv=None)` or custom argument namespace classes, unless the user explicitly asks for such an abstraction.

### Flag Arguments

- **Presence only:** `action='store_true'` -> `bool`
- **Counted:** `action='count'` -> `int`
- **Single value:** `type=str`, `required=True/False`, explicit `default=...`
- **Multiple occurrences:** `action='append'` -> `Optional[List[str]]`

### Positional Arguments

- **Exactly one:** simple positional
- **0/1 (optional):** `nargs='?'`, must be last positional argument
- **0 or more/1 or more:** `nargs='*'`/`nargs='+'`, with plural argument name and singular `metavar`, must be last positional argument

## Testing & Documentation

- Tests must:
  - Run successfully on both Python 2 and 3, POSIX and NT.
  - Be suitable for inclusion in `README.md` under "Usage" or "Quickstart".
    - Simultaneously serve as usage documentation (idiomatic examples).
    - Self-contained and runnable as a script or documentation block.
- Do **not** assume the word "test" means a dedicated unit-test file or a special self-test CLI flag unless the user explicitly asks for that. Prefer README-based smoke checks / walkthroughs that a user can quickly follow to verify behavior and understand capabilities.

## Python Packaging & Distribution

### File Layout

- All files start with a copyright and license block:
  - Boilerplate: `# Copyright (c) 2026 Jifeng Wu\n# Licensed under the <license> License. See LICENSE file in the project root for full license information.`
    - simple infrastructure: MIT/BSD
    - complex infra: Apache-2.0
    - applications: AGPL-3.0

### Required Files & Metadata

- `README.md` with standard boilerplate (see below)
- `LICENSE`
- `requirements.txt` (see example)
- `pyproject.toml` (see template)
- `setup.cfg` (see template)

### `README.md` Boilerplate

```
<Project Description>

## Installation

...

## Usage

...

## Contributing

Contributions are welcome! Please submit pull requests or open issues on the GitHub repository.

## License

This project is licensed under the [<license> License](LICENSE).
```

- All projects are intended to be published to PyPI.
- Therefore, installation instructions should treat the canonical install command as `pip install <package-name>`.

### Example `requirements.txt`

```
enum34; python_version < '3.4'
pyreadline
six
typing; python_version < '3.5'
```

### `pyproject.toml` Template

```
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "<project-name>"
version = "<version>"
description = "<Project Description>"
readme = "README.md"
requires-python = ">=2"
license = "<license>"
authors = [
  { name="Jifeng Wu", email="jifengwu2k@gmail.com" }
]
classifiers = [
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
]
dependencies = [
    "<requirements.txt line 1>",
    "<requirements.txt line 2>",
    "<requirements.txt line 3>"
]

[project.urls]
"Homepage" = "https://github.com/jifengwu2k/<project-name>"
"Bug Tracker" = "https://github.com/jifengwu2k/<project-name>/issues"
```

Replace `<project-name>`, `<version>`, `<license>`, and requirements as appropriate.

### `setup.cfg` Template

```
[bdist_wheel]
universal = 1
```
