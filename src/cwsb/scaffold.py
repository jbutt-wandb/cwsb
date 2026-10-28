"""`cwsb init` — scaffold a sandbox-ready marimo project."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .project import uv_env

# The torch family MUST come from the cu128 index: PyPI's default builds are
# CUDA-13 and ABI-incompatible with this stack (torchcodec from PyPI cost us an
# hour once). Pins for packages you haven't added yet are harmless — they only
# apply when the package enters the dependency tree.
_SOURCES_BLOCK = """\
[tool.uv.sources]
# Any torch-family lib (torchaudio, xformers, torchao, ...) you `cwsb uv add`
# later should be pinned here FIRST, or uv will resolve PyPI's CUDA-13 build.
# NB: cu128-index versions can lag PyPI numbering (e.g. torchcodec is 0.11.x
# there, matched to torch 2.11) — loosen version floors accordingly.
torch = [{ index = "pytorch-cu128" }]
torchvision = [{ index = "pytorch-cu128" }]
torchaudio = [{ index = "pytorch-cu128" }]
torchcodec = [{ index = "pytorch-cu128" }]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
"""

_BASE_DEPS = ['"marimo>=0.13"', '"numpy"', '"torch>=2.8"']
_VISION_DEPS = [
    '"transformers>=5.21"',
    '"accelerate"',
    '"torchvision>=0.26"',
    '"pillow"',
    '"av"',
    '"torchcodec>=0.11"',
    '"nvidia-npp-cu12"',
]
_CPU_DEPS = ['"marimo>=0.13"', '"numpy"', '"polars"']

_GITIGNORE = """\
.env
.sandbox-state.json
__pycache__/
.venv/
"""

_HELLO_GPU = '''\
import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Hello GPU 👋

        This notebook runs **inside a CoreWeave GPU sandbox** — your browser is
        local, the kernel is not.

        - **Libraries**: `cwsb uv add <pkg>` from your terminal (git-tracked, synced
          in seconds). marimo's install banner is ephemeral until promoted.
        - **Files**: `cwsb files add <path>` → `/work/data/`; notebooks sync with
          `cwsb pull` / `cwsb push` (auto-pull on `cwsb down`).
        - **Done**: `cwsb down` — snapshots everything, stops billing.
        """
    )
    return


@app.cell
def _():
    import torch

    cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda else "CPU only — no CUDA?!"
    device
    return (cuda, torch)


@app.cell
def _(cuda, torch):
    if cuda:
        x = torch.randn(4096, 4096, device="cuda")
        result = (x @ x.T).mean().item()
    else:
        result = None
    result
    return
'''

_HELLO_CPU = '''\
import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Hello sandbox 👋

        This notebook runs inside a CoreWeave sandbox (CPU). Use `cwsb uv add` for
        libraries, `cwsb files add` for data, `cwsb down` when done.
        """
    )
    return


@app.cell
def _():
    import numpy as np

    np.arange(10).sum()
    return
'''


def _pyproject(name: str, deps: list[str], gpu: bool, include_sources: bool) -> str:
    dep_lines = "\n".join(f"    {d}," for d in deps)
    gpu_line = "" if gpu else "gpu = false\n"
    sources = f"\n{_SOURCES_BLOCK}" if include_sources else ""
    return f"""\
[project]
name = "{name}"
version = "0.1.0"
description = "marimo notebooks on CoreWeave Sandboxes (managed by `cwsb`)"
requires-python = ">=3.12"
dependencies = [
{dep_lines}
]

# This project is never installed locally — it defines the env that `cwsb`
# syncs into the sandbox at /work/venv (x86_64 Linux).
[tool.uv]
package = false
environments = ["sys_platform == 'linux'"]

[tool.cwsb]
{gpu_line}# tag = "..."            # sandbox tag (default: cwsb-marimo-<dir>-<user>)
# lifetime = "8h"        # hard cap per session
# apt_packages = []      # extra apt installs at bootstrap (ffmpeg is built in)
{sources}"""


def cmd_init(args) -> None:
    root = Path.cwd()
    py = root / "pyproject.toml"
    if py.exists():
        sys.exit(
            "pyproject.toml already exists — not scaffolding over it.\n"
            "To adopt this project for cwsb instead:\n"
            "  1. add a `[tool.cwsb]` section (optional; defaults are fine)\n"
            "  2. if you use torch: pin the torch family to the pytorch-cu128\n"
            "     index in [tool.uv.sources] (run `cwsb init` in an empty dir to\n"
            "     see the exact block)\n"
            "  3. `uv lock`, then `cwsb marimo edit`"
        )

    cpu = bool(args.cpu)
    deps = list(_CPU_DEPS) if cpu else list(_BASE_DEPS)
    if args.vision:
        if cpu:
            sys.exit("--vision needs a GPU preset; drop --cpu.")
        deps += _VISION_DEPS

    py.write_text(_pyproject(root.name.replace("_", "-"), deps, gpu=not cpu, include_sources=not cpu))
    gi = root / ".gitignore"
    if gi.exists():
        existing = gi.read_text()
        missing = [l for l in _GITIGNORE.splitlines() if l and l not in existing]
        if missing:
            gi.write_text(existing.rstrip("\n") + "\n" + "\n".join(missing) + "\n")
    else:
        gi.write_text(_GITIGNORE)
    nb = root / "notebooks"
    nb.mkdir(exist_ok=True)
    hello = nb / "hello.py"
    if not hello.exists():
        hello.write_text(_HELLO_CPU if cpu else _HELLO_GPU)

    print(f"scaffolded {py.name}, .gitignore, notebooks/hello.py ({'CPU' if cpu else 'GPU'}"
          f"{' + vision' if args.vision else ''} preset)")
    print("locking (resolves linux wheels — nothing installs locally)...")
    r = subprocess.run(["uv", "lock"], cwd=root, env=uv_env())
    if r.returncode != 0:
        sys.exit("uv lock failed — fix pyproject.toml and run `uv lock` again.")
    print("done. Next: `cwsb marimo edit`  (first boot downloads the stack; later boots restore a snapshot)")
