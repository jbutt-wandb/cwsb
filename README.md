# cwsb — CoreWeave Sandbox marimo notebooks

`cwsb marimo edit` opens a marimo notebook in your local browser whose kernel
runs on a GPU inside a [CoreWeave Sandbox](https://docs.coreweave.com/products/sandboxes).
The sandbox is an implementation detail: `cwsb` creates, reuses, heals,
snapshots, and deletes it for you. Your project directory stays the source of
truth — notebooks and the dependency lockfile are plain git-tracked files.

This document is written to be sufficient on its own: a human or an AI agent
should be able to install, operate, and troubleshoot `cwsb` from here without
other context.

---

## Requirements

| Requirement | Details |
|---|---|
| `uv` | <https://docs.astral.sh/uv/> — the only local dependency; `cwsb` runs no compute locally |
| `CWSANDBOX_API_KEY` | Token with the **Sandbox User** IAM role from <https://console.coreweave.com/tokens>. Put it in the project's `.env` (auto-loaded, keep it gitignored) or export it |
| GPU entitlement | GPU sandboxes are private preview, entitled **per CoreWeave organization**. Unentitled orgs fail at create in <2 s with a clear error (nothing billed) — contact your account team / forge-support@coreweave.com |
| OS | macOS/Linux tested; sandbox side is always Linux x86_64 |

## Install

```sh
uv tool install git+https://github.com/jbutt-wandb/cwsb
```

The repo is **private** for now, so the installer needs GitHub credentials with
access. If your git credential helper is already authenticated, the line above
just works. Otherwise use a personal access token:

```sh
uv tool install git+https://<YOUR_GITHUB_TOKEN>@github.com/jbutt-wandb/cwsb
```

⚠️ Never paste a real token into shared docs, shell histories you publish, or
committed files. The clean long-term path is making the repo public or
org-visible, after which the first form works for everyone.

Upgrade later with `uv tool upgrade cwsb`; remove with `uv tool uninstall cwsb`.

## Quickstart

```sh
mkdir my-notebooks && cd my-notebooks
echo 'CWSANDBOX_API_KEY=...' > .env     # or export it

cwsb init                     # scaffold: pyproject (cu128 torch pins), notebooks/hello.py,
                              # .gitignore, uv.lock. Add --vision for the VL/video stack.

cwsb marimo edit              # creates a GPU sandbox (~1-2 min first boot), starts marimo,
                              # prints the URL and opens your browser. Re-running reuses it.

cwsb uv add polars            # lock locally + sync the sandbox venv (~seconds);
                              # re-run the notebook cell and `import polars` works.

cwsb files add data.csv       # upload → /work/data/data.csv  (read it by absolute path)

cwsb down                     # pull notebooks home, snapshot /work, DELETE the sandbox.
                              # Billing stops. Next `edit` restores everything in minutes.
```

## Command reference

Every command is per-project: `cwsb` anchors to the nearest `pyproject.toml`
at or above the current directory (like uv). Side effects are stated explicitly.

| Command | What it does | Side effects |
|---|---|---|
| `cwsb init [--vision] [--cpu]` | Scaffold a sandbox-ready project in the cwd and run `uv lock`. `--vision` adds transformers/torchvision/torchcodec/av; `--cpu` makes a torch-free, cheaper preset | Writes `pyproject.toml`, `.gitignore`, `notebooks/hello.py`, `uv.lock`. Refuses if `pyproject.toml` exists (prints adoption steps instead) |
| `cwsb marimo edit [nb.py] [--gpu\|--cpu] [--lifetime 8h] [--fresh] [--no-browser]` | Ensure a sandbox exists (create, restore snapshot, or reuse the running one), push `notebooks/*.py`, ensure marimo is serving, open the browser (deep-linked if `nb.py` given; file is created if missing) | **Creates a billed sandbox** if none is running (billing runs until `down`/`cleanup`/lifetime cap). `--fresh` ignores the saved snapshot. Push overwrites remote notebook copies with local ones |
| `cwsb marimo new [nb.py]` | Create a new local notebook and open it | Same as `edit` |
| `cwsb marimo run <nb.py> [--public]` | Serve a notebook as a read-only app. Token-protected by default; `--public` disables auth on an internet-reachable URL (prints a loud warning) | Restarts the marimo server (edit and run share one port); `edit` switches back |
| `cwsb marimo restart` | Restart the marimo server: fresh kernels, same URL/token. Use after env changes or ghost errors | Running cells die; browser tabs prompt to reconnect |
| `cwsb uv add\|remove <pkg…>` | Update `pyproject.toml` + `uv.lock` locally (nothing installs on your machine — GPU-only wheels are fine) and sync the sandbox venv | Edits pyproject/lock; packages importable in running notebooks in seconds. Works with no sandbox (syncs on next `edit`) |
| `cwsb uv sync` | Re-lock if `pyproject.toml` was hand-edited, then sync the sandbox | Same |
| `cwsb files add <path…> [--to /work/data]` | Upload files (≤500 MB each; creates the remote dir) | Overwrites same-named remote files. For bigger files download sandbox-side: `cwsb exec -- curl -L -o /work/data/f <url>` |
| `cwsb files get <remote…> [--to .]` | Download files from the sandbox | Overwrites same-named local files |
| `cwsb push` / `cwsb pull` | Sync `notebooks/*.py` local → sandbox / sandbox → local (top-level `.py` only) | Push favors local copies; pull skips unchanged files |
| `cwsb status` | Project, sandbox state, mode, snapshot id, URL + access token | None (read-only) |
| `cwsb down` | Pull notebooks → snapshot `/work` (prints size + ETA; retries flaky snapshot RPCs 5×) → delete the sandbox → GC the previous snapshot | **Stops billing.** The snapshot (paid storage) is the restore point for the next `edit` |
| `cwsb cleanup` | Delete ALL sandboxes carrying this project's tag — no pull, no snapshot | **Destroys un-pulled/un-snapshotted work.** Recovery tool for wedged/failed creates and orphans |
| `cwsb exec -- <cmd>` | Run a shell command on the sandbox with the full env (venv on PATH, HF_HOME, LD_LIBRARY_PATH) | Whatever the command does |
| `--project <dir>` (global) | Anchor to an explicit project directory instead of walking up from cwd | — |

## Configuration — `[tool.cwsb]` in pyproject.toml

| Key | Default | Meaning |
|---|---|---|
| `gpu` | `true` | Create GPU sandboxes (1× GPU — currently RTX PRO 6000 96 GB or H100 80 GB, the pool decides). `false` = CPU-only (much cheaper, ~5 s boot) |
| `lifetime` | `"8h"` | Hard lifetime cap per sandbox (`90m`, `2d`, max 30 d). Fixed at creation, **cannot be extended** |
| `tag` | `cwsb-marimo-<dir>-<user>` | Sandbox tag; `cleanup` reaps only this tag. Set explicitly to adopt an existing sandbox/snapshot lineage |
| `port` | `2718` | marimo port behind the HTTPS endpoint |
| `mount_size` | `"100Gi"` | Size of the `/work` snapshot mount |
| `image` | `python:3.12-bookworm` | Container image (needs git/gcc/curl for ML stacks) |
| `apt_packages` | `[]` | Extra apt installs at bootstrap (ffmpeg is always installed) |

## Concepts — read before relying on this in anger

- **Project scoping.** `cwsb` walks up from your cwd to the nearest
  `pyproject.toml`, exactly like uv. Running it inside an unrelated Python repo
  anchors there — use `--project` or cd to the right root. State lives in
  `<root>/.sandbox-state.json` (gitignored): sandbox id, snapshot id, session token.
- **Persistence = `/work` + snapshots, nothing else.** Only the `/work` mount
  survives: the venv, uv cache, `HF_HOME` (model weights), `/work/notebooks`,
  `/work/data`. Anything outside `/work` — including apt packages — dies with
  the sandbox (bootstrap reinstalls apt deps each boot). `cwsb down` takes the
  snapshot; **lifetime-cap expiry does NOT** — it deletes the sandbox with no
  pull and no snapshot, losing everything since your last `down`/`pull`.
- **URL and token rotate every session.** Each created sandbox gets a fresh
  internet-reachable HTTPS URL; marimo's access token is the only auth. Treat
  the tokenized URL as a password. `cwsb status` reprints it.
- **Billing** is per-millisecond from sandbox-ready until deletion, plus paid
  storage for the snapshot between sessions. `cwsb down` when stepping away;
  the lifetime cap is the backstop, not the habit.
- **The cu128 index rule (torch family).** PyPI's default torch-family builds
  are CUDA-13 and ABI-incompatible with this stack. `torch`, `torchvision`,
  `torchaudio`, `torchcodec` must resolve from the `pytorch-cu128` index —
  `cwsb init` pins all four in `[tool.uv.sources]`; add any other torch-family
  lib (xformers, torchao, …) to that list **before** `cwsb uv add`-ing it.
  Watch the version lag: the cu128 index numbers differently (e.g. torchcodec
  is `0.11.x+cu128` there, matched to torch 2.11, while PyPI shows 0.17) —
  loosen version floors accordingly.
- **marimo's in-editor package installer is ephemeral.** It bypasses the
  lockfile; the next sync reverts it, and an unpinned banner install can drag
  torch onto an incompatible CUDA build. `cwsb uv add` is the only install
  that sticks.

## Troubleshooting (every entry field-tested)

| Symptom | Cause | Fix |
|---|---|---|
| "no capacity … retrying" / `RESOURCE_EXHAUSTED` on create | GPU pool at capacity | `cwsb` auto-retries ~20 min with backoff; if it gives up, try later or forge-support@coreweave.com |
| Sandbox stuck CREATING > 15 min, or status `failed` | Platform-side wedged/failed create (unbilled) | `cwsb cleanup`, then `cwsb marimo edit` — fresh creates usually land on a healthy runner |
| Notebook shows errors that don't match reality ("Reconnected" banner; traceback cites code not on disk) | marimo resurrects cached session outputs; kernels also go stale when the venv changes under them | `cwsb marimo restart` + hard-refresh the page; stubborn ghosts: `cwsb exec -- rm -rf /work/notebooks/__marimo__/session` first |
| `ImportError: libcudnn.so.9` or other missing nvidia libs after mixed installs | cu12/cu13 nvidia wheels share install dirs and clobber each other's files on uninstall | `cwsb exec -- "cd /work/project && uv sync --locked --reinstall"` (seconds, from warm cache) |
| torchcodec `OSError: Could not load this library` | Needs the cu128 build + apt ffmpeg + NVIDIA lib path, all three | All baked in (`init` pins, bootstrap installs ffmpeg, env block exports the lib path). From a bare shell, run via `cwsb exec` so the env applies |
| Snapshot phase of `down` takes minutes | Snapshot time scales with `/work` payload (torch stack + model weights add up) | Normal — `down` prints the size and an ETA (~1 min at 7 GB, ~7 min at 24 GB) |
| `uv sync` log says `Removed virtual environment … Creating …` on a restored sandbox | Image Python micro-version drift invalidates the venv | Normal — reinstalls everything from the `/work` cache in ~2 s |
| gRPC `UsageError: Channel is closed.` traceback after a command | SDK async teardown noise | Filtered by `cwsb`; ignore any variant that leaks |
| uv warning about `VIRTUAL_ENV` not matching | An active venv vs. the project | Harmless; `cwsb` strips it for inner uv calls |

## Adopting an existing project

`cwsb init` refuses to overwrite a `pyproject.toml`. To adopt: add a
`[tool.cwsb]` section (optional — defaults are fine; set `tag` to keep an
existing sandbox/snapshot lineage), pin any torch-family deps to the
`pytorch-cu128` index in `[tool.uv.sources]` (run `cwsb init` in an empty dir
to copy the exact block), run `uv lock`, then `cwsb marimo edit`.
