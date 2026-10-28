# sb — marimo-native GPU notebooks on CoreWeave Sandboxes

`sb marimo edit` and you're editing a notebook in your browser whose kernel runs
on a GPU in a CoreWeave serverless sandbox. The sandbox is an implementation
detail: `sb` creates, reuses, heals, snapshots, and reaps it for you.

```sh
uv tool install git+<repo-url>        # once published; for now:
uv tool install /Users/jbutt/Developer/sb

mkdir my-notebooks && cd my-notebooks
sb init                                # scaffold (add --vision for the VL stack)
sb marimo edit                         # ~1 min first boot → browser opens on a GPU
sb uv add "transformers>=5.21" polars  # libraries: git-tracked, live in seconds
sb down                                # snapshot everything, stop billing
```

## One-time setup

- Put `CWSANDBOX_API_KEY` in `<project>/.env` (gitignored; auto-loaded). Create a
  token with the **Sandbox User** role at <https://console.coreweave.com/tokens>.
  GPU sandboxes are entitlement-gated per org — ask your CoreWeave contact or
  forge-support@coreweave.com if creates are rejected.
- Everything is **per-project** (nearest `pyproject.toml`, like uv): its own
  environment (`pyproject.toml` + `uv.lock`), notebooks (`notebooks/*.py`),
  sandbox lineage (`.sandbox-state.json`, gitignored), and sandbox tag — so
  teammates and projects never collide.

## Commands

| Command | What it does |
|---|---|
| `sb init [--vision] [--cpu]` | Scaffold a sandbox-ready project in the cwd (pyproject with cu128 torch pins, notebooks/, .gitignore) + `uv lock` |
| `sb marimo edit [nb.py] [--gpu\|--cpu] [--lifetime 8h] [--fresh] [--no-browser]` | Ensure sandbox (create/reuse/heal) + marimo editor, open the browser. Creates `nb.py` if missing |
| `sb marimo new [nb.py]` | New notebook, opened in the editor |
| `sb marimo run <nb.py> [--public]` | Serve a notebook as a read-only app (token-protected; `--public` disables auth, loudly). Shares the port with edit — switching modes restarts the server |
| `sb marimo restart` | Restart the marimo server: fresh kernels, same URL/token |
| `sb uv add\|remove <pkg…>` | Update `pyproject.toml`+`uv.lock` locally (nothing installs on your machine) and sync the sandbox venv — importable in running notebooks in seconds |
| `sb uv sync` | Re-lock if `pyproject.toml` changed by hand + sync the sandbox |
| `sb files add <path…> [--to /work/data]` | Upload files (≤500 MB each; bigger → `sb exec -- curl -L -o /work/data/f <url>` at datacenter speed) |
| `sb files get <remote…> [--to .]` | Download files from the sandbox |
| `sb push` / `sb pull` | Sync `notebooks/*.py` ↔ sandbox (`down` auto-pulls) |
| `sb status` | State, mode, snapshot, URL + access token |
| `sb down` | Pull notebooks → snapshot `/work` → delete sandbox (billing stops; old snapshot GC'd) |
| `sb cleanup` | Delete ALL of this project's sandboxes, no snapshot (orphan/failed-create recovery) |
| `sb exec -- <cmd>` | Shell on the sandbox with the venv/HF/LD_LIBRARY_PATH env |

Global flag: `--project <dir>` to target a project from elsewhere.

## Configuration — `[tool.sb]` in pyproject.toml

| Key | Default | Meaning |
|---|---|---|
| `gpu` | `true` | GPU sandbox (1× pool GPU: RTX PRO 6000 96 GB or H100 80 GB) vs CPU (cpu4/8Gi, ~5 s boot, much cheaper) |
| `lifetime` | `"8h"` | Hard cap per sandbox (e.g. `90m`, `2d`, max 30 d). Cannot be extended — `sb down` before it hits |
| `tag` | `sb-marimo-<dir>-<user>` | Sandbox tag; `cleanup` reaps only this tag. Set it to adopt an existing lineage |
| `port` | `2718` | marimo port behind the HTTPS endpoint |
| `mount_size` | `"100Gi"` | `/work` snapshot mount size |
| `apt_packages` | `[]` | Extra apt installs at bootstrap (ffmpeg is built in; apt lives outside snapshots, reinstalls each boot) |
| `image` | `python:3.12-bookworm` | Container image (needs git/gcc/curl for ML stacks) |

## Libraries — the rules that matter

- **`sb uv add` is the only install that sticks.** marimo's in-editor install
  banner bypasses the lockfile; the next sync reverts it, and unpinned installs
  can drag torch onto PyPI's CUDA-13 builds (this broke a real env once).
- **The torch family must come from the `pytorch-cu128` index** — `sb init`
  pre-pins torch/torchvision/torchaudio/torchcodec in `[tool.uv.sources]`. Add
  any other torch-family lib (xformers, torchao, …) to that list **before**
  `sb uv add`-ing it.
- cu128-index versions can lag PyPI numbering (torchcodec is `0.11.x+cu128`,
  matched to torch 2.11, vs `0.17` on PyPI) — loosen version floors accordingly.
- Video decoding: `av` works out of the box; torchcodec additionally needs the
  cu128 build + ffmpeg (bootstrap installs it) + the NVIDIA lib path (the env
  block exports it) — all handled by `sb init --vision`.

## Persistence model

Everything durable lives under **`/work`** (the snapshot mount): venv, uv cache,
`HF_HOME=/work/hf` (model weights restore with the snapshot — no re-download),
`/work/notebooks`, `/work/data`. Anything outside `/work` — including apt
packages — dies with the sandbox (bootstrap reinstalls apt deps each boot).
Notebooks are additionally plain git-friendly `.py` files in your repo.

The public HTTPS URL + token **rotate on every create**; the endpoint does no
auth itself (marimo's token is the gate) — treat the URL like a password.
Billing is per-millisecond from ready until deletion: `sb down` when stepping away.

## Troubleshooting (field-tested)

| Symptom | Cause | Fix |
|---|---|---|
| "no capacity … retrying" / `RESOURCE_EXHAUSTED` | GPU pool at capacity | `sb` auto-retries ~20 min with backoff; persisting → later or forge-support@coreweave.com |
| Stuck CREATING > 15 min, or status `failed` | Platform-side wedged/failed create (unbilled) | `sb cleanup` then `sb marimo edit` |
| Notebook shows errors that don't match reality ("Reconnected" banner) | marimo resurrects cached session outputs; kernels go stale when the venv changes under them | `sb marimo restart` + hard-refresh; stubborn: `sb exec -- rm -rf /work/notebooks/__marimo__/session` first |
| `ImportError: libcudnn.so.9` / missing nvidia libs after mixed installs | cu12/cu13 wheels share install dirs and clobber each other | `sb exec -- "cd /work/project && uv sync --locked --reinstall"` (seconds, from cache) |
| torchcodec `Could not load this library` | cu128 build + ffmpeg + NVIDIA lib path all required | All baked in; from a bare shell use `sb exec` so the env block applies |
| `uv sync` says `Removed virtual environment… Creating…` on restore | Image Python micro-version drift | Normal — reinstalls from `/work` cache in ~2 s |
| gRPC `Channel is closed` traceback on exit | SDK async teardown noise | Filtered by `sb`; ignore any variant that leaks |

## Adopting an existing project

`sb init` refuses to overwrite a `pyproject.toml`. To adopt: add a `[tool.sb]`
section (set `tag` to your existing sandbox tag to keep its snapshot lineage —
state-file keys are compatible), pin torch-family packages to the cu128 index
(run `sb init` in an empty dir to copy the exact block), `uv lock`, then
`sb marimo edit`.
