"""cwsb — CoreWeave Sandbox marimo notebooks: GPU kernels, local browser.

  cwsb init [--vision] [--cpu]          scaffold a sandbox-ready project here
  cwsb marimo edit [nb.py] [--gpu|--cpu] [--lifetime 8h] [--fresh] [--no-browser]
  cwsb marimo new [nb.py]               new notebook, opened in the browser
  cwsb marimo run <nb.py> [--public]    serve a notebook as a read-only app
  cwsb marimo restart                   fresh kernels, same URL
  cwsb uv add|remove|sync [...]         manage libraries (git-tracked, live in seconds)
  cwsb files add <path...> [--to dir]   upload files (default /work/data/)
  cwsb files get <remote...> [--to dir] download files
  cwsb push | cwsb pull                   sync notebooks/ <-> sandbox
  cwsb status | cwsb down | cwsb cleanup | cwsb exec -- <cmd>

The sandbox is an implementation detail: `cwsb marimo edit` creates/reuses/heals
it as needed. State and config are per-project (nearest pyproject.toml, like uv);
CWSANDBOX_API_KEY auto-loads from <project>/.env.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from . import remote
from .project import Project, load_project, uv_env
from .scaffold import cmd_init

MINIMAL_NB = '''\
import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)
'''


def _project(args) -> Project:
    return load_project(getattr(args, "project", None))


def _resolve_notebook(proj: Project, arg: str | None, create: bool) -> str | None:
    """Return the notebook's basename for deep-linking; materialize it locally
    if asked for a file that doesn't exist yet (marimo-native: edit creates)."""
    if not arg:
        return None
    name = Path(arg).name
    if not name.endswith(".py"):
        name += ".py"
    candidates = [Path(arg), proj.notebooks / name]
    if not any(p.is_file() for p in candidates):
        if not create:
            sys.exit(f"notebook not found: {arg} (looked at {', '.join(map(str, candidates))})")
        proj.notebooks.mkdir(exist_ok=True)
        (proj.notebooks / name).write_text(MINIMAL_NB)
        print(f"  created notebooks/{name}")
    elif Path(arg).is_file() and Path(arg).resolve().parent != proj.notebooks.resolve():
        # a path outside notebooks/ — copy it in so push picks it up
        proj.notebooks.mkdir(exist_ok=True)
        (proj.notebooks / name).write_bytes(Path(arg).read_bytes())
        print(f"  copied {arg} -> notebooks/{name}")
    return name


def _open(state: dict, file: str | None, no_browser: bool) -> None:
    remote.print_access(state, file)
    if not no_browser and state.get("url"):
        webbrowser.open(remote.tokenized_url(state, file))


def cmd_marimo_edit(args) -> None:
    proj = _project(args)
    gpu = True if args.gpu else (False if args.cpu else None)
    nb = _resolve_notebook(proj, args.notebook, create=True)
    sb, state = remote.ensure_up(proj, args.lifetime, gpu, args.fresh, mode="edit")
    _open(state, nb, args.no_browser)
    secs = state.get("lifetime_seconds")
    cap = f"{secs / 3600:g}h" if secs else args.lifetime or proj.lifetime
    print(f"  lifetime cap {cap} from creation — `cwsb down` to save + stop billing")


def cmd_marimo_new(args) -> None:
    proj = _project(args)
    name = args.notebook
    if not name:
        n, name = 1, "untitled.py"
        while (proj.notebooks / name).exists():
            n += 1
            name = f"untitled_{n}.py"
    args.notebook = name
    args.gpu = args.cpu = args.fresh = False
    args.lifetime = None
    cmd_marimo_edit(args)


def cmd_marimo_run(args) -> None:
    proj = _project(args)
    nb = _resolve_notebook(proj, args.notebook, create=False)
    if args.public:
        print("⚠️  --public serves this app UNAUTHENTICATED on an internet-reachable URL.")
    sb, state = remote.ensure_up(proj, args.lifetime, None, False, mode="run",
                                 app_file=nb, public=args.public)
    url = state["url"] if args.public else remote.tokenized_url(state)
    print(f"\n  app ({'PUBLIC' if args.public else 'token-protected'}): {url}\n")
    print("  note: edit and run share one port — `cwsb marimo edit` switches back.")
    if not args.no_browser:
        webbrowser.open(url)


def cmd_marimo_restart(args) -> None:
    proj = _project(args)
    state = proj.load_state()
    sb = remote.get_sandbox(proj, state)
    print("restarting marimo (open sessions will prompt to reconnect — refresh the page)...")
    remote.start_marimo(sb, proj, state.get("server_mode") or "edit",
                        state.get("app_file"), bool(state.get("public")))
    remote.print_access(state)


def cmd_uv(args) -> None:
    proj = _project(args)
    verb = args.uv_verb
    if verb in ("add", "remove"):
        uv_cmd = ["uv", verb, "--no-sync", *args.packages]
        print(f"  local: {' '.join(uv_cmd)}")
        subprocess.run(uv_cmd, cwd=proj.root, check=True, env=uv_env())
    else:  # sync
        lock = proj.root / "uv.lock"
        py = proj.root / "pyproject.toml"
        if not lock.exists() or py.stat().st_mtime > lock.stat().st_mtime:
            print("  re-locking...")
            subprocess.run(["uv", "lock"], cwd=proj.root, check=True, env=uv_env())

    state = proj.load_state()
    sb = remote.get_sandbox(proj, state, required=False)
    if sb is None or not remote.sandbox_alive(sb):
        print("no running sandbox — lockfile updated; it syncs on the next `cwsb marimo edit`.")
        return
    print("  syncing sandbox venv...")
    remote.remote_sync(sb, proj)
    print("done — new packages are importable in running notebooks (re-run the cell).")


def cmd_files_add(args) -> None:
    proj = _project(args)
    sb = remote.get_sandbox(proj, proj.load_state())
    dest = args.to.rstrip("/")
    if not dest.startswith(remote.MOUNT):
        print(f"  warning: {dest} is outside {remote.MOUNT} — it will NOT survive `cwsb down`")
    remote.sh(sb, f"mkdir -p '{dest}'")
    for f in args.files:
        p = Path(f).expanduser()
        if not p.is_file():
            sys.exit(f"not a file: {f}")
        size_mb = p.stat().st_size / 1e6
        if size_mb > 500:
            sys.exit(
                f"{p.name} is {size_mb:.0f} MB — too big to stream from this machine.\n"
                f"Download it on the sandbox at datacenter speed instead:\n"
                f"  cwsb exec -- curl -L -o {dest}/{p.name} <url>"
            )
        sb.write_file(f"{dest}/{p.name}", p.read_bytes()).result()
        print(f"  ↑ {p.name} ({size_mb:.1f} MB) -> {dest}/{p.name}")


def cmd_files_get(args) -> None:
    proj = _project(args)
    sb = remote.get_sandbox(proj, proj.load_state())
    out_dir = Path(args.to)
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in args.files:
        local = out_dir / Path(r).name
        local.write_bytes(remote.read_remote_file(sb, r))
        print(f"  ↓ {r} -> {local} ({local.stat().st_size / 1e6:.1f} MB)")


def cmd_push(args) -> None:
    proj = _project(args)
    sb = remote.get_sandbox(proj, proj.load_state())
    pushed = remote.push_notebooks(sb, proj)
    print(f"pushed: {', '.join(pushed) or '(none)'}")


def cmd_pull(args) -> None:
    proj = _project(args)
    sb = remote.get_sandbox(proj, proj.load_state())
    remote.pull_notebooks(sb, proj)


def cmd_status(args) -> None:
    proj = _project(args)
    state = proj.load_state()
    if not state.get("sandbox_id"):
        print(f"  no sandbox ({'snapshot ' + state['snapshot_id'] + ' ready to restore' if state.get('snapshot_id') else 'nothing yet'})")
        print("  `cwsb marimo edit` to start one")
        return
    sb = remote.get_sandbox(proj, state)
    print(f"  project:  {proj.root.name} (tag {proj.tag})")
    print(f"  sandbox:  {state['sandbox_id']}")
    print(f"  status:   {remote.unwrap(sb.get_status())}")
    print(f"  mode:     marimo {state.get('server_mode', 'edit')}"
          + (f" ({state.get('app_file')})" if state.get("app_file") else ""))
    print(f"  created:  {state.get('created_at')}  (lifetime {state.get('lifetime_seconds')}s)")
    print(f"  snapshot: {state.get('snapshot_id') or '(none yet)'}")
    if state.get("url"):
        remote.print_access(state)


def cmd_down(args) -> None:
    from cwsandbox import Sandbox

    proj = _project(args)
    state = proj.load_state()
    sb = remote.get_sandbox(proj, state)
    print("Pulling notebooks before anything destructive...")
    remote.pull_notebooks(sb, proj)

    size = (remote.sh(sb, "du -sh /work 2>/dev/null | cut -f1").stdout or "").strip() or "?"
    print(f"Snapshotting /work — {size} (venv + caches + notebooks + data)...")
    print("  snapshot time scales with size (~1 min at 7 GB, ~7 min at 24 GB) — not hung")
    snap_id, last_err = None, None
    for attempt in range(5):
        try:
            snap_id = remote.unwrap(sb.snapshot())
            break
        except Exception as e:  # snapshot RPC flakes — retry, don't discard the mount
            last_err = e
            print(f"  snapshot attempt {attempt + 1}/5 failed: {e}; retrying in 60s")
            time.sleep(60)
    if snap_id is None:
        sys.exit(f"Snapshot failed after 5 attempts ({last_err}). Sandbox left running — retry `cwsb down`.")
    snap_id = getattr(snap_id, "snapshot_id", snap_id)
    print(f"  snapshot ready: {snap_id}")

    old_snap = state.get("snapshot_id")
    print("Deleting sandbox...")
    remote.unwrap(Sandbox.delete(state["sandbox_id"], auth=remote.auth(proj), missing_ok=True))
    state["snapshot_id"] = str(snap_id)
    for k in ("sandbox_id", "url", "server_mode", "app_file", "public"):
        state.pop(k, None)
    proj.save_state(state)
    if old_snap and old_snap != str(snap_id):
        print(f"Deleting previous snapshot {old_snap}...")
        try:
            remote.unwrap(Sandbox.delete_snapshot(old_snap, missing_ok=True, auth=remote.auth(proj)))
        except Exception as e:
            print(f"  (could not delete old snapshot: {e} — delete manually later)")
    print(f"Down. Next `cwsb marimo edit` restores from {snap_id}.")


def cmd_cleanup(args) -> None:
    from cwsandbox import Sandbox

    proj = _project(args)
    n = 0
    for s in remote.unwrap(Sandbox.list(tags=[proj.tag], auth=remote.auth(proj))):
        print(f"  deleting {s.sandbox_id}")
        remote.unwrap(Sandbox.delete(s.sandbox_id, auth=remote.auth(proj), missing_ok=True))
        n += 1
    state = proj.load_state()
    changed = False
    for k in ("sandbox_id", "url", "server_mode", "app_file", "public"):
        changed = state.pop(k, None) is not None or changed
    if changed:
        proj.save_state(state)
    print(f"reaped {n} sandbox(es) tagged '{proj.tag}'. Snapshots were NOT touched.")


def cmd_exec(args) -> None:
    proj = _project(args)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd  # REMAINDER keeps the "--"
    if not cmd:
        sys.exit("usage: cwsb exec -- <command...>")
    sb = remote.get_sandbox(proj, proj.load_state())
    r = remote.sh(sb, " ".join(cmd))
    if r.stdout:
        print(r.stdout, end="")
    if r.stderr:
        print(r.stderr, end="", file=sys.stderr)
    sys.exit(r.returncode or 0)


def main() -> None:
    remote.quiet_grpc_teardown_noise()
    ap = argparse.ArgumentParser(
        prog="cwsb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--project", help="project directory (default: walk up from cwd to pyproject.toml)")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="scaffold a sandbox-ready marimo project in the current directory")
    p.add_argument("--vision", action="store_true", help="add the VL stack (transformers, torchvision, torchcodec, av, ...)")
    p.add_argument("--cpu", action="store_true", help="CPU-only preset (no torch, gpu=false)")
    p.set_defaults(func=cmd_init)

    pm = sub.add_parser("marimo", help="marimo verbs against the sandbox")
    msub = pm.add_subparsers(dest="marimo_command", required=True)

    p = msub.add_parser("edit", help="ensure sandbox + marimo editor, open browser")
    p.add_argument("notebook", nargs="?", help="notebook to open (created if missing)")
    p.add_argument("--gpu", action="store_true", help="force GPU (overrides [tool.cwsb])")
    p.add_argument("--cpu", action="store_true", help="force CPU (overrides [tool.cwsb])")
    p.add_argument("--lifetime", default=None, help="hard cap if creating (e.g. 90m, 8h, 2d)")
    p.add_argument("--fresh", action="store_true", help="ignore saved snapshot when creating")
    p.add_argument("--no-browser", action="store_true", help="print the URL instead of opening it")
    p.set_defaults(func=cmd_marimo_edit)

    p = msub.add_parser("new", help="create a new notebook and open it")
    p.add_argument("notebook", nargs="?", help="name (default untitled.py)")
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_marimo_new)

    p = msub.add_parser("run", help="serve a notebook as a read-only app (token-protected unless --public)")
    p.add_argument("notebook")
    p.add_argument("--lifetime", default=None)
    p.add_argument("--public", action="store_true", help="serve WITHOUT auth on the public URL")
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_marimo_run)

    msub.add_parser("restart", help="restart the marimo server (fresh kernels, same URL)").set_defaults(
        func=cmd_marimo_restart
    )

    pu = sub.add_parser("uv", help="manage sandbox libraries with uv semantics")
    usub = pu.add_subparsers(dest="uv_verb", required=True)
    p = usub.add_parser("add", help="add packages: lock locally, sync sandbox venv")
    p.add_argument("packages", nargs="+")
    p.set_defaults(func=cmd_uv)
    p = usub.add_parser("remove", help="remove packages")
    p.add_argument("packages", nargs="+")
    p.set_defaults(func=cmd_uv)
    usub.add_parser("sync", help="re-lock if pyproject changed + sync sandbox").set_defaults(func=cmd_uv)

    pf = sub.add_parser("files", help="move files to/from the sandbox")
    fsub = pf.add_subparsers(dest="files_command", required=True)
    p = fsub.add_parser("add", help="upload files (default dest /work/data/; <500 MB each)")
    p.add_argument("files", nargs="+")
    p.add_argument("--to", default="/work/data")
    p.set_defaults(func=cmd_files_add)
    p = fsub.add_parser("get", help="download files by remote path")
    p.add_argument("files", nargs="+")
    p.add_argument("--to", default=".")
    p.set_defaults(func=cmd_files_get)

    sub.add_parser("push", help="local notebooks/*.py -> sandbox").set_defaults(func=cmd_push)
    sub.add_parser("pull", help="sandbox notebooks -> local notebooks/").set_defaults(func=cmd_pull)
    sub.add_parser("status", help="sandbox state, mode, URL + token").set_defaults(func=cmd_status)
    sub.add_parser("down", help="pull notebooks, snapshot /work, delete sandbox").set_defaults(func=cmd_down)
    sub.add_parser("cleanup", help="delete ALL of this project's sandboxes (no snapshot)").set_defaults(
        func=cmd_cleanup
    )
    p = sub.add_parser("exec", help="run a shell command on the sandbox (after --)")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_exec)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
