"""Sandbox-side operations. Faithful port of the field-tested launcher logic:
detached long-running commands (attached execs die at the SDK's ~300 s request
timeout), two-phase readiness, capacity retries, snapshot-safe teardown."""

from __future__ import annotations

import logging
import os
import secrets
import sys
import time
from pathlib import Path

from .project import MOUNT, Project, parse_lifetime

# Everything persistent lives under /work (the only thing snapshots capture).
ENV_BLOCK = (
    'export HOME="${HOME:-/root}"\n'
    "export UV_PYTHON_INSTALL_DIR=/work/uv/python\n"
    "export UV_CACHE_DIR=/work/uv/cache\n"
    "export UV_PROJECT_ENVIRONMENT=/work/venv\n"
    "export HF_HOME=/work/hf\n"
    "export PATH=/work/bin:/work/venv/bin:$PATH\n"
    # pip-installed NVIDIA libs (npp for torchcodec, etc.) aren't on the default
    # linker path; torch only preloads its own deps.
    'export LD_LIBRARY_PATH="$(ls -d /work/venv/lib/python3*/site-packages/nvidia/*/lib 2>/dev/null'
    " | tr '\\n' ':')${LD_LIBRARY_PATH:-}\"\n"
)


def quiet_grpc_teardown_noise() -> None:
    """The SDK's async gRPC channel logs a harmless `UsageError: Channel is
    closed.` traceback during interpreter shutdown. Drop just those records."""

    class _Drop(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if "Channel is closed" in record.getMessage():
                return False
            exc = record.exc_info[1] if record.exc_info else None
            return "Channel is closed" not in str(exc or "")

    for name in ("asyncio", "grpc", "grpc.aio"):
        logging.getLogger(name).addFilter(_Drop())


def auth(proj: Project):
    from cwsandbox import AuthStrategy

    proj.load_dotenv()
    if not os.environ.get("CWSANDBOX_API_KEY"):
        sys.exit(
            "CWSANDBOX_API_KEY is not set.\n"
            f"Put it in {proj.root}/.env (gitignored) — token with the 'Sandbox User'\n"
            "role from https://console.coreweave.com/tokens"
        )
    return AuthStrategy.COREWEAVE_API_KEY


def unwrap(obj):
    """Some SDK calls return an OperationRef (observed in 1.19.x) — resolve it."""
    return obj.result() if hasattr(obj, "result") else obj


def get_sandbox(proj: Project, state: dict, required: bool = True):
    from cwsandbox import Sandbox

    sid = state.get("sandbox_id")
    if not sid:
        if required:
            sys.exit("No active sandbox for this project. Run `cwsb marimo edit` first.")
        return None
    try:
        return unwrap(Sandbox.from_id(sid, auth=auth(proj)))
    except Exception as e:
        if required:
            sys.exit(f"Could not reach sandbox {sid}: {e}\n(If it's gone: `cwsb cleanup`, then `cwsb marimo edit`.)")
        return None


def sandbox_alive(sb) -> bool:
    try:
        status = str(unwrap(sb.get_status()))
    except Exception:
        return False
    return not any(t in status.upper() for t in ("TERMINAT", "COMPLETE", "FAIL"))


def sh(sb, cmd: str, timeout=None):
    """Attached shell exec with the /work env on PATH. Short commands only."""
    kwargs = {"timeout_seconds": timeout} if timeout else {}
    return sb.exec(["bash", "-c", ENV_BLOCK + cmd], **kwargs).result()


def run_logged(sb, name: str, script: str, timeout_s: int, poll_s: int = 10) -> None:
    """Run a script detached on the sandbox, stream its log, raise on failure."""
    inner, log = f"/tmp/{name}.sh", f"/tmp/{name}.log"
    done, fail = f"/tmp/{name}.done", f"/tmp/{name}.fail"
    sb.write_file(inner, ("set -euo pipefail\n" + script).encode()).result()
    sb.exec(
        [
            "sh",
            "-c",
            f"rm -f {done} {fail}; nohup sh -c "
            f"'bash {inner} > {log} 2>&1 && touch {done} || touch {fail}' "
            f">/dev/null 2>&1 & echo launched",
        ]
    ).result()

    t0, offset = time.time(), 0
    while True:
        r = sb.exec(
            [
                "sh",
                "-c",
                f"if [ -f {done} ]; then s=DONE; elif [ -f {fail} ]; then s=FAIL; else s=RUNNING; fi; "
                f"echo $s; tail -c +{offset + 1} {log} 2>/dev/null",
            ]
        ).result()
        out = r.stdout or ""
        status, _, new = out.partition("\n")
        if new:
            offset += len(new.encode())
            for line in new.splitlines():
                print(f"    [{name}] {line}")
        if status.strip() == "DONE":
            return
        if status.strip() == "FAIL":
            raise RuntimeError(f"{name} failed — log above (full log: {log} on the sandbox)")
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"{name} did not finish within {timeout_s}s")
        time.sleep(poll_s)


def bootstrap_script(proj: Project) -> str:
    apt = " ".join(["ffmpeg", *proj.apt_packages])
    return f"""{ENV_BLOCK}
# apt packages live in the container layer (NOT snapshotted) — reinstall per sandbox.
need=""
for p in {apt}; do dpkg -s "$p" >/dev/null 2>&1 || need="$need $p"; done
if [ -n "$need" ]; then
  echo "apt: installing$need..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null && apt-get install -y -qq $need >/dev/null
fi
mkdir -p /work/bin /work/notebooks /work/project /work/hf /work/data
if [ ! -x /work/bin/uv ]; then
  echo "installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/work/bin UV_NO_MODIFY_PATH=1 sh
fi
cd /work/project
echo "syncing environment (first run downloads the stack; snapshot restores are seconds)..."
uv sync --locked
mkdir -p "$HOME/.config/marimo"
printf '[package_management]\\nmanager = "uv"\\n' > "$HOME/.config/marimo/marimo.toml"
echo "bootstrap complete"
"""


def marimo_launch_script(proj: Project, mode: str, app_file: str | None, public: bool) -> str:
    """mode: 'edit' (whole notebooks dir, token auth) or 'run' (one app file).
    NB: no pkill — this whole script is one bash -c argv, so any pattern matching
    'marimo' also matches our own parent shell. Restarts kill by PID file."""
    if mode == "edit":
        cmd = f'/work/venv/bin/marimo edit /work/notebooks --token --token-password "$MARIMO_TOKEN"'
    else:
        tok = "--no-token" if public else '--token --token-password "$MARIMO_TOKEN"'
        cmd = f"/work/venv/bin/marimo run /work/notebooks/{app_file} {tok}"
    return f"""{ENV_BLOCK}
nohup {cmd} \\
    --host 0.0.0.0 --port {proj.port} --headless \\
    --skip-update-check \\
    > /tmp/marimo.log 2>&1 &
echo $! > /tmp/marimo.pid
echo "marimo pid $(cat /tmp/marimo.pid)"
"""


def stop_marimo(sb) -> None:
    sh(sb, "[ -f /tmp/marimo.pid ] && kill $(cat /tmp/marimo.pid) 2>/dev/null; sleep 1; true")


def probe_marimo(sb, proj: Project) -> bool:
    r = sh(sb, f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{proj.port}/health || true")
    code = (r.stdout or "").strip()
    return code.isdigit() and 200 <= int(code) < 500


def start_marimo(sb, proj: Project, mode: str = "edit", app_file: str | None = None,
                 public: bool = False) -> None:
    stop_marimo(sb)
    sh(sb, marimo_launch_script(proj, mode, app_file, public))
    print("  waiting for marimo to answer...")
    for _ in range(60):
        if probe_marimo(sb, proj):
            return
        time.sleep(2)
    log = sh(sb, "tail -n 50 /tmp/marimo.log 2>/dev/null || echo '(no log)'")
    raise RuntimeError(f"marimo never came up. Last log lines:\n{log.stdout}")


def service_url(sb, proj: Project) -> str:
    from cwsandbox import Sandbox

    for _ in range(10):  # service_urls is empty until the platform assigns them
        urls = {int(p): u for p, _name, u in sb.service_urls}
        if proj.port in urls:
            return urls[proj.port]
        time.sleep(3)
        sb = unwrap(Sandbox.from_id(sb.sandbox_id, auth=auth(proj)))
    raise RuntimeError(f"no service URL assigned for port {proj.port}")


def create_with_retry(proj: Project, retry_minutes: int = 20, **run_kwargs):
    """Capacity comes and goes (SandboxResourceExhaustedError etc. are marked
    retryable by the SDK) — keep trying with backoff instead of dying."""
    from cwsandbox import Sandbox
    from cwsandbox.exceptions import SandboxResourceExhaustedError, SandboxUnavailableError

    deadline, attempt = time.time() + retry_minutes * 60, 0
    while True:
        attempt += 1
        try:
            return unwrap(Sandbox.run("sleep", "infinity", **run_kwargs))
        except (SandboxResourceExhaustedError, SandboxUnavailableError) as e:
            delay = getattr(e, "retry_delay", None) or min(30 * attempt, 120)
            if time.time() + delay > deadline:
                sys.exit(
                    f"No capacity after {retry_minutes} min of retries "
                    f"(last: {type(e).__name__}). Try later or escalate to forge-support@coreweave.com."
                )
            print(f"  no capacity (attempt {attempt}: {type(e).__name__}) — retrying in {delay:.0f}s")
            time.sleep(delay)


def push_project_files(sb, proj: Project) -> None:
    for f in ("pyproject.toml", "uv.lock"):
        p = proj.root / f
        if not p.exists():
            sys.exit(f"{f} missing — run `uv lock` in {proj.root} first.")
        sb.write_file(f"/work/project/{f}", p.read_bytes()).result()


def push_notebooks(sb, proj: Project) -> list[str]:
    pushed = []
    for p in sorted(proj.notebooks.glob("*.py")):
        sb.write_file(f"/work/notebooks/{p.name}", p.read_bytes()).result()
        pushed.append(p.name)
    return pushed


def read_remote_file(sb, path: str) -> bytes:
    r = sb.read_file(path).result()
    data = getattr(r, "content", r)
    return data if isinstance(data, bytes) else bytes(data)


def pull_notebooks(sb, proj: Project) -> None:
    proj.notebooks.mkdir(exist_ok=True)
    r = sh(sb, "ls -1 /work/notebooks/*.py 2>/dev/null || true")
    names = [Path(line).name for line in (r.stdout or "").split() if line.endswith(".py")]
    if not names:
        print("  no notebooks on the sandbox")
        return
    for name in names:
        remote = read_remote_file(sb, f"/work/notebooks/{name}")
        local = proj.notebooks / name
        if local.exists() and local.read_bytes() == remote:
            print(f"  = {name} (unchanged)")
            continue
        verb = "~ updated" if local.exists() else "+ new"
        local.write_bytes(remote)
        print(f"  {verb} {name}")


def remote_sync(sb, proj: Project) -> None:
    push_project_files(sb, proj)
    run_logged(sb, "uvsync", "cd /work/project\n" + ENV_BLOCK + "uv sync --locked\n", timeout_s=900)


def tokenized_url(state: dict, file: str | None = None) -> str:
    url, tok = state.get("url", ""), state.get("token", "")
    qs = f"?access_token={tok}"
    if file:
        qs = f"?file={file}&access_token={tok}"
    return f"{url}{qs}"


def print_access(state: dict, file: str | None = None) -> None:
    if not state.get("url"):
        print(f"\n  sandbox {state.get('sandbox_id')} exists but no URL was saved "
              "(a previous create likely crashed — `cwsb cleanup`, then retry)\n")
        return
    print(f"\n  marimo:  {tokenized_url(state, file)}")
    print(f"  (token: {state.get('token')})\n")


def create_sandbox(proj: Project, lifetime_str: str, gpu: bool, fresh: bool, state: dict):
    """Create + wait + bootstrap. Saves state immediately after create (crash
    safety). Returns the live sandbox. Raises with logs dumped on failure."""
    from cwsandbox import (
        Endpoint,
        EndpointAuth,
        EndpointKind,
        FileSystemSnapshotOptions,
        Sandbox,  # noqa: F401  (create_with_retry imports its own)
        Service,
        ServiceVisibility,
    )

    snap_id = None if fresh else state.get("snapshot_id")
    token = secrets.token_urlsafe(32)
    lifetime = parse_lifetime(lifetime_str)
    resources = {"cpu": "8", "memory": "32Gi", "gpu": 1} if gpu else {"cpu": "4", "memory": "8Gi"}

    print(f"Creating {'GPU' if gpu else 'CPU'} sandbox (lifetime {lifetime_str}, "
          f"{'restoring snapshot ' + snap_id if snap_id else 'fresh /work'})...")
    sb = create_with_retry(
        proj,
        auth=auth(proj),
        container_image=proj.image,
        placement_mode="serverless",
        resources=resources,
        max_lifetime_seconds=lifetime,
        tags=[proj.tag, f"run-{time.strftime('%Y%m%d-%H%M%S')}"],
        environment_variables={"MARIMO_TOKEN": token},
        services=[
            Service(
                port=proj.port,
                name="marimo",
                visibility=ServiceVisibility.PUBLIC,
                endpoint=Endpoint(kind=EndpointKind.HTTPS, auth=EndpointAuth.OPEN),
            )
        ],
        file_system_snapshot=FileSystemSnapshotOptions(
            mount_path=MOUNT, size=proj.mount_size, file_system_snapshot_id=snap_id
        ),
    )
    state.update(
        sandbox_id=sb.sandbox_id,
        token=token,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        lifetime_seconds=lifetime,
    )
    proj.save_state(state)

    if snap_id:
        print("  waiting for RUNNING (snapshot restore scales with size — minutes is normal)...")
    else:
        print("  waiting for RUNNING...")
    t0 = time.monotonic()
    sb.wait(timeout=1200)
    print(f"  RUNNING after {time.monotonic() - t0:.0f}s")
    if gpu:
        print(f"  GPU: {sh(sb, 'nvidia-smi -L || true').stdout.strip()}")

    try:
        print("  bootstrapping environment...")
        push_project_files(sb, proj)
        run_logged(sb, "bootstrap", bootstrap_script(proj), timeout_s=1500)
        pushed = push_notebooks(sb, proj)
        print(f"  pushed notebooks: {', '.join(pushed) or '(none)'}")
    except Exception:
        print("\n--- bootstrap failed; sandbox left RUNNING for debugging ---")
        print("    inspect: cwsb exec -- tail -n 100 /tmp/bootstrap.log")
        print("    reap:    cwsb cleanup")
        raise
    return sb


def ensure_up(proj: Project, lifetime: str | None, gpu: bool | None, fresh: bool,
              mode: str, app_file: str | None = None, public: bool = False):
    """Idempotent: reuse a live sandbox, reap a dead one, create if absent.
    Ensures marimo is serving in the requested mode. Returns (sb, state)."""
    state = proj.load_state()
    sb = get_sandbox(proj, state, required=False)
    created = False

    if sb is not None and not sandbox_alive(sb):
        from cwsandbox import Sandbox

        print(f"previous sandbox {state['sandbox_id']} is gone/dead — reaping and recreating...")
        unwrap(Sandbox.delete(state["sandbox_id"], auth=auth(proj), missing_ok=True))
        state.pop("sandbox_id", None)
        state.pop("url", None)
        sb = None

    if sb is None:
        sb = create_sandbox(proj, lifetime or proj.lifetime,
                            proj.gpu if gpu is None else gpu, fresh, state)
        created = True
    else:
        if fresh:
            sys.exit("--fresh requires no running sandbox — `cwsb down` (or `cwsb cleanup`) first.")
        push_notebooks(sb, proj)

    mode_changed = state.get("server_mode") != mode or (mode == "run" and state.get("app_file") != app_file)
    if created or mode_changed or not probe_marimo(sb, proj):
        verb = "starting" if created else "restarting"
        print(f"  {verb} marimo ({mode}{' ' + app_file if app_file else ''})...")
        try:
            start_marimo(sb, proj, mode, app_file, public)
        except Exception:
            if created:
                print("\n--- marimo launch failed; sandbox left RUNNING for debugging ---")
                print("    inspect: cwsb exec -- tail -n 100 /tmp/marimo.log")
            raise
        state["server_mode"] = mode
        state["app_file"] = app_file
        state["public"] = public

    if not state.get("url"):
        state["url"] = service_url(sb, proj)
    proj.save_state(state)
    return sb, state
