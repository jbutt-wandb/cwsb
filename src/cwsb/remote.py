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


class CreateFailedError(RuntimeError):
    """A created sandbox landed in a terminal state (or never left CREATING)."""


# Transient, self-healing failures observed live: grpc UNAVAILABLE surfacing as
# "Service unavailable: failed to connect to all addresses ... tcp handshaker
# shutdown", and per-RPC "Deadline Exceeded" status polls. Both cleared within
# ~30-60 s while the sandbox kept running.
_CONN_ERROR_MARKERS = (
    "failed to connect",
    "tcp handshaker",
    "service unavailable",
    "deadline exceeded",
    "connection reset",
    "socket closed",
    "unavailable",
)


def _is_conn_error(e: BaseException) -> bool:
    try:
        from cwsandbox.exceptions import SandboxRequestTimeoutError, SandboxUnavailableError

        if isinstance(e, (SandboxRequestTimeoutError, SandboxUnavailableError)):
            return True
    except ImportError:
        pass
    msg = str(e).lower()
    return any(m in msg for m in _CONN_ERROR_MARKERS)


def with_conn_retry(fn, what: str, tries: int = 3, delay_s: float = 20.0,
                    total_s: float | None = None, quiet: bool = False):
    """Retry fn() across transient connection-level failures. Non-connection
    errors re-raise immediately. `total_s` (a time budget) overrides `tries`."""
    deadline = time.time() + total_s if total_s else None
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as e:
            if not _is_conn_error(e):
                raise
            wait = min(getattr(e, "retry_delay", None) or delay_s, 60.0)
            exhausted = (deadline is None and attempt >= tries) or (
                deadline is not None and time.time() + wait > deadline
            )
            if exhausted:
                raise
            if not quiet:
                print(f"  transient connection issue during {what} "
                      f"({type(e).__name__}) — retrying in {wait:.0f}s")
            time.sleep(wait)


def get_sandbox(proj: Project, state: dict, required: bool = True,
                tries: int = 3, delay_s: float = 20.0, patience_s: float | None = None):
    from cwsandbox import Sandbox

    sid = state.get("sandbox_id")
    if not sid:
        if required:
            sys.exit("No active sandbox for this project. Run `cwsb marimo edit` first.")
        return None
    try:
        return with_conn_retry(
            lambda: unwrap(Sandbox.from_id(sid, auth=auth(proj))),
            f"connecting to sandbox {sid[:8]}",
            tries=tries, delay_s=delay_s, total_s=patience_s,
        )
    except Exception as e:
        if required:
            sys.exit(f"Could not reach sandbox {sid}: {e}\n(If it's gone: `cwsb cleanup`, then `cwsb marimo edit`.)")
        return None


def sandbox_alive(sb) -> bool:
    """True while the sandbox is in a non-terminal state.

    Raises on persistent connection failures instead of returning False — a
    caller that deletes 'dead' sandboxes must never mistake an API blip for
    death (that would destroy a healthy sandbox's /work).
    """
    try:
        status = str(with_conn_retry(lambda: unwrap(sb.get_status()),
                                     "status check", tries=3, delay_s=15))
    except Exception as e:
        if _is_conn_error(e):
            raise
        return False  # not-found / terminal-only errors: genuinely gone
    return not any(t in status.upper() for t in ("TERMINAT", "COMPLETE", "FAIL"))


def sh(sb, cmd: str, timeout=None):
    """Attached shell exec with the /work env on PATH. Short commands only.
    Conn-retried — callers keep their commands idempotent (they are: probes,
    mkdir -p, ls, pid-file kills)."""
    kwargs = {"timeout_seconds": timeout} if timeout else {}
    return with_conn_retry(
        lambda: sb.exec(["bash", "-c", ENV_BLOCK + cmd], **kwargs).result(),
        "exec", tries=3, delay_s=20,
    )


def run_logged(sb, name: str, script: str, timeout_s: int, poll_s: int = 10) -> None:
    """Run a script detached on the sandbox, stream its log, raise on failure."""
    inner, log = f"/tmp/{name}.sh", f"/tmp/{name}.log"
    done, fail = f"/tmp/{name}.done", f"/tmp/{name}.fail"
    with_conn_retry(
        lambda: sb.write_file(inner, ("set -euo pipefail\n" + script).encode()).result(),
        f"{name} upload", tries=3, delay_s=20,
    )
    with_conn_retry(
        lambda: sb.exec(
            [
                "sh",
                "-c",
                # idempotent relaunch: rm sentinels + nohup; a lost-response retry
                # just restarts the (guarded) script
                f"rm -f {done} {fail}; nohup sh -c "
                f"'bash {inner} > {log} 2>&1 && touch {done} || touch {fail}' "
                f">/dev/null 2>&1 & echo launched",
            ]
        ).result(),
        f"{name} launch", tries=3, delay_s=20,
    )

    t0, offset = time.time(), 0
    while True:
        try:
            r = sb.exec(
                [
                    "sh",
                    "-c",
                    f"if [ -f {done} ]; then s=DONE; elif [ -f {fail} ]; then s=FAIL; else s=RUNNING; fi; "
                    f"echo $s; tail -c +{offset + 1} {log} 2>/dev/null",
                ]
            ).result()
        except Exception as e:  # poll blips must not kill a long remote job
            if not _is_conn_error(e) or time.time() - t0 > timeout_s:
                raise
            print(f"    [{name}] poll blip ({type(e).__name__}) — job still running remotely, retrying")
            time.sleep(poll_s)
            continue
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
        try:
            sb = unwrap(Sandbox.from_id(sb.sandbox_id, auth=auth(proj)))
        except Exception as e:
            if not _is_conn_error(e):
                raise
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


def wait_running(sb, overall_s: float) -> None:
    """Wait for RUNNING, riding out transient poll failures (a per-RPC
    'Deadline Exceeded' must not abort a create that is still progressing).
    Raises CreateFailedError on a terminal state or when the budget expires
    (>~15 min CREATING is the field-tested wedge threshold)."""
    try:
        from cwsandbox.exceptions import SandboxFailedError
    except ImportError:  # older SDKs
        class SandboxFailedError(Exception):  # type: ignore[no-redef]
            pass

    deadline = time.time() + overall_s
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise CreateFailedError(
                f"not RUNNING after {overall_s:.0f}s — treating as wedged (CREATING time is unbilled)"
            )
        try:
            sb.wait(timeout=min(remaining, 300))
            return
        except SandboxFailedError as e:
            raise CreateFailedError(f"sandbox entered FAILED state: {e}") from e
        except Exception as e:
            if not _is_conn_error(e):
                raise
            wait_s = min(getattr(e, "retry_delay", None) or 15, 60)
            print(f"  status-poll blip ({type(e).__name__}) — sandbox unaffected, retrying in {wait_s:.0f}s")
            time.sleep(wait_s)
            try:  # cheap terminal check so a FAILED sandbox isn't re-waited forever
                st = str(unwrap(sb.get_status())).upper()
            except Exception:
                continue
            if any(t in st for t in ("FAIL", "TERMINAT", "COMPLETE")):
                raise CreateFailedError(f"sandbox entered terminal state '{st.lower()}' while waiting")


def create_sandbox(proj: Project, lifetime_str: str, gpu: bool, fresh: bool, state: dict,
                   max_attempts: int = 3):
    """Create + wait + bootstrap, with auto-recovery from platform-side failed
    creates (observed 3x/12h: sandboxes landing in `failed`, unbilled). Saves
    state immediately after each create (crash safety). Returns the live sandbox."""
    from cwsandbox import (
        Endpoint,
        EndpointAuth,
        EndpointKind,
        FileSystemSnapshotOptions,
        Sandbox,
        Service,
        ServiceVisibility,
    )

    snap_id = None if fresh else state.get("snapshot_id")
    lifetime = parse_lifetime(lifetime_str)
    resources = {"cpu": "8", "memory": "32Gi", "gpu": 1} if gpu else {"cpu": "4", "memory": "8Gi"}
    overall_deadline = time.time() + 30 * 60  # total budget across capacity retries + recreates

    sb = None
    for attempt in range(1, max_attempts + 1):
        token = secrets.token_urlsafe(32)
        label = f" (attempt {attempt}/{max_attempts})" if attempt > 1 else ""
        print(f"Creating {'GPU' if gpu else 'CPU'} sandbox{label} (lifetime {lifetime_str}, "
              f"{'restoring snapshot ' + snap_id if snap_id else 'fresh /work'})...")
        budget_min = max(2.0, min(20.0, (overall_deadline - time.time()) / 60))
        sb = create_with_retry(
            proj,
            retry_minutes=budget_min,
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
        try:
            wait_running(sb, overall_s=min(900.0, max(60.0, overall_deadline - time.time())))
        except CreateFailedError as e:
            print(f"  create attempt {attempt} failed: {e}")
            try:
                unwrap(Sandbox.delete(sb.sandbox_id, auth=auth(proj), missing_ok=True))
            except Exception:
                pass  # failed sandboxes are often GC'd server-side already
            state.pop("sandbox_id", None)
            proj.save_state(state)
            sb = None
            if attempt == max_attempts or time.time() > overall_deadline:
                sys.exit(
                    "Create kept failing platform-side (unbilled). Try again shortly, "
                    "or escalate to forge-support@coreweave.com."
                )
            print("  recreating — fresh creates usually land on a healthy runner...")
            continue
        print(f"  RUNNING after {time.monotonic() - t0:.0f}s")
        break

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

    if sb is not None:
        try:
            alive = sandbox_alive(sb)
        except Exception as e:
            # A persistent connection blip must NOT be mistaken for death — the
            # reap below would destroy a healthy sandbox's /work.
            sys.exit(
                f"Can't verify sandbox {state['sandbox_id']} right now "
                f"(transient API issue: {type(e).__name__}). Refusing to assume it's dead.\n"
                "Try again in a minute, or `cwsb status`."
            )
    else:
        alive = False

    if sb is not None and not alive:
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
