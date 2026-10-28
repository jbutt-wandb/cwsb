"""Project discovery, [tool.sb] config, state file, .env loading."""

from __future__ import annotations

import getpass
import json
import os
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

STATE_NAME = ".sandbox-state.json"
MOUNT = "/work"

DEFAULTS = {
    "lifetime": "8h",
    "gpu": True,
    "port": 2718,
    "mount_size": "100Gi",
    "image": "python:3.12-bookworm",
    "apt_packages": [],  # ffmpeg is always installed; these are extras
}


def find_root(start: Path | None = None) -> Path:
    """Nearest ancestor (incl. cwd) containing pyproject.toml — like uv."""
    cur = (start or Path.cwd()).resolve()
    for d in [cur, *cur.parents]:
        if (d / "pyproject.toml").is_file():
            return d
    sys.exit(
        "No pyproject.toml found here or above.\n"
        "Run `sb init` to scaffold a sandbox-ready project in this directory,\n"
        "or cd into an existing project (or pass --project <dir>)."
    )


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", s).strip("-").lower() or "project"


@dataclass
class Project:
    root: Path
    tag: str
    lifetime: str
    gpu: bool
    port: int
    mount_size: str
    image: str
    apt_packages: list[str] = field(default_factory=list)

    @property
    def notebooks(self) -> Path:
        return self.root / "notebooks"

    @property
    def state_file(self) -> Path:
        return self.root / STATE_NAME

    def load_state(self) -> dict:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {}

    def save_state(self, state: dict) -> None:
        self.state_file.write_text(json.dumps(state, indent=2) + "\n")

    def load_dotenv(self) -> None:
        """Load KEY=VALUE pairs from <root>/.env (never overrides real env)."""
        env = self.root / ".env"
        if not env.exists():
            return
        for line in env.read_text().splitlines():
            line = line.strip().removeprefix("export ")
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def load_project(root_override: str | None = None) -> Project:
    root = find_root(Path(root_override)) if root_override else find_root()
    cfg = dict(DEFAULTS)
    try:
        data = tomllib.loads((root / "pyproject.toml").read_text())
        cfg.update(data.get("tool", {}).get("sb", {}))
    except tomllib.TOMLDecodeError as e:
        sys.exit(f"{root / 'pyproject.toml'} is not valid TOML: {e}")
    user = _slug(getpass.getuser())
    tag = cfg.get("tag") or f"sb-marimo-{_slug(root.name)}-{user}"
    return Project(
        root=root,
        tag=tag,
        lifetime=str(cfg["lifetime"]),
        gpu=bool(cfg["gpu"]),
        port=int(cfg["port"]),
        mount_size=str(cfg["mount_size"]),
        image=str(cfg["image"]),
        apt_packages=[str(p) for p in cfg.get("apt_packages", [])],
    )


def parse_lifetime(s: str) -> int:
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if s and s[-1] in units:
        return int(float(s[:-1]) * units[s[-1]])
    return int(s)


def uv_env() -> dict:
    """Env for local uv subprocesses: drop any active VIRTUAL_ENV so uv targets
    the project (it would otherwise warn it's ignoring the env var)."""
    return {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
