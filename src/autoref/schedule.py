"""Periodic `autoref update` via systemd user timers (crontab line as fallback)."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

UNIT_DIR = Path("~/.config/systemd/user").expanduser()


def parse_every(every: str) -> str:
    """'3d' / '12h' / '90m' -> systemd time span ('3d', '12h', '90min')."""
    m = re.fullmatch(r"\s*(\d+)\s*([dhm])\s*", every)
    if not m:
        raise ValueError("--every must look like 3d, 12h or 90m")
    n, u = m.groups()
    unit = {"d": "d", "h": "h", "m": "min"}[u]
    return f"{n}{unit}"


def cron_line(every: str, cmd: str) -> str:
    n, u = re.fullmatch(r"\s*(\d+)\s*([dhm])\s*", every).groups()
    spec = {"d": f"0 4 */{n} * *", "h": f"0 */{n} * * *", "m": f"*/{n} * * * *"}[u]
    return f"{spec} {cmd} >> ~/.cache/autoref-cron.log 2>&1"


def autoref_exe() -> str:
    exe = Path(sys.argv[0]).resolve()
    if exe.name == "autoref" and exe.exists():
        return str(exe)
    return shutil.which("autoref") or f"{sys.executable} -m autoref.cli"


def unit_names(topic: str) -> tuple[str, str]:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", topic)
    return f"autoref-{safe}.service", f"autoref-{safe}.timer"


def install(topic: str, config_path: Path, every: str) -> str:
    span = parse_every(every)
    cmd = f"{autoref_exe()} update -c {config_path.resolve()}"
    if not shutil.which("systemctl"):
        return f"systemctl not found. Add this line with `crontab -e`:\n{cron_line(every, cmd)}"
    svc, tmr = unit_names(topic)
    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    (UNIT_DIR / svc).write_text(
        f"[Unit]\nDescription=autoref update for topic {topic}\nAfter=network-online.target\n\n"
        f"[Service]\nType=oneshot\nWorkingDirectory={config_path.resolve().parent}\n"
        f"EnvironmentFile=-%h/.config/autoref/env\nExecStart={cmd}\n")
    (UNIT_DIR / tmr).write_text(
        f"[Unit]\nDescription=Run autoref update for {topic} every {span}\n\n"
        f"[Timer]\nOnBootSec=15min\nOnUnitActiveSec={span}\nRandomizedDelaySec=10min\n\n"
        f"[Install]\nWantedBy=timers.target\n")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", tmr], check=True)
    return (f"Installed {UNIT_DIR / tmr} (every {span}).\n"
            f"API keys for the timer go into ~/.config/autoref/env (KEY=value lines).\n"
            f"Logs: journalctl --user -u {svc}")


def remove(topic: str) -> str:
    svc, tmr = unit_names(topic)
    if shutil.which("systemctl"):
        subprocess.run(["systemctl", "--user", "disable", "--now", tmr], check=False)
    for u in (svc, tmr):
        (UNIT_DIR / u).unlink(missing_ok=True)
    if shutil.which("systemctl"):
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    return f"Removed {tmr} and {svc}."


def status(topic: str) -> str:
    _, tmr = unit_names(topic)
    if not (UNIT_DIR / tmr).exists():
        return "No timer installed."
    r = subprocess.run(["systemctl", "--user", "list-timers", tmr, "--no-pager"], capture_output=True, text=True)
    return r.stdout.strip() or r.stderr.strip()
