"""Start/stop local SearXNG for *won / *woff."""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import requests

from config import SEARXNG_MANAGE, SEARXNG_START_TIMEOUT, SEARXNG_URL

COMPOSE_FILE = Path(__file__).resolve().parent / "searxng" / "docker-compose.yml"
_DOCKER_CANDIDATES = (
    "docker",
    "/usr/bin/docker",
    "/usr/local/bin/docker",
    "/snap/bin/docker",
)


def searxng_ready(timeout: float = 2.0) -> bool:
    if not SEARXNG_URL:
        return False
    url = SEARXNG_URL.rstrip("/") + "/search"
    try:
        response = requests.get(
            url,
            params={"q": "coreline-ping", "format": "json"},
            timeout=timeout,
            headers={"Accept": "application/json", "User-Agent": "CoreLine/1.0"},
        )
    except requests.RequestException:
        return False
    return response.status_code == 200


def _docker_bin() -> str | None:
    for candidate in _DOCKER_CANDIDATES:
        path = shutil.which(candidate) if not candidate.startswith("/") else candidate
        if path and Path(path).is_file():
            return path
    return shutil.which("docker-compose")


def _compose_argv(*args: str) -> list[str] | None:
    docker = _docker_bin()
    if not docker:
        return None
    if docker.endswith("docker-compose") or docker == "docker-compose":
        return [docker, "-f", str(COMPOSE_FILE), *args]
    return [docker, "compose", "-f", str(COMPOSE_FILE), *args]


def _run_compose(*args: str) -> tuple[bool, str]:
    argv = _compose_argv(*args)
    if not argv:
        return False, "Docker не найден. Установи Docker или оставь SearXNG запущенным вручную."
    if not COMPOSE_FILE.is_file():
        return False, f"Нет файла {COMPOSE_FILE}"
    try:
        proc = subprocess.run(
            argv,
            cwd=str(COMPOSE_FILE.parent),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Не удалось выполнить docker compose: {exc}"
    output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        return False, output or f"docker compose {' '.join(args)} завершился с кодом {proc.returncode}"
    return True, output


def start_searxng() -> tuple[bool, str]:
    """Bring SearXNG up. Returns (ok, user-facing message)."""
    if searxng_ready():
        return True, "SearXNG уже запущен."
    if not SEARXNG_MANAGE:
        return False, (
            f"SearXNG не отвечает на {SEARXNG_URL}. "
            "Запусти его вручную или включи SEARXNG_MANAGE=1."
        )
    ok, detail = _run_compose("up", "-d")
    if not ok:
        return False, f"Не удалось запустить SearXNG: {detail}"
    deadline = time.time() + SEARXNG_START_TIMEOUT
    while time.time() < deadline:
        if searxng_ready():
            return True, "SearXNG запущен."
        time.sleep(1)
    return False, (
        f"Контейнер стартовал, но SearXNG не ответил за {int(SEARXNG_START_TIMEOUT)} с "
        f"на {SEARXNG_URL}."
    )


def stop_searxng() -> tuple[bool, str]:
    """Stop the CoreLine SearXNG container."""
    if not SEARXNG_MANAGE:
        return True, "SearXNG оставлен как есть (SEARXNG_MANAGE=0)."
    if _compose_argv("stop") is None and not searxng_ready():
        return True, "SearXNG уже выключен."
    ok, detail = _run_compose("stop")
    if not ok:
        if not searxng_ready():
            return True, "SearXNG выключен."
        return False, f"Не удалось остановить SearXNG: {detail}"
    return True, "SearXNG остановлен."
