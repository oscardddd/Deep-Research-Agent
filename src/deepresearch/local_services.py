from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import urlsplit, urlunsplit

import httpx

from .config import Settings
from .eventlog import log_step


def _ollama_health_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    parsed = urlsplit(base_url)
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        return None
    if parsed.port not in {None, 11434}:
        return None
    return urlunsplit((parsed.scheme or "http", parsed.netloc, "/api/tags", "", ""))


def _service_is_ready(health_url: str, timeout: float = 0.5) -> bool:
    try:
        response = httpx.get(health_url, timeout=timeout)
        response.raise_for_status()
        return True
    except (httpx.HTTPError, ValueError):
        return False


def _stop_owned_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2)


@contextmanager
def managed_local_embedding_service(settings: Settings) -> Iterator[None]:
    """Run local Ollama only for the lifetime of one research run."""
    health_url = _ollama_health_url(settings.embedding_base_url)
    if (
        not settings.hybrid_retrieval_enabled
        or not settings.manage_local_embedding_service
        or health_url is None
    ):
        yield
        return
    if _service_is_ready(health_url):
        log_step("LocalService", "ollama.reused")
        yield
        return

    executable = shutil.which("ollama")
    if executable is None:
        log_step("LocalService", "ollama.unavailable", reason="binary_not_found")
        yield
        return

    settings.ensure_directories()
    log_path = settings.data_dir / "ollama.log"
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            [executable, "serve"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if _service_is_ready(health_url):
                break
            if process.poll() is not None:
                break
            time.sleep(0.1)

        if not _service_is_ready(health_url):
            _stop_owned_process(process)
            log_step(
                "LocalService",
                "ollama.unavailable",
                reason="startup_failed",
                log_path=str(log_path),
            )
            yield
            return

        owned = process.poll() is None
        log_step(
            "LocalService",
            "ollama.started",
            pid=process.pid if owned else None,
            model=settings.embedding_model,
        )
        try:
            yield
        finally:
            if owned:
                _stop_owned_process(process)
                log_step("LocalService", "ollama.stopped")
