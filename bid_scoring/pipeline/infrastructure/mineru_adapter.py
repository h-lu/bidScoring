from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import uuid
import zipfile
from pathlib import Path
from typing import Callable
from urllib import error as urllib_error
from urllib import request as urllib_request

try:  # pragma: no cover - availability is environment dependent
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore[assignment]


def load_content_list_from_output(output_dir: Path) -> list[dict]:
    """Load MinerU `content_list.json` from an output directory."""
    content_list_path = resolve_content_list_path(output_dir)
    if content_list_path is None:
        return []

    data = json.loads(content_list_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("content_list.json must be a JSON list")
    return data


def resolve_content_list_path(output_dir: Path) -> Path | None:
    """Resolve `content_list.json` path from MinerU output layout."""
    direct = output_dir / "content_list.json"
    if direct.exists():
        return direct

    nested = sorted(
        output_dir.rglob("content_list.json"),
        key=lambda item: len(str(item)),
    )
    if nested:
        return nested[0]
    return None


def parse_pdf_with_mineru(
    pdf_path: Path,
    *,
    parser_mode: str | None = None,
    output_root: Path | None = None,
    command_template: str | None = None,
    timeout_seconds: int | None = None,
    run_command: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> Path:
    """Run MinerU parse for one PDF and return output directory.

    parser_mode:
    - cli: use local `magic-pdf` / `mineru` command
    - api: use MinerU cloud API
    - auto: try cli first, then api fallback when api key exists
    """
    mode = (parser_mode or os.getenv("MINERU_PDF_PARSER", "auto")).lower()
    if mode not in {"auto", "cli", "api"}:
        raise ValueError(f"Unsupported MinerU parser mode: {mode}")

    if mode == "cli":
        return parse_pdf_with_mineru_cli(
            pdf_path,
            output_root=output_root,
            command_template=command_template,
            timeout_seconds=timeout_seconds,
            run_command=run_command,
        )

    if mode == "api":
        return parse_pdf_with_mineru_api(
            pdf_path,
            output_root=output_root,
        )

    cli_error: Exception | None = None
    try:
        return parse_pdf_with_mineru_cli(
            pdf_path,
            output_root=output_root,
            command_template=command_template,
            timeout_seconds=timeout_seconds,
            run_command=run_command,
        )
    except Exception as exc:
        cli_error = exc

    if os.getenv("MINERU_API_KEY"):
        try:
            return parse_pdf_with_mineru_api(
                pdf_path,
                output_root=output_root,
            )
        except Exception as api_exc:
            raise RuntimeError(
                f"MinerU auto parse failed. cli_error={cli_error}; api_error={api_exc}"
            ) from api_exc

    if cli_error is not None:
        raise cli_error
    raise RuntimeError("MinerU auto parse failed")


def parse_pdf_with_mineru_cli(
    pdf_path: Path,
    *,
    output_root: Path | None = None,
    command_template: str | None = None,
    timeout_seconds: int | None = None,
    run_command: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> Path:
    """Run local MinerU command for one PDF and return output directory."""
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    selected_output_root = output_root or _resolve_output_root()
    selected_output_root.mkdir(parents=True, exist_ok=True)

    output_dir = selected_output_root / f"{pdf_path.stem}-{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    timeout = timeout_seconds if timeout_seconds is not None else _resolve_timeout()
    command_candidates = _build_command_candidates(
        pdf_path=pdf_path,
        output_dir=output_dir,
        command_template=command_template or os.getenv("MINERU_PDF_COMMAND"),
    )
    if not command_candidates:
        raise RuntimeError(
            "No MinerU command candidates configured. "
            "Set MINERU_PDF_COMMAND or install magic-pdf/mineru CLI."
        )

    runner = run_command or subprocess.run
    errors: list[str] = []

    for command in command_candidates:
        try:
            result = runner(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            errors.append(f"{command[0]}: command not found")
            continue
        except subprocess.TimeoutExpired:
            errors.append(f"{' '.join(command)}: timeout({timeout}s)")
            continue

        if result.returncode != 0:
            err_msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
            errors.append(f"{' '.join(command)}: exit={result.returncode}, {err_msg}")
            continue

        if resolve_content_list_path(output_dir) is not None:
            return output_dir
        errors.append(f"{' '.join(command)}: succeeded but content_list.json not found")

    raise RuntimeError(
        "MinerU direct parse failed for "
        f"{pdf_path}. Tried commands: {' | '.join(errors)}"
    )


def _resolve_timeout() -> int:
    raw = os.getenv("MINERU_PDF_TIMEOUT_SECONDS", "1800")
    try:
        value = int(raw)
    except ValueError:
        return 1800
    return value if value > 0 else 1800


def parse_pdf_with_mineru_api(
    pdf_path: Path,
    *,
    output_root: Path | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
    request_timeout_seconds: int | None = None,
    poll_timeout_seconds: int | None = None,
    poll_interval_seconds: int | None = None,
    request_json_fn: Callable[..., dict] | None = None,
    upload_file_fn: Callable[..., None] | None = None,
    download_file_fn: Callable[..., None] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> Path:
    """Run MinerU cloud API parse for one PDF and return output directory."""
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    resolved_api_url = api_url or os.getenv("MINERU_API_URL", "https://mineru.net/api/v4")
    resolved_api_key = api_key or os.getenv("MINERU_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("MINERU_API_KEY is required for api parser mode")

    output_root_path = output_root or _resolve_output_root()
    output_root_path.mkdir(parents=True, exist_ok=True)
    output_dir = output_root_path / f"{pdf_path.stem}-{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    request_timeout = (
        request_timeout_seconds
        if request_timeout_seconds is not None
        else _resolve_int_env("MINERU_API_REQUEST_TIMEOUT_SECONDS", 30)
    )
    poll_timeout = (
        poll_timeout_seconds
        if poll_timeout_seconds is not None
        else _resolve_int_env("MINERU_API_POLL_TIMEOUT_SECONDS", 900)
    )
    poll_interval = (
        poll_interval_seconds
        if poll_interval_seconds is not None
        else _resolve_int_env("MINERU_API_POLL_INTERVAL_SECONDS", 5)
    )
    upload_max_retries = _resolve_int_env("MINERU_API_UPLOAD_MAX_RETRIES", 3)

    request_json = request_json_fn or _request_json
    if upload_file_fn is not None:
        upload_file = upload_file_fn
    else:
        upload_file = _upload_file_requests if requests is not None else _upload_file
    download_file = download_file_fn or _download_file
    sleep = sleep_fn or time.sleep

    headers = {"Authorization": f"Bearer {resolved_api_key}"}
    file_urls_resp = request_json(
        "POST",
        f"{resolved_api_url.rstrip('/')}/file-urls/batch",
        headers=headers,
        payload={"files": [{"name": pdf_path.name}]},
        timeout=request_timeout,
    )
    data = file_urls_resp.get("data") if isinstance(file_urls_resp, dict) else None
    batch_id = data.get("batch_id") if isinstance(data, dict) else None
    file_urls = data.get("file_urls") if isinstance(data, dict) else None
    if not batch_id or not isinstance(file_urls, list) or not file_urls:
        raise RuntimeError(f"Invalid MinerU file-urls response: {file_urls_resp}")

    upload_file(
        file_urls[0],
        pdf_path,
        timeout=request_timeout,
        max_retries=upload_max_retries,
    )

    deadline = time.monotonic() + poll_timeout
    full_zip_url: str | None = None
    while time.monotonic() <= deadline:
        status_resp = request_json(
            "GET",
            f"{resolved_api_url.rstrip('/')}/extract-results/batch/{batch_id}",
            headers=headers,
            payload=None,
            timeout=request_timeout,
        )
        status_data = status_resp.get("data") if isinstance(status_resp, dict) else None
        extract_result = (
            status_data.get("extract_result") if isinstance(status_data, dict) else None
        )
        selected = _select_extract_item(extract_result, pdf_path.name)
        if selected is None:
            sleep(poll_interval)
            continue

        state = selected.get("state")
        if state == "done":
            full_zip_url = selected.get("full_zip_url")
            if full_zip_url:
                break
            raise RuntimeError(
                f"MinerU API completed but missing full_zip_url: {selected}"
            )

        if state == "failed":
            err_msg = selected.get("err_msg") or "unknown error"
            raise RuntimeError(f"MinerU API extraction failed: {err_msg}")

        sleep(poll_interval)

    if not full_zip_url:
        raise TimeoutError(f"MinerU API polling timeout after {poll_timeout}s")

    zip_path = output_dir / "mineru_result.zip"
    download_file(full_zip_url, zip_path, timeout=request_timeout)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    return output_dir


def _resolve_output_root() -> Path:
    raw = os.getenv("MINERU_OUTPUT_ROOT")
    if raw:
        return Path(raw)
    return Path(".mineru-output")


def _resolve_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _build_command_candidates(
    *,
    pdf_path: Path,
    output_dir: Path,
    command_template: str | None,
) -> list[list[str]]:
    if command_template:
        rendered = command_template.format(
            pdf_path=str(pdf_path),
            output_dir=str(output_dir),
        )
        return [shlex.split(rendered)]

    return [
        ["magic-pdf", "-p", str(pdf_path), "-o", str(output_dir)],
        ["magic-pdf", "--path", str(pdf_path), "--output-dir", str(output_dir)],
        ["mineru", "-p", str(pdf_path), "-o", str(output_dir)],
        ["mineru", "--path", str(pdf_path), "--output-dir", str(output_dir)],
    ]


def _request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None,
    payload: dict | None,
    timeout: int,
) -> dict:
    body: bytes | None = None
    request_headers = dict(headers or {})
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")

    req = urllib_request.Request(
        url=url,
        data=body,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            content = resp.read()
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc

    if not content:
        return {}
    try:
        return json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}") from exc


def _upload_file(
    url: str,
    file_path: Path,
    *,
    timeout: int,
    max_retries: int = 3,
) -> None:
    data = file_path.read_bytes()
    req = urllib_request.Request(url=url, data=data, method="PUT")

    attempts = max(1, max_retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            with urllib_request.urlopen(req, timeout=timeout):
                return
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Upload failed HTTP {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:
            if attempt >= attempts:
                raise RuntimeError(f"Upload failed: {exc}") from exc
            time.sleep(min(2 ** (attempt - 1), 8))


def _upload_file_requests(
    url: str,
    file_path: Path,
    *,
    timeout: int,
    max_retries: int = 3,
) -> None:
    if requests is None:
        raise RuntimeError("requests package is required for _upload_file_requests")

    attempts = max(1, max_retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            with open(file_path, "rb") as f:
                resp = requests.put(url, data=f, timeout=timeout)
            resp.raise_for_status()
            return
        except Exception as exc:
            if attempt >= attempts:
                raise RuntimeError(f"Upload failed: {exc}") from exc
            time.sleep(min(2 ** (attempt - 1), 8))


def _download_file(url: str, output_path: Path, *, timeout: int) -> None:
    req = urllib_request.Request(url=url, method="GET")
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            output_path.write_bytes(resp.read())
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Download failed HTTP {exc.code}: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Download failed: {exc}") from exc


def _select_extract_item(extract_result: object, file_name: str) -> dict | None:
    if not isinstance(extract_result, list):
        return None
    if not extract_result:
        return None
    for item in extract_result:
        if not isinstance(item, dict):
            continue
        if item.get("file_name") == file_name:
            return item
    first = extract_result[0]
    return first if isinstance(first, dict) else None
