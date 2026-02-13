from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path
from urllib import error as urllib_error

from bid_scoring.pipeline.infrastructure.mineru_adapter import (
    _upload_file,
    _upload_file_requests,
    load_content_list_from_output,
    parse_pdf_with_mineru_api,
    parse_pdf_with_mineru,
    resolve_content_list_path,
)


def test_load_content_list_from_output_reads_json_list(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = [{"type": "text", "text": "ok"}]
    (output_dir / "content_list.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_content_list_from_output(output_dir)
    assert loaded == payload


def test_load_content_list_from_output_returns_empty_when_missing(tmp_path: Path):
    loaded = load_content_list_from_output(tmp_path)
    assert loaded == []


def test_resolve_content_list_path_supports_nested_layout(tmp_path: Path):
    nested = tmp_path / "docA" / "content_list.json"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_text("[]", encoding="utf-8")
    assert resolve_content_list_path(tmp_path) == nested


def test_parse_pdf_with_mineru_uses_command_template(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%test")

    captured: dict[str, object] = {}

    def _fake_run(command, capture_output, text, timeout, check):
        _ = (capture_output, text, check)
        captured["command"] = command
        captured["timeout"] = timeout
        out_idx = command.index("-o") + 1
        output_dir = Path(command[out_idx])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "content_list.json").write_text(
            json.dumps([{"type": "text", "text": "ok"}]),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    output_dir = parse_pdf_with_mineru(
        pdf_path,
        output_root=tmp_path / "mineru-out",
        command_template="fake-mineru -p {pdf_path} -o {output_dir}",
        timeout_seconds=123,
        run_command=_fake_run,
    )

    assert output_dir.exists()
    assert resolve_content_list_path(output_dir) == output_dir / "content_list.json"
    assert captured["command"][0] == "fake-mineru"
    assert captured["timeout"] == 123


def test_parse_pdf_with_mineru_api_downloads_and_extracts_result_zip(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%test")
    output_root = tmp_path / "mineru-api-out"

    zip_path = tmp_path / "result.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "content_list.json",
            json.dumps([{"type": "text", "text": "api-result"}]),
        )

    request_calls: list[tuple[str, str]] = []

    def _fake_request_json(method, url, *, headers, payload, timeout):
        _ = (headers, payload, timeout)
        request_calls.append((method, url))
        if url.endswith("/file-urls/batch"):
            return {
                "data": {
                    "batch_id": "batch-1",
                    "file_urls": ["https://upload.example.com/one"],
                }
            }
        if "/extract-results/batch/" in url:
            return {
                "data": {
                    "extract_result": [
                        {
                            "file_name": pdf_path.name,
                            "state": "done",
                            "full_zip_url": "https://download.example.com/full.zip",
                        }
                    ]
                }
            }
        raise AssertionError(f"unexpected url: {url}")

    uploaded: dict[str, bytes] = {}

    def _fake_upload(url, file_path, *, timeout, max_retries=0):
        _ = (timeout, max_retries)
        uploaded["url"] = url.encode("utf-8")
        uploaded["data"] = file_path.read_bytes()

    def _fake_download(url, output_path, *, timeout):
        _ = (url, timeout)
        output_path.write_bytes(zip_path.read_bytes())

    output_dir = parse_pdf_with_mineru_api(
        pdf_path,
        output_root=output_root,
        api_url="https://mineru.example.com/api/v4",
        api_key="key-123",
        poll_timeout_seconds=30,
        poll_interval_seconds=0,
        request_json_fn=_fake_request_json,
        upload_file_fn=_fake_upload,
        download_file_fn=_fake_download,
    )

    loaded = load_content_list_from_output(output_dir)
    assert loaded == [{"type": "text", "text": "api-result"}]
    assert (output_dir / "content_list.json").exists()
    assert request_calls[0][0] == "POST"
    assert request_calls[1][0] == "GET"
    assert uploaded["data"].startswith(b"%PDF-1.4")


def test_upload_file_retries_on_broken_pipe(monkeypatch, tmp_path: Path):
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n%test")

    state = {"calls": 0}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            return False

    def _fake_urlopen(req, timeout):
        _ = (req, timeout)
        state["calls"] += 1
        if state["calls"] == 1:
            raise urllib_error.URLError(OSError(32, "Broken pipe"))
        return _Resp()

    monkeypatch.setattr(
        "bid_scoring.pipeline.infrastructure.mineru_adapter.urllib_request.urlopen",
        _fake_urlopen,
    )

    _upload_file("https://upload.example.com/one", file_path, timeout=1)
    assert state["calls"] == 2


def test_upload_file_requests_retries_once_then_succeeds(monkeypatch, tmp_path: Path):
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n%test")

    state = {"calls": 0}

    class _Resp:
        def raise_for_status(self):
            return None

    def _fake_put(url, data, timeout):
        _ = (url, timeout)
        data.read()
        state["calls"] += 1
        if state["calls"] == 1:
            raise OSError(32, "Broken pipe")
        return _Resp()

    monkeypatch.setattr("bid_scoring.pipeline.infrastructure.mineru_adapter.requests.put", _fake_put)

    _upload_file_requests("https://upload.example.com/one", file_path, timeout=1)
    assert state["calls"] == 2
