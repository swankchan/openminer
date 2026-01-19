"""Smoke-check the SharePoint auto-ingest loop without real Graph calls.

This is a lightweight validation script that:
- patches `main.sharepoint_service` with an in-memory fake
- provides a fake processing queue so no OCR/MinerU/AI work runs
- runs the auto-ingest loop briefly for BOTH modes:
  - NONE: should not list or enqueue anything
  - NON_PROD: should list one PDF, enqueue once, and attempt a move

Run:
  conda run -n mineru2.5 python scripts/smoke_sharepoint_auto_ingest.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple


class FakeProcessingQueue:
    def __init__(self, tmp_dir: Path) -> None:
        self.enqueued: List[Tuple[str, Optional[str]]] = []
        self.tmp_dir = tmp_dir

    async def enqueue(self, label: str, websocket_id: Optional[str], fn) -> None:  # noqa: ANN001
        # Intentionally do NOT run fn(); this is a smoke check.
        self.enqueued.append((label, websocket_id))

        # Pretend the processing pipeline produced these outputs.
        csv_path = self.tmp_dir / "out.csv"
        pdf_path = self.tmp_dir / "out_searchable.pdf"
        csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
        pdf_path.write_bytes(b"%PDF-1.4\n%fake searchable\n")
        return {
            "template_csv_path": str(csv_path),
            "searchable_pdf_path": str(pdf_path),
        }


class FakeSharePointService:
    def __init__(self, tmp_dir: Path) -> None:
        self.tmp_dir = tmp_dir
        self.list_calls = 0
        self.move_calls: List[Tuple[str, str, Optional[str]]] = []
        self.ensure_calls: List[str] = []
        self.upload_calls: List[Tuple[str, str, int]] = []

    def reload_from_env(self) -> None:
        return None

    def list_pdf_files_recursive(self, folder_path: str) -> List[Dict[str, Any]]:
        self.list_calls += 1
        return [
            {
                "server_relative_url": "/Non-Production/Inbox/Beverage/B2A1-GO China Wine/x.pdf",
                "name": "x.pdf",
            }
        ]

    async def download_file(self, file_url: str, folder_path: Optional[str] = None) -> Path:
        p = self.tmp_dir / Path(file_url).name
        p.write_bytes(b"%PDF-1.4\n%fake\n")
        return p

    def move_item(self, item_server_relative: str, dest_folder_path: str, dest_name: Optional[str] = None) -> Dict[str, Any]:
        self.move_calls.append((item_server_relative, dest_folder_path, dest_name))
        return {"ok": True}

    def ensure_folder_path(self, folder_path: str) -> Dict[str, Any]:
        self.ensure_calls.append(folder_path)
        return {"ok": True}

    def upload_file(self, folder_path: str, filename: str, content: bytes) -> Dict[str, Any]:
        self.upload_calls.append((folder_path, filename, len(content or b"")))
        return {"ok": True}


async def _run_once(mode: str) -> Dict[str, Any]:
    # Ensure DEBUG is enabled in the imported module (config reads env at import time).
    os.environ["DEBUG"] = "true"

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import main  # imported here so env vars above are applied during config load

    tmp_dir = Path.cwd() / "_tmp_smoke"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # The loop reloads `.env` each cycle with override=True, which would overwrite any
    # environment variables we set in-process. Point it at a temporary `.env` instead.
    tmp_env_root = tmp_dir / "env_root"
    tmp_env_root.mkdir(parents=True, exist_ok=True)
    (tmp_env_root / ".env").write_text(
        "\n".join(
            [
                f"SP_ACTIVATE={mode}",
                "SHAREPOINT_POLL_INTERVAL=5",
                r"SP_NON_PROD_INBOX_DIR=Non-Production\\Inbox",
                r"SP_NON_PROD_PROCESSED_DIR=Non-Production\\Processed",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fake_sp = FakeSharePointService(tmp_dir)
    fake_pq = FakeProcessingQueue(tmp_dir)

    # Patch global service used by the loop.
    main.sharepoint_service = fake_sp  # type: ignore[assignment]
    main.BASE_DIR = tmp_env_root  # type: ignore[assignment]

    app = SimpleNamespace(state=SimpleNamespace(processing_queue=fake_pq))
    stop = asyncio.Event()
    wake = asyncio.Event()

    task = asyncio.create_task(main._sharepoint_auto_ingest_loop(app, stop, wake))

    # Let the loop run a little.
    await asyncio.sleep(0.5)

    # Stop cleanly (also wake it in case it's sleeping).
    stop.set()
    wake.set()
    await asyncio.wait_for(task, timeout=3)

    return {
        "mode": mode,
        "list_calls": fake_sp.list_calls,
        "enqueued": list(fake_pq.enqueued),
        "ensure_calls": list(fake_sp.ensure_calls),
        "move_calls": list(fake_sp.move_calls),
        "upload_calls": list(fake_sp.upload_calls),
    }


def main_entry() -> None:
    results = asyncio.run(_run_once("NONE"))
    print("[SMOKE] mode=NONE", results)

    results = asyncio.run(_run_once("NON_PROD"))
    print("[SMOKE] mode=NON_PROD", results)


if __name__ == "__main__":
    main_entry()
