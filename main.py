"""
PDF OCR 與資料提取應用程式主檔案
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Set, Callable, Awaitable, Any
import uvicorn
import json
import asyncio
from dataclasses import dataclass

from services.pdf_processor import PDFProcessor
from services.mineru_service import MinerUService
from services.azure_ai_service import AzureAIService
from services.sharepoint_service import SharePointService
from services.folder_service import FolderService
from services.ocr_app import ocr_to_searchable_pdf
from config import (
    UPLOAD_DIR,
    JSON_OUTPUT_DIR,
    OUTPUT_DIR,
    CSV_OUTPUT_DIR,
    GENERATE_TEMPLATE_CSV,
    FORCE_OCR,
    DEBUG,
    JSON_GENERATION_METHOD,
    INIT_MINERU_ON_STARTUP,
    BASE_DIR,
    STATIC_DIR,
    AI_SERVICE,
    AZURE_OPENAI_TEMPERATURE,
    OLLAMA_TEMPERATURE,
    MAX_CONCURRENT_JOBS,
    SEARCHABLE_PDF_OUTPUT_DIR,
    GENERATE_SEARCHABLE_PDF,
    OCR_LANGUAGE,
    OCR_ROTATE_PAGES,
    OCR_DESKEW,
    OCR_JOBS,
    OCR_OPTIMIZE,
    OCR_FORCE_REDO,
    OCR_IMAGE_DPI,
    OCR_TESSDATA_PREFIX,
)
import re
from dotenv import load_dotenv, dotenv_values
from urllib.parse import urlparse


def _parse_sp_activate(raw: str) -> str:
    s = _strip_wrapping_quotes(raw or "").strip()
    if not s:
        return "NONE"
    low = s.lower()
    if low in ("0", "false", "no", "off", "none", "disabled", "deactivated"):
        return "NONE"
    if low in ("non_prod", "non-prod", "nonproduction", "non-production"):
        return "NON_PROD"
    if low in ("prod", "production"):
        return "PROD"
    up = s.upper()
    return up if up in ("NONE", "NON_PROD", "PROD") else "NONE"


def _normalize_sp_folder_env_path(raw: str) -> str:
    """Normalize folder paths stored in .env.

    Supports values like:
            Non-Production\\Inbox
      Non-Production/Inbox
      /Non-Production/Inbox
    Returns a Graph-usable folder path string (leading slash).
    """
    s = _strip_wrapping_quotes(raw or "").strip()
    s = s.replace("\\", "/")
    s = s.strip()
    while "//" in s:
        s = s.replace("//", "/")
    s = s.lstrip("/")
    return "/" + s if s else "/"


def _sp_processed_name(inbox_folder_norm: str, file_server_rel: str, original_name: str) -> str:
    """Generate a stable destination filename for processed items.

    If the file is in a subfolder under the inbox, encode that relative path into the name
    to reduce collisions when moving everything into one processed folder.
    """
    inbox = (inbox_folder_norm or "/").rstrip("/")
    src = (file_server_rel or "").strip()
    if not src:
        return Path(str(original_name or "")).name

    # Try to compute relative path within the inbox folder.
    rel = src
    if inbox and src.startswith(inbox + "/"):
        rel = src[len(inbox) + 1 :]
    rel = rel.lstrip("/")
    if not rel:
        return Path(str(original_name or "")).name
    return rel.replace("/", "__").replace("\\", "__")


def _sp_relative_subdir(inbox_folder_norm: str, file_server_rel: str) -> str:
    """Return the subfolder path (relative) under the inbox for a file.

    Example:
      inbox=/Non-production/Inbox
      file=/sites/X/Shared Documents/Non-production/Inbox/Beverage/A/x.pdf
      -> Beverage/A

    Returns "" when file is directly under inbox or when the inbox prefix can't be located.
    """
    inbox = (inbox_folder_norm or "").replace("\\", "/")
    file_path = (file_server_rel or "").replace("\\", "/")

    inbox = "/" + inbox.strip("/") if inbox.strip("/") else "/"
    file_path = "/" + file_path.lstrip("/") if file_path else ""

    inbox_low = inbox.lower().rstrip("/")
    file_low = file_path.lower()

    rel = ""
    prefix = inbox_low + "/"
    if file_low.startswith(prefix):
        rel = file_path[len(inbox.rstrip("/")) + 1 :]
    else:
        token = "/" + inbox.strip("/").lower().rstrip("/") + "/"
        idx = file_low.find(token)
        if idx >= 0:
            rel = file_path[idx + len(token) :]

    rel = rel.strip("/")
    if not rel:
        return ""
    parent = str(Path(rel).parent).replace("\\", "/")
    if parent in (".", ""):
        return ""
    return parent.strip("/")


async def _sharepoint_auto_ingest_loop(app: FastAPI, stop_event: asyncio.Event, wake_event: asyncio.Event) -> None:
    """Background worker: when SP_ACTIVATE is enabled, process PDFs under the configured inbox."""
    in_progress: Set[str] = set()
    env_path = BASE_DIR / ".env"

    while not stop_event.is_set():
        # Reload env each cycle so UI changes take effect without restart.
        try:
            load_dotenv(dotenv_path=str(env_path), override=True)
        except Exception:
            pass
        try:
            sharepoint_service.reload_from_env()
        except Exception:
            pass

        runtime_mode = getattr(getattr(app, "state", None), "sp_activate_runtime", None)
        if isinstance(runtime_mode, str) and runtime_mode:
            mode = _parse_sp_activate(runtime_mode)
        else:
            mode = _parse_sp_activate(os.getenv("SP_ACTIVATE") or "")

        # Poll interval (seconds)
        poll_s = 60
        poll_raw = _strip_wrapping_quotes(os.getenv("SHAREPOINT_POLL_INTERVAL") or "")
        if poll_raw:
            try:
                poll_s = max(5, int(float(str(poll_raw).split("#", 1)[0].strip())))
            except Exception:
                poll_s = 60

        if mode == "NONE":
            # Sleep (or wake early if user toggles)
            try:
                await asyncio.wait_for(wake_event.wait(), timeout=float(poll_s))
            except asyncio.TimeoutError:
                pass
            wake_event.clear()
            continue

        if mode == "NON_PROD":
            inbox_env = os.getenv("SP_NON_PROD_INBOX_DIR") or ""
            processed_env = os.getenv("SP_NON_PROD_PROCESSED_DIR") or ""
        else:
            inbox_env = os.getenv("SP_PROD_INBOX_DIR") or ""
            processed_env = os.getenv("SP_PROD_PROCESSED_DIR") or ""

        inbox_folder = _normalize_sp_folder_env_path(inbox_env)
        processed_folder = _normalize_sp_folder_env_path(processed_env)

        if DEBUG:
            print(f"[SP_AUTO] mode={mode} inbox={inbox_folder} processed={processed_folder} poll={poll_s}s")

        try:
            pdf_items = await asyncio.to_thread(
                sharepoint_service.list_pdf_files_recursive,
                inbox_folder,
            )
        except Exception as e:
            if DEBUG:
                print(f"[SP_AUTO] list failed: {e}")
            pdf_items = []

        for it in pdf_items:
            if stop_event.is_set():
                break
            server_rel = str(it.get("server_relative_url") or "").strip()
            if not server_rel:
                continue
            if server_rel in in_progress:
                continue
            in_progress.add(server_rel)

            try:
                # Download
                local_pdf = await sharepoint_service.download_file(server_rel, None)
                file_id = str(uuid.uuid4())
                base_name = sanitize_filename(local_pdf.name)

                # Enqueue processing through the same pipeline used by manual upload/sharepoint processing.
                pq: ProcessingQueue = app.state.processing_queue
                result = await pq.enqueue(
                    f"SP_AUTO:{base_name}",
                    None,
                    lambda: process_pdf_file(
                        file_path=local_pdf,
                        file_id=file_id,
                        base_name=base_name,
                        mineru_json_variant=None,
                    ),
                )

                # Move to Processed folder (best-effort)
                if processed_folder and processed_folder != "/":
                    try:
                        subdir = _sp_relative_subdir(inbox_folder, server_rel)
                        dest_folder = processed_folder.rstrip("/")
                        if subdir:
                            dest_folder = f"{dest_folder}/{subdir}"

                        await asyncio.to_thread(
                            sharepoint_service.ensure_folder_path,
                            dest_folder,
                        )

                        await asyncio.to_thread(
                            sharepoint_service.move_item,
                            server_rel,
                            dest_folder,
                        )

                        # Upload outputs (best-effort) into the same processed folder.
                        if isinstance(result, dict):
                            template_csv_path = result.get("template_csv_path")
                            searchable_pdf_path = result.get("searchable_pdf_path")

                            if template_csv_path:
                                try:
                                    csv_p = Path(str(template_csv_path))
                                    if csv_p.exists():
                                        csv_bytes = await asyncio.to_thread(csv_p.read_bytes)
                                        await asyncio.to_thread(
                                            sharepoint_service.upload_file,
                                            dest_folder,
                                            csv_p.name,
                                            csv_bytes,
                                        )
                                except Exception as ue:
                                    if DEBUG:
                                        print(f"[SP_AUTO] upload template CSV failed ({server_rel}): {ue}")

                            if searchable_pdf_path:
                                try:
                                    spdf_p = Path(str(searchable_pdf_path))
                                    if spdf_p.exists():
                                        spdf_bytes = await asyncio.to_thread(spdf_p.read_bytes)
                                        await asyncio.to_thread(
                                            sharepoint_service.upload_file,
                                            dest_folder,
                                            spdf_p.name,
                                            spdf_bytes,
                                        )
                                except Exception as ue:
                                    if DEBUG:
                                        print(f"[SP_AUTO] upload searchable PDF failed ({server_rel}): {ue}")
                    except Exception as me:
                        if DEBUG:
                            print(f"[SP_AUTO] move failed ({server_rel}): {me}")

            except Exception as e:
                if DEBUG:
                    print(f"[SP_AUTO] processing failed ({server_rel}): {e}")
            finally:
                in_progress.discard(server_rel)

        # Wait for next cycle (or wake early)
        try:
            await asyncio.wait_for(wake_event.wait(), timeout=float(poll_s))
        except asyncio.TimeoutError:
            pass
        wake_event.clear()

# 初始化服務（全域變數）
pdf_processor = PDFProcessor()
mineru_service = MinerUService()
azure_ai_service = AzureAIService()
sharepoint_service = SharePointService()
folder_service = FolderService()


def _derive_sharepoint_site_url(link: str) -> str:
    if not link:
        return ""
    try:
        parsed = urlparse(link)
        if not parsed.scheme or not parsed.netloc:
            return ""

        # Prefer keeping the site root: https://{host}/sites/{site}
        path = parsed.path or ""
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2 and parts[0].lower() == "sites":
            return f"{parsed.scheme}://{parsed.netloc}/sites/{parts[1]}"
        return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        return ""


def _write_dotenv_updates(env_path: Path, updates: Dict[str, str]) -> None:
    """Update specific keys in a .env file while preserving unrelated lines and comments."""
    env_path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines: List[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)

    keys = set(updates.keys())
    found: Set[str] = set()
    out_lines: List[str] = []

    for line in existing_lines:
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        if not m:
            out_lines.append(line)
            continue

        key = m.group(1)
        if key not in keys:
            out_lines.append(line)
            continue

        found.add(key)

        # Preserve a trailing comment when it looks like a comment separator (space + #)
        comment = ""
        comment_match = re.search(r"\s+#.*$", line)
        if comment_match:
            comment = comment_match.group(0).rstrip("\r\n")

        raw_value = updates.get(key, "")
        # Quote if value contains spaces or a '#'
        needs_quotes = (" " in raw_value) or ("\t" in raw_value) or ("#" in raw_value)
        value_str = f"\"{raw_value.replace('\\"', '\\\\"')}\"" if needs_quotes else raw_value

        newline = "\n"
        if line.endswith("\r\n"):
            newline = "\r\n"
        out_lines.append(f"{key}={value_str}{comment}{newline}")

    # Append missing keys
    for key in updates.keys():
        if key in found:
            continue
        raw_value = updates.get(key, "")
        needs_quotes = (" " in raw_value) or ("\t" in raw_value) or ("#" in raw_value)
        value_str = f"\"{raw_value.replace('\\"', '\\\\"')}\"" if needs_quotes else raw_value
        out_lines.append(f"{key}={value_str}\n")

    env_path.write_text("".join(out_lines), encoding="utf-8")


def _strip_wrapping_quotes(value: str) -> str:
    """Remove a single pair of wrapping quotes from a string value (common when users paste secrets)."""
    if value is None:
        return ""
    s = str(value).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1].strip()
    return s


def _extract_aadsts_code(message: str) -> Optional[str]:
    if not message:
        return None
    m = re.search(r"AADSTS(\d{4,})", message)
    return f"AADSTS{m.group(1)}" if m else None


def _aadsts_hint(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    hints = {
        "AADSTS7000215": "Invalid client secret. Make sure you pasted the *secret Value* (not the Secret ID), and that the secret is not expired.",
        "AADSTS7000222": "Client secret is expired. Create a new secret Value and update SHAREPOINT_SECRET.",
        "AADSTS700016": "Application (client) ID may be wrong, or the app isn't found in this tenant.",
        "AADSTS500011": "Resource principal not found. The site/resource URL might be wrong, or admin consent/app permissions are missing.",
        "AADSTS65001": "Admin consent required or missing permissions. Grant/admin-consent the required SharePoint permissions.",
    }
    return hints.get(code)


def _decode_env_newlines(value: str) -> str:
    if value is None:
        return ""
    s = str(value)
    # Tolerate legacy values that were accidentally double-escaped.
    return s.replace("\\\\n", "\n").replace("\\n", "\n")


def _encode_env_newlines(value: str) -> str:
    if value is None:
        return ""
    s = str(value).replace("\r\n", "\n")
    return s.replace("\n", "\\n")


def _read_dotenv_raw_value(env_path: Path, key: str) -> Optional[str]:
    """Best-effort: read a KEY=value from the raw .env text.

    This is intentionally more lenient than python-dotenv parsing so we can recover
    from invalid quoting (e.g., unescaped '"' inside a quoted prompt value).
    """
    try:
        if not env_path.exists():
            return None
        for line in env_path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)\s*$", line)
            if not m:
                continue
            return m.group(1)
    except Exception:
        return None
    return None


def _lenient_unquote_env_value(raw: Optional[str]) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1]
    return s

# WebSocket 連接管理器
class ProgressManager:
    """管理 WebSocket 連接和進度更新"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.progress_data: Dict[str, dict] = {}
    
    async def connect(self, websocket: WebSocket, task_id: str):
        """接受 WebSocket 連接"""
        await websocket.accept()
        self.active_connections[task_id] = websocket
        if DEBUG:
            print(f"[WEBSOCKET] 連接建立: {task_id}")
    
    def disconnect(self, task_id: str):
        """斷開 WebSocket 連接"""
        if task_id in self.active_connections:
            del self.active_connections[task_id]
        if task_id in self.progress_data:
            del self.progress_data[task_id]
        if DEBUG:
            print(f"[WEBSOCKET] 連接斷開: {task_id}")
    
    async def send_progress(self, task_id: str, percentage: int, message: str, stage: Optional[str] = None):
        """發送進度更新"""
        if task_id in self.active_connections:
            try:
                # 強制進度「只增不減」：避免心跳進度跑到前面後，
                # 真正的 MinerU 階段訊息（例如 Layout/MFD）反而把百分比拉低。
                last = self.progress_data.get(task_id, {})
                last_pct = int(last.get("percentage", -1)) if last else -1
                safe_pct = max(int(percentage), last_pct)

                progress_data = {
                    "percentage": safe_pct,
                    "message": message,
                    "stage": stage or message
                }

                # 去重：同樣內容就不重覆送，減少前端抖動/刷屏
                if last and last.get("percentage") == progress_data["percentage"] and last.get("message") == progress_data["message"] and last.get("stage") == progress_data["stage"]:
                    return
                self.progress_data[task_id] = progress_data
                await self.active_connections[task_id].send_json(progress_data)
                if DEBUG:
                    print(f"[WEBSOCKET] 進度更新 [{task_id}]: {progress_data['percentage']}% - {message}")
            except Exception as e:
                if DEBUG:
                    print(f"[WEBSOCKET] 發送進度失敗 [{task_id}]: {e}")
                self.disconnect(task_id)
    
    async def send_error(self, task_id: str, error_message: str):
        """發送錯誤訊息"""
        if task_id in self.active_connections:
            try:
                error_data = {
                    "percentage": -1,
                    "message": f"錯誤: {error_message}",
                    "stage": "error",
                    "error": True
                }
                await self.active_connections[task_id].send_json(error_data)
            except Exception as e:
                if DEBUG:
                    print(f"[WEBSOCKET] 發送錯誤失敗 [{task_id}]: {e}")
                self.disconnect(task_id)


@dataclass
class QueuedJob:
    """排隊中的工作（逐個處理）"""
    description: str
    task_id: Optional[str]
    position: int
    run: Callable[[], Awaitable[Any]]
    future: "asyncio.Future[Any]"


class ProcessingQueue:
    """單機排隊器：用 asyncio.Queue 控制同一時間最多 N 個工作在跑（預設 1）"""

    def __init__(self, progress: ProgressManager, concurrency: int = 1):
        self._progress = progress
        self._queue: "asyncio.Queue[QueuedJob]" = asyncio.Queue()
        self._workers: List[asyncio.Task] = []
        self._concurrency = max(1, int(concurrency))

    async def start(self):
        if self._workers:
            return
        for i in range(self._concurrency):
            self._workers.append(asyncio.create_task(self._worker_loop(i)))

    async def stop(self):
        for t in self._workers:
            t.cancel()
        for t in self._workers:
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._workers = []

    def _queue_position(self) -> int:
        # position 是「入隊時」快照（不做動態更新）
        return int(self._queue.qsize()) + 1

    async def enqueue(self, description: str, task_id: Optional[str], run: Callable[[], Awaitable[Any]]) -> Any:
        loop = asyncio.get_running_loop()
        fut: "asyncio.Future[Any]" = loop.create_future()
        pos = self._queue_position()
        job = QueuedJob(description=description, task_id=task_id, position=pos, run=run, future=fut)

        # 先告知前端：已入隊
        if task_id:
            if pos <= 1:
                await self._progress.send_progress(task_id, 3, "排隊中…", "queued")
            else:
                await self._progress.send_progress(task_id, 3, f"排隊中…（第 {pos} 位）", "queued")

        await self._queue.put(job)
        return await fut

    async def _worker_loop(self, worker_idx: int):
        while True:
            job = await self._queue.get()
            try:
                if job.task_id:
                    await self._progress.send_progress(job.task_id, 4, "輪到你了，準備開始…", "dequeue")
                result = await job.run()
                if not job.future.done():
                    job.future.set_result(result)
            except Exception as e:
                if job.task_id:
                    await self._progress.send_error(job.task_id, str(e))
                if not job.future.done():
                    job.future.set_exception(e)
            finally:
                self._queue.task_done()


def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除不安全的字符和 UUID 前綴
    
    處理以下情況：
    1. 移除 UUID 前綴（例如：35d4eae2-5958-4aef-9f59-8cc8342c003d_YYC-2 → YYC-2）
    2. 如果整個文件名是 UUID，使用默認名稱
    3. 移除不安全的字符
    """
    # 移除擴展名
    name = Path(filename).stem
    
    # UUID 格式：8-4-4-4-12 或 32 個十六進制字符
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    uuid_pattern_no_dash = r'^[0-9a-f]{32}'
    
    # 檢查並移除 UUID 前綴（帶下劃線或連字符分隔）
    # 例如：35d4eae2-5958-4aef-9f59-8cc8342c003d_YYC-2 → YYC-2
    if re.match(uuid_pattern, name, re.IGNORECASE):
        # 移除 UUID 前綴（包括後面的分隔符）
        name = re.sub(rf'^{uuid_pattern}[_-]?', '', name, flags=re.IGNORECASE)
    elif re.match(uuid_pattern_no_dash, name, re.IGNORECASE):
        # 移除 32 字符的 UUID 前綴
        name = re.sub(rf'^{uuid_pattern_no_dash}[_-]?', '', name, flags=re.IGNORECASE)
    
    # 檢查是否整個文件名都是 UUID（移除前綴後為空）
    if not name or name.strip() == '':
        # 如果整個文件名都是 UUID，嘗試從原始文件名中提取有意義的部分
        # 例如：如果原始文件名是 "document.pdf"，使用 "document"
        original_stem = Path(filename).stem
        # 嘗試找到第一個非 UUID 部分
        parts = re.split(r'[_-]', original_stem)
        for part in parts:
            # 跳過 UUID 格式的部分
            if not re.match(rf'^{uuid_pattern}$|^{uuid_pattern_no_dash}$', part, re.IGNORECASE):
                if part and len(part) > 1:  # 至少 2 個字符
                    name = part
                    break
        
        # 如果還是找不到有意義的部分，使用默認名稱
        if not name or name.strip() == '':
            name = "document"
    
    # 移除或替換不安全的字符（Windows 不允許的字符）
    # 不允許: < > : " / \ | ? *
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # 移除多個連續的下劃線
    name = re.sub(r'_+', '_', name)
    # 移除開頭和結尾的點、空格和下劃線
    name = name.strip('. _')
    
    # 如果為空，使用默認名稱
    if not name:
        name = "document"
    
    # 限制長度（Windows 路徑限制，保留一些餘量）
    if len(name) > 200:
        name = name[:200]
    
    return name


def get_unique_csv_path(base_name: str, output_dir: Path) -> Path:
    """獲取唯一的 CSV 文件路徑（永遠加 timestamp，避免覆蓋且方便追蹤批次）。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_dir / f"{base_name}_{ts}.csv"
    counter = 1
    while candidate.exists():
        candidate = output_dir / f"{base_name}_{ts}_{counter}.csv"
        counter += 1
    return candidate


def get_unique_json_path(base_name: str, output_dir: Path) -> Path:
    """獲取唯一的 JSON 文件路徑（永遠加 timestamp，避免覆蓋且方便追蹤批次）。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_dir / f"{base_name}_{ts}.json"
    counter = 1
    while candidate.exists():
        candidate = output_dir / f"{base_name}_{ts}_{counter}.json"
        counter += 1
    return candidate


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理（啟動和關閉）"""
    # 啟動排隊器（預設單一併發：逐個處理）
    app.state.processing_queue = ProcessingQueue(progress_manager, concurrency=MAX_CONCURRENT_JOBS)
    await app.state.processing_queue.start()

    # 啟動時初始化 MinerU（如果配置啟用）
    if INIT_MINERU_ON_STARTUP:
        if DEBUG:
            print("[APP] 正在初始化 MinerU 服務...")
        try:
            # 不要阻塞 FastAPI 啟動：vLLM 初始化在某些環境可能非常久，
            # 會導致前端頁面/WS 端點都無法連線。
            asyncio.create_task(mineru_service._initialize())
            if DEBUG:
                print("[APP] MinerU 初始化已在背景啟動（不阻塞啟動）")
        except Exception as e:
            if DEBUG:
                print(f"[APP] MinerU 初始化警告: {e}")
    else:
        if DEBUG:
            print("[APP] MinerU 將在第一次使用時才初始化（INIT_MINERU_ON_STARTUP=False）")

    # SharePoint auto-ingest worker (optional; controlled by SP_ACTIVATE)
    app.state.sp_auto_stop = asyncio.Event()
    app.state.sp_auto_wake = asyncio.Event()
    app.state.sp_auto_task = asyncio.create_task(
        _sharepoint_auto_ingest_loop(app, app.state.sp_auto_stop, app.state.sp_auto_wake)
    )
    
    yield
    
    # 關閉時清理資源
    try:
        await app.state.processing_queue.stop()
    except Exception:
        pass

    # Stop SharePoint auto-ingest worker
    try:
        app.state.sp_auto_stop.set()
        app.state.sp_auto_wake.set()
        await asyncio.wait_for(app.state.sp_auto_task, timeout=5)
    except Exception:
        pass

    if DEBUG:
        print("[APP] 正在關閉 MinerU 服務...")
    try:
        await mineru_service.shutdown()
    except Exception as e:
        if DEBUG:
            print(f"[APP] MinerU 關閉警告: {e}")


app = FastAPI(
    title="PDF OCR 與資料提取應用程式",
    lifespan=lifespan
)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 掛載靜態文件目錄（用於 favicon 等）
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 全局進度管理器
progress_manager = ProgressManager()


@app.websocket("/ws/progress/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    """WebSocket 端點用於實時進度更新"""
    await progress_manager.connect(websocket, task_id)
    try:
        while True:
            # 保持連接活躍，等待客戶端斷開
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        progress_manager.disconnect(task_id)
    except Exception as e:
        if DEBUG:
            print(f"[WEBSOCKET] 連接錯誤 [{task_id}]: {e}")
        progress_manager.disconnect(task_id)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """主頁面"""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(
            content=f.read(),
            headers={
                # index.html 內嵌大量 CSS/JS，開發/迭代時容易被瀏覽器快取而「睇唔到改動」
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
            },
        )


@app.get("/api/settings/sharepoint")
async def get_sharepoint_settings():
    """Return SharePoint settings for the UI. Secret is never returned."""
    env_path = BASE_DIR / ".env"

    # Prefer .env values (if present) so UI reflects what's saved.
    values: Dict[str, str] = dict(os.environ)
    if env_path.exists():
        try:
            for k, v in dotenv_values(env_path).items():
                if v is not None:
                    values[k] = v
        except Exception:
            pass

    client_id = _strip_wrapping_quotes(values.get("SHAREPOINT_CLIENT_ID") or "")
    tenant = _strip_wrapping_quotes(values.get("SHAREPOINT_TENANT") or values.get("SHAREPOINT_TENANT_ID") or "")
    link = _strip_wrapping_quotes(values.get("SHAREPOINT_LINK") or values.get("SHAREPOINT_SITE_URL") or "")
    folder = _strip_wrapping_quotes(values.get("SHAREPOINT_FOLDER") or "")

    sp_activate = _parse_sp_activate(_strip_wrapping_quotes(values.get("SP_ACTIVATE") or ""))
    sp_non_prod_inbox = _strip_wrapping_quotes(values.get("SP_NON_PROD_INBOX_DIR") or "")
    sp_non_prod_processed = _strip_wrapping_quotes(values.get("SP_NON_PROD_PROCESSED_DIR") or "")
    sp_prod_inbox = _strip_wrapping_quotes(values.get("SP_PROD_INBOX_DIR") or "")
    sp_prod_processed = _strip_wrapping_quotes(values.get("SP_PROD_PROCESSED_DIR") or "")

    poll_raw = _strip_wrapping_quotes(values.get("SHAREPOINT_POLL_INTERVAL") or "")
    poll_interval: Optional[int] = None
    if poll_raw:
        try:
            poll_interval = int(float(str(poll_raw).split("#", 1)[0].strip()))
        except Exception:
            poll_interval = None

    secret_set = bool(
        _strip_wrapping_quotes(values.get("SHAREPOINT_SECRET") or values.get("SHAREPOINT_CLIENT_SECRET") or "")
    )

    return {
        "client_id": client_id,
        "tenant": tenant,
        "link": link,
        "folder": folder,
        "poll_interval": poll_interval,
        "secret_set": secret_set,
        "sp_activate": sp_activate,
        "sp_non_prod_inbox_dir": sp_non_prod_inbox,
        "sp_non_prod_processed_dir": sp_non_prod_processed,
        "sp_prod_inbox_dir": sp_prod_inbox,
        "sp_prod_processed_dir": sp_prod_processed,
    }


@app.get("/api/settings/sharepoint/activate")
async def get_sharepoint_activate():
    runtime_mode = getattr(getattr(app, "state", None), "sp_activate_runtime", None)
    if isinstance(runtime_mode, str) and runtime_mode:
        return {"mode": _parse_sp_activate(runtime_mode), "source": "runtime"}

    env_path = BASE_DIR / ".env"
    values = dict(dotenv_values(env_path)) if env_path.exists() else dict(os.environ)
    mode = _parse_sp_activate(_strip_wrapping_quotes(values.get("SP_ACTIVATE") or ""))
    return {"mode": mode, "source": "env"}


@app.post("/api/settings/sharepoint/activate")
async def set_sharepoint_activate(payload: Dict[str, Any] = Body(...)):
    raw = payload.get("mode")
    mode = _parse_sp_activate(str(raw) if raw is not None else "")

    # Runtime-only toggle: do NOT persist SP_ACTIVATE to .env.
    # This ensures SP_ACTIVATE stays e.g. NONE in .env even if the UI activates.
    try:
        if mode == "NONE":
            app.state.sp_activate_runtime = None
        else:
            app.state.sp_activate_runtime = mode
    except Exception:
        pass

    # Wake the background worker so changes take effect immediately.
    try:
        app.state.sp_auto_wake.set()
    except Exception:
        pass

    return {"status": "ok", "mode": mode, "source": "runtime"}


@app.post("/api/settings/sharepoint")
async def set_sharepoint_settings(payload: Dict[str, Any] = Body(...)):
    """Update SharePoint settings in .env. Secret is write-only."""
    env_path = BASE_DIR / ".env"

    client_id = _strip_wrapping_quotes(payload.get("client_id") or "")
    tenant = _strip_wrapping_quotes(payload.get("tenant") or "")
    secret = _strip_wrapping_quotes(payload.get("secret") or "")
    link = _strip_wrapping_quotes(payload.get("link") or "")
    folder = _strip_wrapping_quotes(payload.get("folder") or "")

    poll_interval_raw = payload.get("poll_interval")
    poll_interval_str = _strip_wrapping_quotes(str(poll_interval_raw)) if poll_interval_raw is not None else ""
    poll_interval = ""
    if poll_interval_str != "":
        try:
            poll_interval = str(max(1, int(float(poll_interval_str))))
        except Exception:
            raise HTTPException(status_code=400, detail="poll_interval must be an integer >= 1")

    updates: Dict[str, str] = {
        # User-provided 6 keys
        "SHAREPOINT_CLIENT_ID": client_id,
        "SHAREPOINT_TENANT": tenant,
        "SHAREPOINT_LINK": link,
        "SHAREPOINT_FOLDER": folder,
    }

    # Note: SP_ACTIVATE is intentionally NOT persisted here; activation is runtime-only.
    if "sp_non_prod_inbox_dir" in payload:
        updates["SP_NON_PROD_INBOX_DIR"] = _strip_wrapping_quotes(str(payload.get("sp_non_prod_inbox_dir") or ""))
    if "sp_non_prod_processed_dir" in payload:
        updates["SP_NON_PROD_PROCESSED_DIR"] = _strip_wrapping_quotes(str(payload.get("sp_non_prod_processed_dir") or ""))
    if "sp_prod_inbox_dir" in payload:
        updates["SP_PROD_INBOX_DIR"] = _strip_wrapping_quotes(str(payload.get("sp_prod_inbox_dir") or ""))
    if "sp_prod_processed_dir" in payload:
        updates["SP_PROD_PROCESSED_DIR"] = _strip_wrapping_quotes(str(payload.get("sp_prod_processed_dir") or ""))
    if poll_interval != "":
        updates["SHAREPOINT_POLL_INTERVAL"] = poll_interval

    # Only overwrite secret if user provided a non-empty value
    if secret != "":
        updates["SHAREPOINT_SECRET"] = secret
        # Also keep legacy key in sync for code that expects it
        updates["SHAREPOINT_CLIENT_SECRET"] = secret

    # Keep legacy keys in sync for existing SharePoint client code
    if tenant:
        updates["SHAREPOINT_TENANT_ID"] = tenant

    derived_site_url = _derive_sharepoint_site_url(link)
    if derived_site_url:
        updates["SHAREPOINT_SITE_URL"] = derived_site_url

    _write_dotenv_updates(env_path, updates)

    # Reload process environment so changes take effect without restart
    load_dotenv(dotenv_path=str(env_path), override=True)
    try:
        sharepoint_service.reload_from_env()
    except Exception:
        # If reload fails, the settings are still saved; user can restart the app.
        pass

    # Wake the background worker in case poll interval / activation changed.
    try:
        app.state.sp_auto_wake.set()
    except Exception:
        pass

    # Never return secret
    current = dict(dotenv_values(env_path)) if env_path.exists() else {}
    secret_set = bool(
        _strip_wrapping_quotes(current.get("SHAREPOINT_SECRET") or current.get("SHAREPOINT_CLIENT_SECRET") or "")
    )
    return {"status": "ok", "secret_set": secret_set}


@app.get("/api/settings/prompts")
async def get_prompt_settings():
    """Return the current default prompts (from .env when present)."""
    env_path = BASE_DIR / ".env"
    values = dict(dotenv_values(env_path)) if env_path.exists() else {}

    system_prompt_raw = _strip_wrapping_quotes(values.get("AZURE_OPENAI_SYSTEM_PROMPT") or "")
    user_prompt_raw = _strip_wrapping_quotes(values.get("AZURE_OPENAI_USER_PROMPT") or "")

    # If dotenv parsing yields empty, fall back to raw-line parsing (handles invalid quoting).
    if not system_prompt_raw:
        system_prompt_raw = _lenient_unquote_env_value(_read_dotenv_raw_value(env_path, "AZURE_OPENAI_SYSTEM_PROMPT"))
    if not user_prompt_raw:
        user_prompt_raw = _lenient_unquote_env_value(_read_dotenv_raw_value(env_path, "AZURE_OPENAI_USER_PROMPT"))

    # Handle common escaping patterns when reading raw lines.
    system_prompt_raw = system_prompt_raw.replace("\\\"", '"').replace("\\\\", "\\")
    user_prompt_raw = user_prompt_raw.replace("\\\"", '"').replace("\\\\", "\\")
    return {
        "system_prompt": _decode_env_newlines(system_prompt_raw),
        "user_prompt": _decode_env_newlines(user_prompt_raw),
    }


@app.post("/api/settings/prompts")
async def set_prompt_settings(payload: Dict[str, Any] = Body(...)):
    """Update default prompts in .env and reload env for immediate effect."""
    env_path = BASE_DIR / ".env"

    system_prompt = payload.get("system_prompt")
    user_prompt = payload.get("user_prompt")

    if system_prompt is None:
        system_prompt = ""
    if user_prompt is None:
        user_prompt = ""

    if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
        raise HTTPException(status_code=400, detail="system_prompt and user_prompt must be strings")

    # Keep .env single-line: store newlines as literal \\n sequences.
    system_prompt_enc = _encode_env_newlines(system_prompt)
    user_prompt_enc = _encode_env_newlines(user_prompt)

    max_len = 50000
    if len(system_prompt_enc) > max_len or len(user_prompt_enc) > max_len:
        raise HTTPException(status_code=400, detail=f"Prompts are too long (max {max_len} chars)")

    updates: Dict[str, str] = {
        "AZURE_OPENAI_SYSTEM_PROMPT": system_prompt_enc,
        "AZURE_OPENAI_USER_PROMPT": user_prompt_enc,
    }

    _write_dotenv_updates(env_path, updates)
    load_dotenv(dotenv_path=str(env_path), override=True)

    return {"status": "ok"}


@app.get("/api/runtime_status")
async def runtime_status():
    """提供前端顯示用的「實際運行狀態」：MinerU / CSV / AI（不包含敏感資訊）"""
    mineru = mineru_service.get_runtime_status()

    ai_client_ready = bool(azure_ai_service.client)
    ai_service = azure_ai_service.ai_service or AI_SERVICE

    # 提取方式：若選了 azure_openai_json 但 AI client 未初始化，實際會回退到 mineru_json
    configured_csv = JSON_GENERATION_METHOD
    effective_csv = configured_csv
    csv_fallback = None
    if configured_csv == "azure_openai_csv" and not ai_client_ready:
        effective_csv = "mineru_csv"
        csv_fallback = "azure_openai_csv -> mineru_csv（AI 未初始化）"

    return {
        "mineru": mineru,
        "csv": {
            "configured_method": configured_csv,
            "effective_method": effective_csv,
            "fallback": csv_fallback,
        },
        "ai": {
            "service": ai_service,
            "ready": ai_client_ready,
            "azure_deployment": getattr(azure_ai_service, "deployment_name", None) if ai_service != "ollama" else None,
            "ollama_model": getattr(azure_ai_service, "ollama_model", None) if ai_service == "ollama" else None,
            "temperature": (
                float(getattr(azure_ai_service, "ollama_temperature", OLLAMA_TEMPERATURE))
                if ai_service == "ollama"
                else float(getattr(azure_ai_service, "azure_openai_temperature", AZURE_OPENAI_TEMPERATURE))
            ),
        },
    }


@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    task_id: Optional[str] = Form(None),
    output_format: Optional[str] = Form(None),
    mineru_json_variant: Optional[str] = Form(None),
    system_prompt: Optional[str] = Form(None),
    user_prompt: Optional[str] = Form(None),
):
    """上傳並處理 PDF 或 JSON 檔案
    
    Args:
        file: 上傳的 PDF 或 JSON 檔案
        會固定產生 JSON 輸出；是否額外產生 template CSV 由 .env 的 GENERATE_TEMPLATE_CSV 控制。
    """
    try:
        import json
        
        if DEBUG:
            print(f"[UPLOAD] 收到上傳請求: {file.filename}")
            print(f"[UPLOAD] task_id: {task_id}")
        
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="Missing upload file")

        # 生成基於原始文件名的安全名稱
        base_name = sanitize_filename(file.filename)
        
        # 儲存上傳的檔案（仍使用 UUID 前綴以避免衝突）
        file_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 檢查文件類型
        file_extension = Path(file.filename).suffix.lower()
        
        # 支援 .json, .pdf
        if file_extension == ".json":
            # 如果是 JSON 文件，直接讀取並跳過 MinerU 處理
            if DEBUG:
                print(f"[JSON_UPLOAD] 檢測到 JSON 文件，跳過 MinerU 處理: {file.filename}")
            
            try:
                async def run_json_job():
                    if task_id:
                        await progress_manager.send_progress(task_id, 6, "讀取 JSON…", "json_read")
                    with open(file_path, "r", encoding="utf-8") as f:
                        mineru_json = json.load(f)

                    if DEBUG:
                        print(f"[JSON_UPLOAD] JSON 文件讀取成功，直接交給 AI/程序生成 JSON")

                    # 輸出固定為 JSON
                    final_output_format = "json"

                    if True:
                        async def csv_progress_callback(p, msg, stage):
                            if task_id:
                                await progress_manager.send_progress(task_id, 20 + int(p * 0.75), msg, stage)
                        if task_id:
                            await progress_manager.send_progress(task_id, 20, "生成 JSON 中…", "csv_start")
                        if DEBUG:
                            print(f"[JSON_GENERATION] 開始生成 JSON（從 JSON）: {base_name}")
                        out_path = get_unique_json_path(base_name, JSON_OUTPUT_DIR)
                        output_path_local = await azure_ai_service.generate_csv_from_json(
                            mineru_json,
                            out_path,
                            progress_callback=csv_progress_callback if task_id else None,
                            output_format=final_output_format,
                            system_prompt_override=system_prompt,
                            user_prompt_override=user_prompt,
                        )
                        if task_id:
                            await progress_manager.send_progress(task_id, 100, "完成", "complete")
                        return {"output_path": str(output_path_local)}
                    

                # 入隊：逐個處理（避免多人同時跑 AI/MinerU）
                pq: ProcessingQueue = app.state.processing_queue
                result = await pq.enqueue(f"JSON:{base_name}", task_id, run_json_job)
                output_path = Path(result["output_path"]) if result.get("output_path") else None
                
                response_data = {
                    "status": "success",
                    "file_id": file_id,
                    "filename": base_name,
                    "message": "JSON 處理完成",
                    "file_type": "json",
                    "needs_ocr": False,  # JSON 文件不需要 OCR
                    "json_available": True,
                    "generate_template_csv": bool(GENERATE_TEMPLATE_CSV)
                }

                # JSON output is always produced; provide a JSON download link.
                if output_path:
                    response_data["output_path"] = str(output_path)
                    response_data["json_filename"] = output_path.name
                    response_data["download_url"] = f"/api/download/{output_path.stem}"

                return JSONResponse(content=response_data)
                
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"JSON 文件格式錯誤: {str(e)}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"處理 JSON 文件時發生錯誤: {str(e)}")
        
        elif file_extension == ".pdf":
            # 如果是 PDF 文件，使用現有流程
            if DEBUG:
                print(f"[PDF_UPLOAD] 檢測到 PDF 文件，使用標準處理流程: {file.filename}")
            
            # 使用提供的 task_id 或生成新的
            if not task_id:
                task_id = str(uuid.uuid4())
                if DEBUG:
                    print(f"[PDF_UPLOAD] 未提供 task_id，生成新的: {task_id}")
            else:
                if DEBUG:
                    print(f"[PDF_UPLOAD] 使用提供的 task_id: {task_id}")
            
            pq: ProcessingQueue = app.state.processing_queue
            result = await pq.enqueue(
                f"PDF:{base_name}",
                task_id,
                lambda: process_pdf_file(
                    file_path=file_path,
                    file_id=file_id,
                    base_name=base_name,
                    task_id=task_id,
                    output_format=output_format,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    mineru_json_variant=mineru_json_variant,
                ),
            )
            
            response_data = {
                "status": "success",
                "file_id": file_id,
                "filename": base_name,
                "message": "PDF 處理完成",
                "file_type": "pdf",
                "needs_ocr": result.get("needs_ocr"),
                "json_available": result.get("json_available", False),
                "generate_template_csv": bool(GENERATE_TEMPLATE_CSV)
            }

            # If a searchable PDF was generated, expose a direct download URL for it.
            if result.get("searchable_pdf_path"):
                try:
                    sp = Path(str(result.get("searchable_pdf_path")))
                    response_data["searchable_pdf_path"] = str(sp)
                    response_data["searchable_pdf_filename"] = sp.name
                    response_data["searchable_pdf_download_url"] = f"/api/download/{sp.name}"
                except Exception:
                    pass
            
            out_path = result.get("output_path")
            if out_path:
                out_obj = Path(out_path)
                response_data["output_path"] = str(out_obj)
                response_data["json_filename"] = out_obj.name
                response_data["download_url"] = f"/api/download/{out_obj.stem}"

            if result.get("template_csv_path"):
                tpath = Path(str(result.get("template_csv_path")))
                response_data["template_csv_path"] = str(tpath)
                response_data["template_csv_filename"] = tpath.name
                response_data["template_csv_download_url"] = f"/api/download/{tpath.stem}"
            
            return JSONResponse(content=response_data)
        
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"不支援的文件類型: {file_extension}。請上傳 PDF 或 JSON 文件。"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理錯誤: {str(e)}")


@app.post("/api/process-sharepoint")
async def process_sharepoint(
    file_url: str = Form(...),
    folder_path: Optional[str] = Form(None),
    mineru_json_variant: Optional[str] = Form(None),
):
    """從 SharePoint 處理 PDF
    
    Args:
        file_url: SharePoint 檔案 URL 或相對路徑
        folder_path: SharePoint 資料夾路徑（可選）
        是否額外產生 template CSV 由 .env 的 GENERATE_TEMPLATE_CSV 控制。
    """
    try:
        # Reload SharePoint settings so updated .env values take effect
        env_path = BASE_DIR / ".env"
        try:
            load_dotenv(dotenv_path=str(env_path), override=True)
        except Exception:
            pass
        sharepoint_service.reload_from_env()

        # 從 SharePoint 下載檔案
        file_path = await sharepoint_service.download_file(file_url, folder_path)
        
        file_id = str(uuid.uuid4())
        base_name = sanitize_filename(file_path.name)
        
        # 入隊：逐個處理
        pq: ProcessingQueue = app.state.processing_queue
        result = await pq.enqueue(
            f"SharePoint:{base_name}",
            None,
            lambda: process_pdf_file(
                file_path=file_path,
                file_id=file_id,
                base_name=base_name,
                mineru_json_variant=mineru_json_variant,
            ),
        )
        
        response_data = {
            "status": "success",
            "file_id": file_id,
            "filename": base_name,
            "message": "SharePoint PDF 處理完成",
            "needs_ocr": result.get("needs_ocr"),
            "json_available": result.get("json_available", False),
            "generate_template_csv": bool(GENERATE_TEMPLATE_CSV)
        }

        if result.get("searchable_pdf_path"):
            try:
                sp = Path(str(result.get("searchable_pdf_path")))
                response_data["searchable_pdf_path"] = str(sp)
                response_data["searchable_pdf_filename"] = sp.name
                response_data["searchable_pdf_download_url"] = f"/api/download/{sp.name}"
            except Exception:
                pass
        
        out_path = result.get("output_path")
        if out_path:
            out_obj = Path(out_path)
            response_data["output_path"] = str(out_obj)
            response_data["json_filename"] = out_obj.name
            response_data["download_url"] = f"/api/download/{out_obj.stem}"

        if result.get("template_csv_path"):
            tpath = Path(str(result.get("template_csv_path")))
            response_data["template_csv_path"] = str(tpath)
            response_data["template_csv_filename"] = tpath.name
            response_data["template_csv_download_url"] = f"/api/download/{tpath.stem}"
        
        return JSONResponse(content=response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理錯誤: {str(e)}")


@app.get("/api/sharepoint/browse")
async def browse_sharepoint(folder_path: Optional[str] = None):
    """Browse a SharePoint folder (subfolders + files) for the frontend folder picker."""
    try:
        # Reload SharePoint settings so updated .env values take effect
        env_path = BASE_DIR / ".env"
        try:
            load_dotenv(dotenv_path=str(env_path), override=True)
        except Exception:
            pass
        sharepoint_service.reload_from_env()

        # Default to configured folder; fallback to root
        default_folder = os.getenv("SHAREPOINT_FOLDER") or "/"
        path = (folder_path or "").strip() or default_folder.strip() or "/"

        # Graph HTTP calls are synchronous; run in a thread to avoid blocking the event loop.
        data = await asyncio.to_thread(sharepoint_service.browse_folder, path)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sharepoint/upload")
async def upload_sharepoint_file(
    folder_path: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """Upload a file to the specified SharePoint folder (defaults to SHAREPOINT_FOLDER)."""
    try:
        env_path = BASE_DIR / ".env"
        try:
            load_dotenv(dotenv_path=str(env_path), override=True)
        except Exception:
            pass
        sharepoint_service.reload_from_env()

        default_folder = os.getenv("SHAREPOINT_FOLDER") or "/"
        target_folder = (folder_path or "").strip() or default_folder.strip() or "/"

        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="Missing upload file")

        content = await file.read()
        result = await asyncio.to_thread(sharepoint_service.upload_file, target_folder, file.filename, content)
        return JSONResponse(content={"ok": True, "folder": target_folder, "result": result})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sharepoint/delete")
async def delete_sharepoint_item(payload: Dict[str, Any] = Body(...)):
    """Delete a SharePoint file/folder by server-relative path."""
    try:
        env_path = BASE_DIR / ".env"
        try:
            load_dotenv(dotenv_path=str(env_path), override=True)
        except Exception:
            pass
        sharepoint_service.reload_from_env()

        drive_id = str((payload or {}).get("drive_id") or "").strip()
        item_id = str((payload or {}).get("item_id") or "").strip()
        path = str((payload or {}).get("path") or "").strip()

        if drive_id and item_id:
            result = await asyncio.to_thread(sharepoint_service.delete_item_by_id, drive_id, item_id)
        else:
            if not path:
                raise HTTPException(status_code=400, detail="Missing 'path' (or provide drive_id + item_id)")
            result = await asyncio.to_thread(sharepoint_service.delete_item, path)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sharepoint/tree")
async def sharepoint_tree(
    folder_path: Optional[str] = None,
    depth: int = 4,
    include_files: bool = False,
    max_nodes: int = 1500,
):
    """Return a depth-limited SharePoint folder tree for frontend display."""
    try:
        # Reload SharePoint settings so updated .env values take effect
        env_path = BASE_DIR / ".env"
        try:
            load_dotenv(dotenv_path=str(env_path), override=True)
        except Exception:
            pass
        sharepoint_service.reload_from_env()

        default_folder = os.getenv("SHAREPOINT_FOLDER") or "/"
        path = (folder_path or "").strip() or default_folder.strip() or "/"

        data = await asyncio.to_thread(
            sharepoint_service.get_folder_tree,
            path,
            depth=int(depth),
            include_files=bool(include_files),
            max_nodes=int(max_nodes),
        )
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sharepoint/validate")
async def validate_sharepoint(folder_path: Optional[str] = None):
    """Validate SharePoint credentials + basic access and return a UI-friendly diagnosis."""
    try:
        env_path = BASE_DIR / ".env"
        try:
            load_dotenv(dotenv_path=str(env_path), override=True)
        except Exception:
            pass
        sharepoint_service.reload_from_env()
        result = await asyncio.to_thread(sharepoint_service.validate_connection, folder_path)

        if not isinstance(result, dict):
            return JSONResponse(content={"ok": False, "error": {"message": "Unexpected validation result"}})

        if not result.get("ok"):
            msg = str((result.get("error") or {}).get("message") or "")
            code = _extract_aadsts_code(msg)
            hint = _aadsts_hint(code)
            if hint:
                result.setdefault("error", {})
                result["error"]["aadsts_code"] = code
                result["error"]["hint"] = hint
            elif code:
                result.setdefault("error", {})
                result["error"]["aadsts_code"] = code
            return JSONResponse(content=result)

        return JSONResponse(content=result)
    except Exception as e:
        msg = str(e)
        code = _extract_aadsts_code(msg)
        payload = {"ok": False, "error": {"message": msg}}
        if code:
            payload["error"]["aadsts_code"] = code
            hint = _aadsts_hint(code)
            if hint:
                payload["error"]["hint"] = hint
        return JSONResponse(content=payload)


@app.post("/api/process-folder")
async def process_folder(
    folder_path: str = Form(...),
    mineru_json_variant: Optional[str] = Form(None),
):
    """從 Windows 資料夾處理 PDF
    
    Args:
        folder_path: Windows 資料夾路徑
        是否額外產生 template CSV 由 .env 的 GENERATE_TEMPLATE_CSV 控制。
    """
    try:
        # 從資料夾讀取所有 PDF
        pdf_files = folder_service.get_pdf_files(folder_path)
        
        async def run_folder_job():
            results_local = []
            for pdf_path in pdf_files:
                file_id = str(uuid.uuid4())
                base_name = sanitize_filename(Path(pdf_path).name)
                result = await process_pdf_file(
                    file_path=pdf_path,
                    file_id=file_id,
                    base_name=base_name,
                    mineru_json_variant=mineru_json_variant,
                )
                result_item = {
                    "file_id": file_id,
                    "filename": base_name,
                    "original_filename": Path(pdf_path).name,
                    "needs_ocr": result.get("needs_ocr"),
                    "json_available": result.get("json_available", False),
                    "generate_template_csv": bool(GENERATE_TEMPLATE_CSV)
                }
                if result.get("searchable_pdf_path"):
                    try:
                        sp = Path(str(result.get("searchable_pdf_path")))
                        result_item["searchable_pdf_path"] = str(sp)
                        result_item["searchable_pdf_filename"] = sp.name
                        result_item["searchable_pdf_download_url"] = f"/api/download/{sp.name}"
                    except Exception:
                        pass
                out_path = result.get("output_path")
                if out_path:
                    out_obj = Path(out_path)
                    result_item["output_path"] = str(out_obj)
                    result_item["json_filename"] = out_obj.name
                    result_item["download_url"] = f"/api/download/{out_obj.stem}"

                if result.get("template_csv_path"):
                    tpath = Path(str(result.get("template_csv_path")))
                    result_item["template_csv_path"] = str(tpath)
                    result_item["template_csv_filename"] = tpath.name
                    result_item["template_csv_download_url"] = f"/api/download/{tpath.stem}"
                results_local.append(result_item)
            return results_local

        # 入隊：逐個處理（整個資料夾當一個 job）
        pq: ProcessingQueue = app.state.processing_queue
        results = await pq.enqueue(
            f"Folder:{Path(folder_path).name}",
            None,
            run_folder_job,
        )
        
        return JSONResponse(content={
            "status": "success",
            "message": f"已處理 {len(results)} 個 PDF 檔案",
            "generate_template_csv": bool(GENERATE_TEMPLATE_CSV),
            "results": results
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理錯誤: {str(e)}")


@app.get("/api/download/{filename}")
async def download_output(filename: str):
    """下載生成的輸出檔案（JSON 優先，CSV 向後相容；使用文件名而非 UUID）"""
    # Support direct PDF download by full filename.
    # Example: /api/download/invoice_searchable_20260116_201122.pdf
    if filename.lower().endswith(".pdf"):
        pdf_path = SEARCHABLE_PDF_OUTPUT_DIR / filename
        if pdf_path.exists():
            return FileResponse(path=pdf_path, filename=pdf_path.name, media_type="application/pdf")

    # 優先嘗試 JSON，若找不到再嘗試 CSV（向後相容）
    json_path = JSON_OUTPUT_DIR / f"{filename}.json"
    if json_path.exists():
        return FileResponse(path=json_path, filename=json_path.name, media_type="application/json")

    # Backward compatibility: older runs may have written JSON into OUTPUT_DIR.
    legacy_json_path = OUTPUT_DIR / f"{filename}.json"
    if legacy_json_path.exists():
        return FileResponse(path=legacy_json_path, filename=legacy_json_path.name, media_type="application/json")

    csv_path = CSV_OUTPUT_DIR / f"{filename}.csv"
    if csv_path.exists():
        return FileResponse(path=csv_path, filename=csv_path.name, media_type="text/csv")

    # Backward compatibility: some older runs may have written CSV into OUTPUT_DIR.
    legacy_csv_path = OUTPUT_DIR / f"{filename}.csv"
    if legacy_csv_path.exists():
        return FileResponse(path=legacy_csv_path, filename=legacy_csv_path.name, media_type="text/csv")

    # 嘗試查找匹配的 JSON 或 CSV（處理帶數字後綴的情況）
    matching_json = list(JSON_OUTPUT_DIR.glob(f"{filename}*.json"))
    if matching_json:
        return FileResponse(path=matching_json[0], filename=matching_json[0].name, media_type="application/json")

    matching_legacy_json = list(OUTPUT_DIR.glob(f"{filename}*.json"))
    if matching_legacy_json:
        return FileResponse(path=matching_legacy_json[0], filename=matching_legacy_json[0].name, media_type="application/json")

    matching_csv = list(CSV_OUTPUT_DIR.glob(f"{filename}*.csv"))
    if matching_csv:
        return FileResponse(path=matching_csv[0], filename=matching_csv[0].name, media_type="text/csv")

    matching_legacy_csv = list(OUTPUT_DIR.glob(f"{filename}*.csv"))
    if matching_legacy_csv:
        return FileResponse(path=matching_legacy_csv[0], filename=matching_legacy_csv[0].name, media_type="text/csv")

    raise HTTPException(status_code=404, detail=f"輸出檔案不存在: {filename}.json 或 {filename}.csv")


async def process_pdf_file(
    file_path: Path, 
    file_id: str, 
    base_name: Optional[str] = None,
    task_id: Optional[str] = None,
    output_format: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    mineru_json_variant: Optional[str] = None,
) -> dict:
    """處理 PDF 檔案的完整流程
    
    Args:
        file_path: PDF 檔案路徑
        file_id: 檔案 ID（用於內部追蹤）
        base_name: 基礎文件名（用於生成 CSV）
        task_id: WebSocket 任務 ID（用於進度更新）
    """
    try:
        # 如果沒有提供 base_name，從 file_path 生成
        if base_name is None:
            base_name = sanitize_filename(file_path.name)

        should_generate_template_csv = bool(GENERATE_TEMPLATE_CSV)
        
        # 步驟 1: 檢查 PDF 是否需要 OCR（除非強制 OCR）
        if task_id:
            await progress_manager.send_progress(task_id, 10, "檢查檔案…", "check_format")
        
        if FORCE_OCR:
            if DEBUG:
                print(f"[FORCE_OCR] 強制執行 OCR，跳過檢查: {file_path}")
            needs_ocr = True
        else:
            if task_id:
                await progress_manager.send_progress(task_id, 15, "分析內容…", "analyze_pdf")
            needs_ocr = await pdf_processor.needs_ocr(file_path)
        
        searchable_pdf_path: Optional[Path] = None

        # 步驟 2: 如果需要 OCR，先用 ocrmypdf 產生「可搜尋 PDF」，再讓 MinerU 讀取該 PDF
        if needs_ocr:
            if DEBUG:
                print(f"[OCR_PROCESS] 使用 MinerU 進行 OCR 處理: {file_path}")
            if task_id:
                await progress_manager.send_progress(task_id, 20, "文字辨識中…", "ocr_start")

            # Policy: when OCR is needed, MinerU must run on the searchable PDF.
            # If we cannot produce it, stop processing (no fallback to the uploaded PDF).
            if not GENERATE_SEARCHABLE_PDF:
                if task_id:
                    await progress_manager.send_progress(task_id, 100, "已停止：未啟用可搜尋 PDF 產生", "failed")
                raise RuntimeError(
                    "OCR required but GENERATE_SEARCHABLE_PDF=False. Enable searchable PDF generation to proceed."
                )

            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                searchable_pdf_path = SEARCHABLE_PDF_OUTPUT_DIR / f"{base_name}_searchable_{ts}.pdf"
                if task_id:
                    await progress_manager.send_progress(task_id, 22, "建立可搜尋 PDF…", "searchable_pdf")

                # ocrmypdf is CPU-bound / external-tool heavy; run it off the event loop.
                searchable_pdf_path = await asyncio.to_thread(
                    ocr_to_searchable_pdf,
                    file_path,
                    searchable_pdf_path,
                    language=str(OCR_LANGUAGE or "eng"),
                    redo=bool(FORCE_OCR) or bool(OCR_FORCE_REDO),
                    deskew=bool(OCR_DESKEW),
                    rotate=bool(OCR_ROTATE_PAGES),
                    jobs=int(OCR_JOBS),
                    optimize=int(OCR_OPTIMIZE),
                    image_dpi=OCR_IMAGE_DPI,
                    tessdata_prefix=(Path(str(OCR_TESSDATA_PREFIX)).expanduser() if str(OCR_TESSDATA_PREFIX).strip() else None),
                )
                if not searchable_pdf_path or (not Path(searchable_pdf_path).exists()):
                    raise RuntimeError("Searchable PDF was not produced")

                pdf_for_mineru = searchable_pdf_path

                if DEBUG:
                    print(f"[SEARCHABLE_PDF] ✓ Created searchable PDF: {searchable_pdf_path}")
            except Exception as e:
                searchable_pdf_path = None
                if task_id:
                    await progress_manager.send_progress(task_id, 100, f"已停止：建立可搜尋 PDF 失敗：{e}", "failed")
                raise RuntimeError(f"Failed to create searchable PDF; aborting: {e}") from e

            # 傳遞進度回調給 MinerU 服務
            async def mineru_progress_callback(p, msg, stage):
                if task_id:
                    # 將 MinerU 的進度 (0-100) 映射到總進度的 20-70% 範圍
                    total_progress = 20 + int(p * 0.5)
                    await progress_manager.send_progress(task_id, total_progress, msg, stage)
            
            try:
                mineru_json = await mineru_service.process_pdf(
                    pdf_for_mineru,
                    progress_callback=mineru_progress_callback if task_id else None,
                    json_variant=mineru_json_variant,
                )
            except Exception as e:
                # MinerU 失敗：回退到「純文字提取」避免生成空/假資料 CSV
                if DEBUG:
                    print(f"[OCR_PROCESS] MinerU 失敗，回退到純文字提取: {e}")
                if task_id:
                    await progress_manager.send_progress(task_id, 35, "MinerU 失敗，改用文字提取…", "ocr_fallback")
                # Prefer extracting text from the searchable-PDF if we created it.
                mineru_json = await pdf_processor.extract_text_to_json(pdf_for_mineru)
        else:
            if DEBUG:
                print(f"[TEXT_EXTRACT] 直接提取文字（不需要 OCR）: {file_path}")
            if task_id:
                await progress_manager.send_progress(task_id, 30, "整理文字…", "extract_text")
            # 如果不需要 OCR，直接提取文字
            mineru_json = await pdf_processor.extract_text_to_json(file_path)

        # Always write extracted JSON to JSON_OUTPUT_DIR so JSON is available even if table/CSV output is disabled
        try:
            out_json_path = get_unique_json_path(f"{base_name}_mineru", JSON_OUTPUT_DIR)
            with open(out_json_path, "w", encoding="utf-8") as jf:
                json.dump(mineru_json, jf, ensure_ascii=False, indent=2)

            if DEBUG:
                print(f"[MINERU_OUTPUT] ✓ Extracted JSON saved: {out_json_path}")
        except Exception as e:
            # Non-fatal: continue with AI generation even if saving extracted JSON fails
            out_json_path = None
            if DEBUG:
                print(f"[MINERU_OUTPUT] ⚠ Failed to save extracted JSON: {e}")

        # Step 3: Generate final structured output (JSON by default; CSV only for backward compatibility)
        if task_id:
            await progress_manager.send_progress(task_id, 70, "生成結構化輸出…", "generate_output")

        # Build multimodal inputs for Azure OpenAI: all PDF pages as images + full MinerU JSON text.
        image_data_urls = None
        try:
            from config import (
                AZURE_OPENAI_INCLUDE_IMAGES,
                AZURE_OPENAI_IMAGE_MAX_PAGES,
                AZURE_OPENAI_IMAGE_MAX_SIDE,
                AZURE_OPENAI_IMAGE_FORMAT,
                AI_SERVICE,
            )
            include_images = bool(AZURE_OPENAI_INCLUDE_IMAGES)
            if include_images and (getattr(azure_ai_service, "ai_service", None) or AI_SERVICE) == "azure_openai" and getattr(azure_ai_service, "client", None):
                # pdf2image + PIL are synchronous; run in a thread.
                def _encode_pdf_images() -> list[str]:
                    import base64
                    import io
                    from PIL import Image

                    images = pdf_processor.convert_to_images(file_path)
                    max_pages = int(AZURE_OPENAI_IMAGE_MAX_PAGES or 0)
                    if max_pages > 0:
                        images = images[:max_pages]

                    fmt = (AZURE_OPENAI_IMAGE_FORMAT or "jpeg").strip().lower()
                    if fmt not in ("jpeg", "jpg", "png"):
                        fmt = "jpeg"
                    pil_format = "JPEG" if fmt in ("jpeg", "jpg") else "PNG"
                    mime = "image/jpeg" if pil_format == "JPEG" else "image/png"

                    max_side = int(AZURE_OPENAI_IMAGE_MAX_SIDE or 0)
                    out: list[str] = []
                    for im in images:
                        if not isinstance(im, Image.Image):
                            continue

                        # Resize to control payload size.
                        if max_side and max_side > 0:
                            w, h = im.size
                            longest = max(w, h)
                            if longest > max_side:
                                scale = max_side / float(longest)
                                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                                im = im.copy()
                                im.thumbnail(new_size)

                        # Ensure RGB for JPEG.
                        if pil_format == "JPEG" and im.mode not in ("RGB", "L"):
                            im = im.convert("RGB")

                        buf = io.BytesIO()
                        save_kwargs = {}
                        if pil_format == "JPEG":
                            save_kwargs["quality"] = 80
                            save_kwargs["optimize"] = True
                        im.save(buf, format=pil_format, **save_kwargs)
                        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                        out.append(f"data:{mime};base64,{b64}")
                    return out

                image_data_urls = await asyncio.to_thread(_encode_pdf_images)
                if DEBUG:
                    print(f"[AZURE_OPENAI] Prepared {len(image_data_urls)} image(s) for multimodal request")
        except Exception as e:
            image_data_urls = None
            if DEBUG:
                print(f"[AZURE_OPENAI] ⚠ Failed to prepare PDF images; continuing without images: {e}")

        async def ai_progress_callback(p: int, msg: str, stage: str):
            if not task_id:
                return
            try:
                # Map AI progress (0-100) into 70-98 range
                total_progress = 70 + int(max(0, min(100, int(p))) * 0.28)
                await progress_manager.send_progress(task_id, total_progress, msg, stage)
            except Exception:
                pass

        try:
            out_path = get_unique_json_path(base_name, JSON_OUTPUT_DIR)

            final_path = await azure_ai_service.generate_csv_from_json(
                mineru_json,
                out_path,
                progress_callback=ai_progress_callback if task_id else None,
                output_format="json",
                system_prompt_override=system_prompt,
                user_prompt_override=user_prompt,
                image_data_urls=image_data_urls,
            )

            if task_id:
                await progress_manager.send_progress(task_id, 100, "完成", "done")

            result: Dict[str, Any] = {
                "needs_ocr": needs_ocr,
                "json_available": True,
            }
            if out_json_path:
                result["mineru_output_path"] = str(out_json_path)
            if searchable_pdf_path:
                result["searchable_pdf_path"] = str(searchable_pdf_path)

            result["output_path"] = str(final_path)

            # Optional: template-based CSV output (uses OpenAI JSON + MinerU sidecar)
            if should_generate_template_csv:
                try:
                    pdf_for_sidecar = searchable_pdf_path or file_path
                    _, sidecar_path = mineru_service.find_selected_json_and_sidecar(
                        pdf_for_sidecar,
                        json_variant=mineru_json_variant,
                    )
                    if sidecar_path:
                        from services.jsoncsv_app import generate_template_csv_file

                        template_csv_out = get_unique_csv_path(f"{base_name}_template", CSV_OUTPUT_DIR)
                        template_csv_path = generate_template_csv_file(
                            data_json_path=Path(final_path),
                            sidecar_json_path=Path(sidecar_path),
                            output_path=Path(template_csv_out),
                        )
                        result["template_csv_path"] = str(template_csv_path)
                    else:
                        if DEBUG:
                            print("[TEMPLATE_CSV] ⚠ MinerU sidecar not found; skipping template CSV")
                except Exception as e:
                    if DEBUG:
                        print(f"[TEMPLATE_CSV] ⚠ Failed to generate template CSV (continuing): {e}")

            return result

        except Exception as e:
            # If AI output generation fails, return extracted JSON if we have it.
            if DEBUG:
                print(f"[JSON_GENERATION] ✗ Output generation failed: {e}")
            if task_id:
                await progress_manager.send_progress(task_id, 95, "輸出生成失敗（回退到提取內容）", "output_fallback")

            if out_json_path:
                return {
                    "needs_ocr": needs_ocr,
                    "json_available": True,
                    "output_path": str(out_json_path),
                    "fallback": "mineru_json",
                }
            raise

    except Exception as e:
        if DEBUG:
            print(f"[PROCESS_PDF] ✗ Processing failed: {e}")
        raise
