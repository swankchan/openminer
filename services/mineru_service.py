"""MinerU 2.5 integration service (using vllm-async-engine)."""
from pathlib import Path
import json
from typing import Dict, Any, Optional, Union, List, Tuple
import os
import io
import asyncio
from datetime import datetime
from PIL import Image
import pdf2image

from config import (
    MINERU_METHOD, 
    DEBUG as APP_DEBUG, 
    USE_VLLM_ASYNC,
    MINERU_MODEL_NAME,
    MINERU_DEVICE,
    MINERU_OUTPUT_SOURCE,
    MINERU_JSON_VARIANT,
    MINERU_OUTPUT_DIR,
    MINERU_INCLUDE_HEADER_ZONES,
    MINERU_HEADER_ZONE_RATIO,
    MINERU_INCLUDE_PDF_TEXT_LAYER,
    MINERU_PDF_TEXT_LAYER_MAX_LINES,
    SIDECAR_OUTPUT_DIR,
    ALLOW_MOCK_MINERU_OUTPUT,
)

# Lazy imports: only used when vllm-async-engine is enabled.
_async_llm = None
_client = None


class MinerUService:
    """MinerU 2.5 OCR service integration (using vllm-async-engine)."""
    
    def __init__(self):
        self.output_dir = Path(MINERU_OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
        self._initialized = False
        self._init_lock = asyncio.Lock()
    
    async def _initialize(self):
        """Initialize AsyncLLM and MinerUClient (lazy initialization)."""
        global _async_llm, _client
        
        # Prevent concurrent requests/startup flows from initializing twice
        # (especially since vLLM initialization can be slow).
        async with self._init_lock:
            if self._initialized:
                return
        
            if not USE_VLLM_ASYNC:
                if APP_DEBUG:
                    print("[MINERU] Using CLI mode (USE_VLLM_ASYNC=False)")
                self._initialized = True
                return
        
            try:
                from vllm.v1.engine.async_llm import AsyncLLM
                from vllm.engine.arg_utils import AsyncEngineArgs
                from mineru_vl_utils import MinerUClient
            
                # Try to import MinerULogitsProcessor (vllm>=0.10.1).
                try:
                    from mineru_vl_utils import MinerULogitsProcessor
                    logits_processors = [MinerULogitsProcessor]
                except ImportError:
                    logits_processors = []
                    if APP_DEBUG:
                        print("[MINERU] MinerULogitsProcessor not available (vllm version may be < 0.10.1)")
            
                if APP_DEBUG:
                    print(f"[MINERU] Initializing vllm-async-engine, model: {MINERU_MODEL_NAME}")
                    print(f"[MINERU] Device setting: {MINERU_DEVICE} (vllm auto-detects)")
            
                # Device environment (if CPU is explicitly requested).
                # Note: On Windows, vllm may not support CPU mode and may fall back automatically.
                if MINERU_DEVICE == "cpu":
                    # Do not set CUDA_VISIBLE_DEVICES; let vllm handle it.
                    if APP_DEBUG:
                        print("[MINERU] Attempting CPU mode (may not be supported)")
            
                # Initialize AsyncLLM.
                # Note: AsyncEngineArgs does not support a device argument; device is auto-detected.
                # vllm auto-detects available GPUs; it may fail if no GPU is available.
                try:
                    engine_args = AsyncEngineArgs(
                        model=MINERU_MODEL_NAME,
                        logits_processors=logits_processors if logits_processors else None
                    )
                except Exception as e:
                    if APP_DEBUG:
                        print(f"[MINERU] Failed to create AsyncEngineArgs: {e}")
                    raise
            
                _async_llm = AsyncLLM.from_engine_args(engine_args)
            
                # Initialize MinerUClient.
                _client = MinerUClient(
                    backend="vllm-async-engine",
                    vllm_async_llm=_async_llm,
                    # Ensure headers/footers (paratext) are not discarded by default.
                    abandon_paratext=False,
                )
            
                self._initialized = True
            
                if APP_DEBUG:
                    print("[MINERU] vllm-async-engine initialized successfully")
                
            except ImportError as e:
                if APP_DEBUG:
                    print(f"[MINERU] Could not import vllm-async-engine; falling back to CLI mode: {e}")
                self._initialized = True
            except Exception as e:
                if APP_DEBUG:
                    print(f"[MINERU] Failed to initialize vllm-async-engine; falling back to CLI mode: {e}")
                self._initialized = True
    
    async def process_pdf(
        self,
        pdf_path: Path,
        progress_callback=None,
        json_variant: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a PDF with MinerU and return a JSON result.
        
        Args:
            pdf_path: Path to the PDF file
            progress_callback: Progress callback that receives (percentage, message, stage)
        """
        # Ensure initialized.
        await self._initialize()
        
        # If MINERU_OUTPUT_SOURCE=json, prefer CLI mode to obtain a complete content_list.json,
        # because vllm-async-engine does not generate content_list.json and the output may be incomplete.
        if MINERU_OUTPUT_SOURCE == "json":
            if APP_DEBUG:
                print("[MINERU] MINERU_OUTPUT_SOURCE=json; using CLI mode to obtain a complete content_list.json")
            effective_variant = (json_variant or MINERU_JSON_VARIANT or "").strip()
            return await self._process_with_cli(pdf_path, progress_callback, json_variant=effective_variant)

        # If using vllm-async-engine and initialized successfully.
        if USE_VLLM_ASYNC and _client is not None:
            return await self._process_with_vllm_async(pdf_path, progress_callback)
        else:
            # Fall back to CLI mode.
            return await self._process_with_cli(pdf_path, progress_callback)

    @staticmethod
    def _normalize_json_variant(value: Optional[str]) -> str:
        if value is None:
            return ""
        v = str(value).strip().lower()
        if not v:
            return ""

        # Friendly aliases
        aliases = {
            "content_list": "_content_list.json",
            "contentlist": "_content_list.json",
            "middle": "_middle.json",
            "model": "_model.json",
        }
        if v in aliases:
            return aliases[v]

        # Allow users to pass the suffix directly.
        # Examples: _content_list.json, content_list.json, middle.json
        if not v.endswith(".json"):
            v = f"{v}.json"
        if not v.startswith("_"):
            v = f"_{v}"
        return v

    @staticmethod
    def _select_json_file(json_files: list, json_variant: str) -> Path:
        if not json_files:
            raise FileNotFoundError("No MinerU-generated JSON files found")

        # Deterministic ordering
        ordered = sorted(json_files, key=lambda p: (p.name.lower(), str(p)))

        variant_suffix = MinerUService._normalize_json_variant(json_variant)
        if variant_suffix:
            matches = [p for p in ordered if p.name.lower().endswith(variant_suffix)]
            if matches:
                return matches[0]
            available = ", ".join([p.name for p in ordered[:10]])
            more = "" if len(ordered) <= 10 else f" (+{len(ordered) - 10} more)"
            raise FileNotFoundError(
                f"Requested MinerU JSON variant '{json_variant}' not found (suffix: {variant_suffix}). "
                f"Available: {available}{more}"
            )

        # Default: prefer content_list.json; otherwise use the first JSON file.
        for p in ordered:
            if p.name.lower().endswith("_content_list.json"):
                return p
        return ordered[0]

    def get_runtime_status(self) -> Dict[str, Any]:
        """Report MinerU runtime status (for frontend display; no sensitive info)."""
        global _client

        vllm_available = bool(USE_VLLM_ASYNC) and (_client is not None)

        # Note: When MINERU_OUTPUT_SOURCE=json, process_pdf forces CLI,
        # because vllm-async-engine does not generate a complete content_list.json.
        forced_cli = MINERU_OUTPUT_SOURCE == "json"

        effective_engine = "cli"
        if (not forced_cli) and vllm_available:
            effective_engine = "vllm-async-engine"

        return {
            "initialized": bool(self._initialized),
            "use_vllm_async": bool(USE_VLLM_ASYNC),
            "vllm_available": bool(vllm_available),
            "forced_cli": bool(forced_cli),
            "effective_engine": effective_engine,
            "mineru_method": MINERU_METHOD,
            "output_source": MINERU_OUTPUT_SOURCE,
            "model_name": MINERU_MODEL_NAME,
            "device": MINERU_DEVICE,
        }
    
    async def _process_with_vllm_async(self, pdf_path: Path, progress_callback=None) -> Dict[str, Any]:
        """Process a PDF using vllm-async-engine."""
        try:
            if APP_DEBUG:
                print(f"[MINERU_VLLM] Starting PDF processing: {pdf_path}")
            
            # Check whether the caller expects PDF output rather than JSON.
            if MINERU_OUTPUT_SOURCE in ["layout_pdf", "span_pdf", "both_pdf"]:
                pdf_paths = self._find_mineru_pdfs(pdf_path)
                if pdf_paths and (pdf_paths.get("layout_pdf") or pdf_paths.get("span_pdf")):
                    if APP_DEBUG:
                        print(f"[MINERU_VLLM] Using PDF output source: {MINERU_OUTPUT_SOURCE}")
                        if pdf_paths.get("layout_pdf"):
                            print(f"[MINERU_VLLM]   layout_pdf: {pdf_paths['layout_pdf']}")
                        if pdf_paths.get("span_pdf"):
                            print(f"[MINERU_VLLM]   span_pdf: {pdf_paths['span_pdf']}")
                    return pdf_paths
                else:
                    if APP_DEBUG:
                        print("[MINERU_VLLM] ⚠ No PDF output found; falling back to JSON")
                    # Continue with JSON generation.
            
            # Convert PDF pages to images (MinerU 2.5 accepts image input).
            images = pdf2image.convert_from_path(str(pdf_path))
            
            if APP_DEBUG:
                print(f"[MINERU_VLLM] Converted PDF into {len(images)} images")
            
            # Process each page.
            pages_data = []
            for page_num, image in enumerate(images, 1):
                if APP_DEBUG:
                    print(f"[MINERU_VLLM] Processing page {page_num}")
                
                # Extract content via MinerU.
                extracted_blocks = await _client.aio_two_step_extract(image)
                
                # Convert to the standard output format.
                page_data = self._convert_blocks_to_page_format(
                    extracted_blocks, 
                    page_num
                )
                pages_data.append(page_data)
            
            # Compose the full document structure.
            result = {
                "document": {
                    "pages": pages_data,
                    "total_pages": len(pages_data),
                    "metadata": {
                        "source_file": str(pdf_path),
                        "processing_method": "vllm-async-engine"
                    }
                }
            }
            
            if APP_DEBUG:
                print(f"[MINERU_VLLM] Finished processing ({len(pages_data)} pages)")
            
            return result
            
        except Exception as e:
            if APP_DEBUG:
                print(f"[MINERU_VLLM] Processing failed: {e}")
            # Fall back to CLI mode.
            return await self._process_with_cli(pdf_path)
    
    def _convert_blocks_to_page_format(
        self, 
        extracted_blocks: Any, 
        page_num: int
    ) -> Dict[str, Any]:
        """Convert MinerU blocks into the standard page format."""
        # Extract text content.
        text_parts = []
        elements = []
        tables = []
        
        # Process extracted_blocks (adjust as needed to match the actual format).
        if isinstance(extracted_blocks, list):
            for block in extracted_blocks:
                if isinstance(block, dict):
                    block_type = block.get("type", "text").lower()
                    
                    # Extract text.
                    block_text = block.get("text", "") or block.get("content", "")
                    if block_text:
                        text_parts.append(block_text)
                    
                    # Build element.
                    element = {
                        "type": block_type,
                        "content": block_text,
                        "bbox": block.get("bbox", [0, 0, 0, 0])
                    }
                    
                    # If this is a table, extract table content.
                    if block_type == "table":
                        table_body = block.get("table_body", "")
                        if table_body:
                            element["table_body"] = table_body
                            tables.append({
                                "table_body": table_body,
                                "text": block_text
                            })
                    
                    # Attach optional metadata.
                    if "font_size" in block:
                        element["font_size"] = block["font_size"]
                    if "font_family" in block:
                        element["font_family"] = block["font_family"]
                    
                    elements.append(element)
        
        # Merge all text.
        full_text = "\n".join(text_parts) if text_parts else ""
        
        return {
            "page_number": page_num,
            "text": full_text,
            "elements": elements,
            "tables": tables,  # includes extracted table content
            "figures": []  # figures/images require additional handling
        }
    
    async def _process_with_cli(
        self,
        pdf_path: Path,
        progress_callback=None,
        json_variant: str = "",
    ) -> Dict[str, Any]:
        """Process a PDF via the CLI (fallback path)."""
        try:
            # MinerU creates a folder named after the PDF.
            # Output layout: mineru_outputs/{pdf_name}/hybrid_{method}/*.json
            output_path = self.output_dir
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Map MINERU_METHOD to the expected subfolder name (defined early for later use).
            method_to_folder = {
                "auto": "hybrid_auto",
                "ocr": "hybrid_ocr",
                "txt": "hybrid_txt"
            }
            hybrid_folder = method_to_folder.get(MINERU_METHOD, "hybrid_auto")
            
            # Execute MinerU CLI, -m ocr/auto/txt
            # -b pipeline / hybrid-auto-engine (ONLY WHEN YOU HAVE RTX5090!)
            cmd = [
                os.getenv("MINERU_CMD", "mineru"),
                "-p", str(pdf_path),
                "-o", str(output_path),
                "-m", MINERU_METHOD,
                "-b", str("pipeline"),
            ]
            
            # Send initial progress.
            if progress_callback:
                await progress_callback(5, "Preparing to start MinerU…", "mineru_init")
            
            if APP_DEBUG:
                import time
                start_time = time.time()
                print(f"[MINERU_CLI] Starting PDF processing: {pdf_path.name}")
                print(f"[MINERU_CLI] Command: {' '.join(cmd)}")
                print(f"[MINERU_CLI] Output dir: {output_path}")
                print(f"[MINERU_CLI] Method: {MINERU_METHOD} -> {hybrid_folder}")
                print(f"[MINERU_CLI] Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            import subprocess
            import threading
            import queue
            
            # Send progress update.
            if progress_callback:
                await progress_callback(10, "Starting MinerU…", "mineru_execute")
            
            # Real-time output mode:
            # - If progress_callback is provided, we must stream output; otherwise the event loop
            #   can be blocked and the frontend only updates after the subprocess finishes.
            # - Also stream in DEBUG mode to help troubleshooting.
            stream_mode = bool(progress_callback) or APP_DEBUG
            if stream_mode:
                if APP_DEBUG:
                    print("[MINERU_CLI] Running MinerU (this may take a while)...")
                    print("[MINERU_CLI] Tip: processing large PDFs may take several minutes")
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Use queues to collect output.
                stdout_queue = queue.Queue()
                stderr_queue = queue.Queue()
                
                # Queue for progress updates produced in threads.
                progress_queue = queue.Queue() if progress_callback else None
                
                # Async task that consumes the progress queue and calls the callback.
                async def process_progress_queue():
                    """Read progress updates from the queue and invoke the callback."""
                    if not progress_callback or not progress_queue:
                        return
                    try:
                        process_running = True
                        while process_running:
                            # Prefer draining all queued items first.
                            processed_any = False
                            while not progress_queue.empty():
                                try:
                                    progress_data = progress_queue.get_nowait()
                                    if progress_data:
                                        percentage, message, stage = progress_data
                                        await progress_callback(percentage, message, stage)
                                        if APP_DEBUG:
                                            print(f"[MINERU_PROGRESS] {percentage}% - {message} ({stage})")
                                        processed_any = True
                                except queue.Empty:
                                    break
                            
                            # If we processed items, continue quickly.
                            if processed_any:
                                await asyncio.sleep(0.01)  # brief yield so other tasks can run
                                continue
                            
                            # If the queue is empty, check the process status.
                            if process.poll() is not None:
                                # Process finished; flush remaining updates.
                                while not progress_queue.empty():
                                    try:
                                        progress_data = progress_queue.get_nowait()
                                        if progress_data:
                                            percentage, message, stage = progress_data
                                            await progress_callback(percentage, message, stage)
                                    except queue.Empty:
                                        break
                                process_running = False
                                break
                            
                            # Queue empty and process still running: brief wait.
                            await asyncio.sleep(0.05)
                    except Exception as e:
                        if APP_DEBUG:
                            print(f"[MINERU_PROGRESS] Error while processing progress queue: {e}")
                
                # Start progress processing task.
                progress_task = None
                if progress_callback and progress_queue:
                    progress_task = asyncio.create_task(process_progress_queue())
                
                # Progress heartbeat: if MinerU doesn't print parseable stage text, still emit
                # periodic progress so the UI shows it's running.
                heartbeat_task = None
                if progress_callback and progress_queue:
                    import time
                    last_emit_ts = time.monotonic()
                    heartbeat_pct = 12  # slowly advance between "execute" (10) and "extract" (60)
                    
                    # When the queue consumer actually sends updates, it refreshes last_emit_ts.
                    original_progress_callback = progress_callback
                    async def wrapped_progress_callback(pct, msg, stage):
                        nonlocal last_emit_ts
                        last_emit_ts = time.monotonic()
                        await original_progress_callback(pct, msg, stage)
                    progress_callback = wrapped_progress_callback
                    
                    async def heartbeat():
                        nonlocal heartbeat_pct, last_emit_ts
                        while process.poll() is None:
                            now = time.monotonic()
                            # If no progress for 2 seconds, advance one step.
                            if (now - last_emit_ts) >= 2.0 and heartbeat_pct < 60:
                                progress_queue.put((heartbeat_pct, "MinerU parsing…", "mineru_running"))
                                heartbeat_pct += 1
                                last_emit_ts = now
                            await asyncio.sleep(1.0)
                    
                    heartbeat_task = asyncio.create_task(heartbeat())
                
                def read_stdout():
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            line = line.strip()
                            stdout_queue.put(line)
                            if APP_DEBUG:
                                print(f"[MINERU_STDOUT] {line}")
                            # Parse MinerU output and update progress (via the queue).
                            if progress_callback and progress_queue:
                                line_lower = line.lower()
                                if "layout" in line_lower or "layout predict" in line_lower:
                                    progress_queue.put((30, "Analyzing layout…", "layout_predict"))
                                elif "mfd" in line_lower or "mfd predict" in line_lower:
                                    progress_queue.put((40, "Recognizing structure…", "mfd_predict"))
                                elif "table" in line_lower or "table-ocr" in line_lower or "table ocr" in line_lower:
                                    progress_queue.put((50, "Recognizing tables…", "table_ocr"))
                                elif "extract" in line_lower or "extracting" in line_lower:
                                    progress_queue.put((60, "Preparing output…", "extract_content"))
                    process.stdout.close()
                
                def read_stderr():
                    for line in iter(process.stderr.readline, ''):
                        if line:
                            line = line.strip()
                            stderr_queue.put(line)
                            if APP_DEBUG:
                                print(f"[MINERU_STDERR] {line}")
                            # MinerU sends many progress/log lines to stderr; parse it as well.
                            if progress_callback and progress_queue:
                                line_lower = line.lower()
                                if "layout" in line_lower or "layout predict" in line_lower:
                                    progress_queue.put((30, "Analyzing layout…", "layout_predict"))
                                elif "mfd" in line_lower or "mfd predict" in line_lower:
                                    progress_queue.put((40, "Recognizing structure…", "mfd_predict"))
                                elif "table" in line_lower or "table-ocr" in line_lower or "table ocr" in line_lower:
                                    progress_queue.put((50, "Recognizing tables…", "table_ocr"))
                                elif "extract" in line_lower or "extracting" in line_lower:
                                    progress_queue.put((60, "Preparing output…", "extract_content"))
                    process.stderr.close()
                
                # Start reader threads.
                stdout_thread = threading.Thread(target=read_stdout, daemon=True)
                stderr_thread = threading.Thread(target=read_stderr, daemon=True)
                stdout_thread.start()
                stderr_thread.start()
                
                # Wait for process completion. Do not use process.wait() here, because it can
                # block the event loop and prevent real-time progress updates.
                while process.poll() is None:
                    await asyncio.sleep(0.1)
                return_code = process.returncode
                
                # Wait for threads to finish.
                stdout_thread.join(timeout=1)
                stderr_thread.join(timeout=1)
                
                # Wait for progress task to finish.
                if progress_task:
                    try:
                        await asyncio.wait_for(progress_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        if APP_DEBUG:
                            print("[MINERU_PROGRESS] Progress handler timed out; continuing")
                    except Exception as e:
                        if APP_DEBUG:
                            print(f"[MINERU_PROGRESS] Error while awaiting progress task: {e}")
                
                # Stop heartbeat task.
                if heartbeat_task:
                    try:
                        heartbeat_task.cancel()
                    except Exception:
                        pass
                
                # Collect all output.
                stdout_lines = []
                while not stdout_queue.empty():
                    stdout_lines.append(stdout_queue.get())
                
                stderr_lines = []
                while not stderr_queue.empty():
                    stderr_lines.append(stderr_queue.get())
                
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, cmd, stderr='\n'.join(stderr_lines))
                
                result = type('obj', (object,), {
                    'returncode': return_code,
                    'stdout': '\n'.join(stdout_lines),
                    'stderr': '\n'.join(stderr_lines)
                })()
            else:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
            
            # Send progress update.
            if progress_callback:
                await progress_callback(65, "MinerU finished, reading results…", "mineru_complete")
            
            if APP_DEBUG:
                elapsed_time = time.time() - start_time
                print(f"[MINERU_CLI] MinerU completed in {elapsed_time:.2f}s ({elapsed_time/60:.2f} min)")
                if result.stdout:
                    stdout_lines = result.stdout.split('\n')
                    print(f"[MINERU_CLI] Stdout summary: {len(stdout_lines)} lines")
                    # Show last few lines (may include progress info).
                    if len(stdout_lines) > 0:
                        print("[MINERU_CLI] Last stdout lines:")
                        for line in stdout_lines[-5:]:
                            if line.strip():
                                print(f"[MINERU_CLI]   {line}")
                if result.stderr:
                    stderr_lines = result.stderr.split('\n')
                    print(f"[MINERU_CLI] Stderr summary: {len(stderr_lines)} lines")
                    if len(stderr_lines) > 0:
                        for line in stderr_lines[-5:]:
                            if line.strip():
                                print(f"[MINERU_CLI]   {line}")
            
            # PDF file name (without extension). MinerU uses this to create the output folder.
            pdf_name = pdf_path.stem
            pdf_output_dir = output_path / pdf_name
            
            if APP_DEBUG:
                print(f"[MINERU_CLI] PDF output dir: {pdf_output_dir}")
                print(f"[MINERU_CLI] Method: {MINERU_METHOD}")
            
            # Find generated JSON files.
            # MinerU may decide whether to use hybrid based on runtime results, so check both:
            # 1. hybrid_{method} (e.g., hybrid_auto, hybrid_ocr, hybrid_txt)
            # 2. {method} (e.g., auto, ocr, txt)
            json_files = []
            
            # Candidate folder names (priority: hybrid first, then non-hybrid).
            possible_folders = [
                f"hybrid_{MINERU_METHOD}",  # hybrid mode
                MINERU_METHOD                # non-hybrid mode
            ]
            
            if APP_DEBUG:
                print(f"[MINERU_CLI] Will check possible folders: {possible_folders}")
            
            # Check expected folders first (in priority order).
            for folder_name in possible_folders:
                expected_json_path = pdf_output_dir / folder_name
                if APP_DEBUG:
                    print(f"[MINERU_CLI] Checking path: {expected_json_path} (exists: {expected_json_path.exists()})")
                
                if expected_json_path.exists():
                    found = list(expected_json_path.glob("*.json"))
                    if found:
                        json_files.extend(found)
                        if APP_DEBUG:
                            print(f"[MINERU_CLI] ✓ Found {len(found)} JSON file(s) in {folder_name}")
                            for json_file in found[:3]:  # show first 3
                                print(f"[MINERU_CLI]   - {json_file.name}")
                        break  # stop searching after a hit
                    else:
                        if APP_DEBUG:
                            print(f"[MINERU_CLI] ⚠ Folder exists but no JSON files found: {folder_name}")
            
            # If still not found, search all subfolders under the PDF output directory.
            if not json_files:
                if APP_DEBUG:
                    print("[MINERU_CLI] Searching all candidate folders under the PDF output dir...")
                
                if pdf_output_dir.exists():
                    subdirs = [d for d in pdf_output_dir.iterdir() if d.is_dir()]
                    if APP_DEBUG:
                        print(f"[MINERU_CLI] Found {len(subdirs)} subfolder(s): {[d.name for d in subdirs]}")
                    
                    # Check all candidate folder names (hybrid_* and plain method name).
                    for subdir in subdirs:
                        if subdir.is_dir():
                            # Match either hybrid_{method} or {method}.
                            folder_name = subdir.name
                            is_hybrid_match = folder_name == f"hybrid_{MINERU_METHOD}"
                            is_method_match = folder_name == MINERU_METHOD
                            
                            if is_hybrid_match or is_method_match:
                                found = list(subdir.glob("*.json"))
                                if found:
                                    json_files.extend(found)
                                    if APP_DEBUG:
                                        print(f"[MINERU_CLI] ✓ Found {len(found)} JSON file(s) in {folder_name}")
            
            # If still not found, recursively search within THIS PDF's directory (last resort).
            # Important: never rglob on output_path globally, or we might pick up JSON from previous
            # runs and appear successful while returning stale results.
            if not json_files:
                if APP_DEBUG:
                    print("[MINERU_CLI] Recursive search (last resort)...")
                if pdf_output_dir.exists():
                    json_files = list(pdf_output_dir.rglob("*.json"))
                else:
                    json_files = []
                if APP_DEBUG:
                    if json_files:
                        print(f"[MINERU_CLI] ✓ Recursive search found {len(json_files)} JSON file(s)")
                        for json_file in json_files[:3]:  # show first 3
                            try:
                                print(f"[MINERU_CLI]   - {json_file.relative_to(pdf_output_dir)}")
                            except Exception:
                                print(f"[MINERU_CLI]   - {json_file}")
                    else:
                        print("[MINERU_CLI] ✗ Recursive search found no JSON files")
            
            if not json_files:
                raise FileNotFoundError(
                    f"Could not find MinerU-generated JSON files. "
                    f"Checked path: {pdf_output_dir} (possible folders: {possible_folders})"
                )
            
            # Select the requested JSON variant (default: content_list.json).
            json_file = self._select_json_file(json_files, json_variant)
            if APP_DEBUG:
                if json_variant:
                    print(f"[MINERU_CLI] Requested JSON variant: {json_variant} -> {json_file.name}")
                else:
                    print(f"[MINERU_CLI] Selected JSON file (default preference): {json_file.name}")
            
            if APP_DEBUG:
                print(f"[MINERU_CLI] Final selected JSON file: {json_file}")
            
            # Check whether we should return PDF output instead of JSON.
            if MINERU_OUTPUT_SOURCE in ["layout_pdf", "span_pdf", "both_pdf"]:
                pdf_paths = self._find_mineru_pdfs(pdf_path)
                if pdf_paths and (pdf_paths.get("layout_pdf") or pdf_paths.get("span_pdf")):
                    if APP_DEBUG:
                        print(f"[MINERU_CLI] Using PDF output source: {MINERU_OUTPUT_SOURCE}")
                        if pdf_paths.get("layout_pdf"):
                            print(f"[MINERU_CLI]   layout_pdf: {pdf_paths['layout_pdf']}")
                        if pdf_paths.get("span_pdf"):
                            print(f"[MINERU_CLI]   span_pdf: {pdf_paths['span_pdf']}")
                    return pdf_paths
                else:
                    if APP_DEBUG:
                        print("[MINERU_CLI] ⚠ No PDF output found; falling back to JSON")
            
            # Default: return JSON.
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Merge PDF embedded text layer (header) to avoid missing vendor names like JEBSEN.
            data, pdf_text_layer_meta = self._merge_pdf_text_layer_header(pdf_path, data)

            # Header zone plugin: promote discarded header blocks to text to avoid downstream dropping.
            data, header_plugin_meta = self._apply_header_zone_plugin(data)

            # Write a sidecar JSON with confidence + metadata into the MinerU output folder.
            try:
                sidecar = self._build_mineru_sidecar(
                    pdf_path=pdf_path,
                    selected_json_file=json_file,
                    mineru_json=data,
                    json_variant=json_variant,
                    pdf_text_layer_meta=pdf_text_layer_meta,
                    header_plugin_meta=header_plugin_meta,
                )
                self._write_mineru_sidecar(selected_json_file=json_file, sidecar=sidecar)
            except Exception as e:
                if APP_DEBUG:
                    print(f"[MINERU_SIDECAR] ⚠ Failed to write sidecar (continuing): {e}")
            return data
                
        except Exception as e:
            if APP_DEBUG:
                print(f"[MINERU_CLI] Processing failed: {e}")
            # By default we do NOT return "mock OCR" data, to avoid producing CSV output that appears
            # successful but contains fake content. If you really want mock output (e.g., for a UI demo),
            # set ALLOW_MOCK_MINERU_OUTPUT=True.
            if ALLOW_MOCK_MINERU_OUTPUT:
                return self._generate_mock_mineru_output(pdf_path)
            raise

    def _apply_header_zone_plugin(self, mineru_json: Any) -> Tuple[Any, Dict[str, Any]]:
        meta: Dict[str, Any] = {
            "enabled": bool(MINERU_INCLUDE_HEADER_ZONES),
            "promoted_discarded_total": 0,
            "promoted_discarded_headers": 0,
            "header_ratio": float(MINERU_HEADER_ZONE_RATIO) if MINERU_HEADER_ZONE_RATIO else 0.12,
        }

        if not MINERU_INCLUDE_HEADER_ZONES:
            return mineru_json, meta

        if not isinstance(mineru_json, list):
            return mineru_json, meta

        if not mineru_json:
            return mineru_json, meta

        # NOTE: "header zone" tagging still uses header_ratio, but retention is forced:
        # any non-empty discarded text will be promoted to normal content so downstream
        # logic cannot accidentally drop headers/footers.
        header_ratio = meta["header_ratio"]
        if header_ratio <= 0:
            header_ratio = 0.12
        meta["header_ratio"] = header_ratio

        # Estimate per-page height by max bbox y2.
        page_heights: Dict[int, float] = {}
        for item in mineru_json:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            try:
                y2 = float(bbox[3])
            except Exception:
                continue
            page_idx = item.get("page_idx", 0)
            try:
                page_idx_int = int(page_idx)
            except Exception:
                page_idx_int = 0
            page_heights[page_idx_int] = max(page_heights.get(page_idx_int, 0.0), y2)

        promoted_total = 0
        promoted_headers = 0
        for item in mineru_json:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "discarded":
                continue
            text = (item.get("text") or "").strip()
            if not text:
                continue
            bbox = item.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue

            try:
                y2 = float(bbox[3])
            except Exception:
                continue

            page_idx = item.get("page_idx", 0)
            try:
                page_idx_int = int(page_idx)
            except Exception:
                page_idx_int = 0

            page_height = page_heights.get(page_idx_int, 0.0)

            # Force retention: always promote non-empty discarded blocks to text.
            item["type"] = "text"
            item.setdefault("original_type", "discarded")
            promoted_total += 1

            # Tag zone for downstream use (optional).
            zone = item.get("zone")
            if not zone:
                if page_height > 0:
                    try:
                        y1 = float(bbox[1])
                    except Exception:
                        y1 = None
                    header_y2_threshold = page_height * header_ratio
                    footer_y1_threshold = page_height * (1.0 - header_ratio)
                    if y2 <= header_y2_threshold:
                        zone = "header"
                    elif (y1 is not None) and (y1 >= footer_y1_threshold):
                        zone = "footer"
                    else:
                        zone = "body"
                else:
                    zone = "unknown"
                item["zone"] = zone

            if zone == "header":
                promoted_headers += 1

        if APP_DEBUG:
            print(
                f"[MINERU_HEADER_PLUGIN] promoted_discarded_total={promoted_total} "
                f"promoted_discarded_headers={promoted_headers} ratio={header_ratio}"
            )

        meta["promoted_discarded_total"] = int(promoted_total)
        meta["promoted_discarded_headers"] = int(promoted_headers)
        return mineru_json, meta

    def _merge_pdf_text_layer_header(self, pdf_path: Path, mineru_json: Any) -> Tuple[Any, Dict[str, Any]]:
        meta: Dict[str, Any] = {
            "enabled": bool(MINERU_INCLUDE_PDF_TEXT_LAYER),
            "injected_header_blocks": 0,
            "max_lines": int(MINERU_PDF_TEXT_LAYER_MAX_LINES) if MINERU_PDF_TEXT_LAYER_MAX_LINES else 60,
        }

        if not MINERU_INCLUDE_PDF_TEXT_LAYER:
            return mineru_json, meta

        if not isinstance(mineru_json, list):
            return mineru_json, meta

        try:
            from PyPDF2 import PdfReader
        except Exception:
            return mineru_json, meta

        try:
            reader = PdfReader(str(pdf_path))
        except Exception:
            return mineru_json, meta

        max_lines = meta["max_lines"]
        if max_lines <= 0:
            return mineru_json, meta

        existing_text = "\n".join(
            [
                (it.get("text") or "")
                for it in mineru_json
                if isinstance(it, dict) and (it.get("text") or "").strip()
            ]
        )

        injected = 0
        for page_idx, page in enumerate(reader.pages):
            try:
                text = (page.extract_text() or "").strip()
            except Exception:
                continue

            if not text:
                continue

            # Keep only the top portion (first N non-empty lines) to approximate "header".
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if not lines:
                continue

            header_text = "\n".join(lines[:max_lines]).strip()
            if not header_text:
                continue

            # Avoid injecting if it looks like it's already present.
            if header_text in existing_text:
                continue

            mineru_json.insert(
                0,
                {
                    "type": "text",
                    "text": header_text,
                    "bbox": [0, 0, 0, 0],
                    "page_idx": page_idx,
                    "zone": "header",
                    "original_type": "pdf_text_layer",
                    "source": "pdf_text_layer",
                },
            )
            injected += 1

        if APP_DEBUG and injected:
            print(f"[MINERU_PDF_TEXT_LAYER] injected_header_blocks={injected} max_lines={max_lines}")

        meta["injected_header_blocks"] = int(injected)
        return mineru_json, meta

    @staticmethod
    def _safe_float01(value: Any) -> Optional[float]:
        try:
            v = float(value)
        except Exception:
            return None
        if v != v:  # NaN
            return None
        # Normalize 0-100 into 0-1 if it looks like percent.
        if v > 1.5:
            v = v / 100.0
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        return v

    def _compute_overall_confidence(self, mineru_json: Any) -> Dict[str, Any]:
        """Compute an overall confidence score.

        MinerU CLI JSON does not reliably include per-block confidence. When present (score/confidence/conf),
        we use it. Otherwise we fall back to a simple quality heuristic based on output richness.
        """

        # Prefer explicit confidence/score fields if present.
        values: List[float] = []
        weighted_sum = 0.0
        weight_total = 0.0

        if isinstance(mineru_json, list):
            for it in mineru_json:
                if not isinstance(it, dict):
                    continue
                text = (it.get("text") or "")
                w = float(len(text.strip())) if isinstance(text, str) else 1.0
                if w <= 0:
                    w = 1.0
                for key in ("confidence", "conf", "score"):
                    if key not in it:
                        continue
                    v01 = self._safe_float01(it.get(key))
                    if v01 is None:
                        continue
                    values.append(v01)
                    weighted_sum += v01 * w
                    weight_total += w
                    break

        if values and weight_total > 0:
            overall = weighted_sum / weight_total
            return {
                "overall": float(overall),
                "method": "weighted_block_confidence",
                "samples": int(len(values)),
            }

        # Heuristic fallback: use text richness + bbox presence.
        total_blocks = 0
        text_blocks = 0
        bbox_blocks = 0
        text_chars = 0

        if isinstance(mineru_json, list):
            for it in mineru_json:
                if not isinstance(it, dict):
                    continue
                total_blocks += 1
                t = (it.get("text") or "")
                if isinstance(t, str) and t.strip():
                    text_blocks += 1
                    text_chars += len(t.strip())
                bbox = it.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4:
                    bbox_blocks += 1

        block_ratio = (text_blocks / total_blocks) if total_blocks else 0.0
        bbox_ratio = (bbox_blocks / total_blocks) if total_blocks else 0.0

        # Map avg chars/page proxy into 0-1.
        # Without page count, assume 1 page to keep it conservative.
        page_count = 1
        try:
            max_idx = max(int(it.get("page_idx", 0)) for it in mineru_json if isinstance(it, dict))  # type: ignore[arg-type]
            page_count = max(1, max_idx + 1)
        except Exception:
            page_count = 1

        avg_chars_per_page = float(text_chars) / float(page_count)
        richness = min(1.0, avg_chars_per_page / 300.0)  # 300 chars/page ~ reasonably rich

        overall = 0.10 + 0.50 * richness + 0.20 * block_ratio + 0.20 * bbox_ratio
        if overall < 0.0:
            overall = 0.0
        if overall > 1.0:
            overall = 1.0

        return {
            "overall": float(overall),
            "method": "heuristic",
            "details": {
                "total_blocks": int(total_blocks),
                "text_blocks": int(text_blocks),
                "bbox_blocks": int(bbox_blocks),
                "text_chars": int(text_chars),
                "page_count_proxy": int(page_count),
                "avg_chars_per_page": float(avg_chars_per_page),
                "block_ratio": float(block_ratio),
                "bbox_ratio": float(bbox_ratio),
            },
        }

    def _build_mineru_sidecar(
        self,
        *,
        pdf_path: Path,
        selected_json_file: Path,
        mineru_json: Any,
        json_variant: str,
        pdf_text_layer_meta: Dict[str, Any],
        header_plugin_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        conf = self._compute_overall_confidence(mineru_json)
        # Note: Do NOT embed SourceURL/ProcessedURL into the sidecar.
        # The canonical source/processed hints are stored in `src_dst.txt` and are
        # consumed by CSV generation. Keep sidecar minimal and avoid duplicating
        # source/destination metadata here.
        return {
            "schema": "openminer.mineru_sidecar.v1",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "source_pdf": str(pdf_path),
            # Source/Processed hints intentionally omitted; read `src_dst.txt` instead.
            "selected_json_file": str(selected_json_file),
            "json_variant": str(json_variant or ""),
            "mineru": {
                "method": MINERU_METHOD,
                "output_source": MINERU_OUTPUT_SOURCE,
                "model_name": MINERU_MODEL_NAME,
                "device": MINERU_DEVICE,
            },
            "plugins": {
                "pdf_text_layer": pdf_text_layer_meta,
                "header_zone": header_plugin_meta,
            },
            "confidence": conf,
        }

    @staticmethod
    def _write_mineru_sidecar(*, selected_json_file: Path, sidecar: Dict[str, Any]) -> Path:
        stem = Path(selected_json_file).stem
        filename = f"{stem}_sidecar.json"

        # If a separate sidecar output directory is configured, write sidecars there.
        # Otherwise, keep legacy behavior (write next to the selected JSON).
        if SIDECAR_OUTPUT_DIR is not None:
            sidecar_path = Path(SIDECAR_OUTPUT_DIR) / filename
        else:
            sidecar_path = Path(selected_json_file).with_name(filename)

        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
        return sidecar_path
    
    def _generate_mock_mineru_output(self, pdf_path: Path) -> Dict[str, Any]:
        """Generate mock MinerU output (for tests or when MinerU is unavailable)."""
        return {
            "document": {
                "pages": [
                    {
                        "page_number": 1,
                        "text": "This is extracted text from the PDF (mock data).",
                        "elements": [
                            {
                                "type": "text",
                                "content": "This is extracted text from the PDF (mock data).",
                                "bbox": [100, 100, 500, 200],
                                "font_size": 12,
                                "font_family": "Arial"
                            }
                        ],
                        "tables": [],
                        "figures": []
                    }
                ],
                "total_pages": 1,
                "metadata": {
                    "source_file": str(pdf_path),
                    "processing_method": "mock_ocr"
                }
            }
        }
    
    async def shutdown(self):
        """Shut down AsyncLLM (called when the application is shutting down)."""
        global _async_llm
        if _async_llm is not None:
            try:
                _async_llm.shutdown()
                if APP_DEBUG:
                    print("[MINERU] AsyncLLM shut down")
            except Exception as e:
                if APP_DEBUG:
                    print(f"[MINERU] Error while shutting down AsyncLLM: {e}")
    
    def _find_mineru_pdfs(self, pdf_path: Path) -> Dict[str, Optional[Path]]:
        """
        Find MinerU-generated PDF outputs (_layout.pdf and _span.pdf).
        
        Returns:
            A dict containing PDF paths: {"layout_pdf": Path, "span_pdf": Path}
        """
        pdf_name = pdf_path.stem
        pdf_output_dir = self.output_dir / pdf_name
        
        # Candidate folder names.
        method_to_folder = {
            "auto": "hybrid_auto",
            "ocr": "hybrid_ocr",
            "txt": "hybrid_txt"
        }
        hybrid_folder = method_to_folder.get(MINERU_METHOD, "hybrid_auto")
        
        result = {"layout_pdf": None, "span_pdf": None}
        
        # Check possible folders.
        possible_folders = [hybrid_folder, MINERU_METHOD]
        
        for folder_name in possible_folders:
            folder_path = pdf_output_dir / folder_name
            if folder_path.exists():
                layout_pdf = folder_path / f"{pdf_name}_layout.pdf"
                span_pdf = folder_path / f"{pdf_name}_span.pdf"
                
                if layout_pdf.exists():
                    result["layout_pdf"] = layout_pdf
                if span_pdf.exists():
                    result["span_pdf"] = span_pdf
                
                if result["layout_pdf"] or result["span_pdf"]:
                    break
        
        return result

    def find_selected_json_and_sidecar(
        self,
        pdf_path: Path,
        json_variant: Optional[str] = None,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """Locate the MinerU-selected JSON file and its sidecar for a previously processed PDF.

        Returns (selected_json_path, sidecar_path). Either may be None if not found.
        """
        try:
            pdf_name = Path(pdf_path).stem
            pdf_output_dir = Path(self.output_dir) / pdf_name

            method_to_folder = {
                "auto": "hybrid_auto",
                "ocr": "hybrid_ocr",
                "txt": "hybrid_txt",
            }
            hybrid_folder = method_to_folder.get(MINERU_METHOD, f"hybrid_{MINERU_METHOD}")

            candidates = [pdf_output_dir / hybrid_folder, pdf_output_dir / MINERU_METHOD]
            json_files: List[Path] = []
            for folder in candidates:
                if folder.exists():
                    found = list(folder.glob("*.json"))
                    if found:
                        json_files = found
                        break

            if not json_files and pdf_output_dir.exists():
                json_files = list(pdf_output_dir.rglob("*.json"))

            if not json_files:
                return None, None

            selected = self._select_json_file(json_files, (json_variant or "").strip())
            stem = Path(selected).stem
            sidecar_name = f"{stem}_sidecar.json"

            # Prefer configured sidecar output folder (if any), else legacy adjacent file.
            candidates: List[Path] = []
            if SIDECAR_OUTPUT_DIR is not None:
                candidates.append(Path(SIDECAR_OUTPUT_DIR) / sidecar_name)
            candidates.append(Path(selected).with_name(sidecar_name))

            for c in candidates:
                if c.exists():
                    return selected, c
            return selected, None
        except Exception:
            return None, None