"""Application configuration.

Configuration priority:
1. Environment variables (.env file or system environment variables)
2. Defaults defined in config.py

This allows you to:
- Set sane defaults in config.py (convenient for development)
- Override via .env (recommended for production and secrets)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file (if present)
# Prefer .env values over any existing environment variables so
# edits to .env take effect immediately (and later duplicates win).
load_dotenv(override=True)

def _env_clean(value: str) -> str:
    """Clean a value read from .env: remove trailing comments and extra whitespace/quotes."""
    if value is None:
        return ""
    # Supports: FOO=123 # comment
    value = value.split("#", 1)[0].strip()
    # Remove common quote wrappers
    if (len(value) >= 2) and ((value[0] == value[-1]) and value[0] in ("'", '"')):
        value = value[1:-1].strip()
    return value


def _decode_dotenv_newlines(value: str) -> str:
    """Allow storing prompts as a single-line string in .env, using \n as newline."""
    if not value:
        return ""
    # Normalize CRLF escapes first, then decode \n.
    return value.replace("\\r\\n", "\\n").replace("\\n", "\n")

def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name, None)
    if raw is None:
        return default
    cleaned = _env_clean(raw)
    return cleaned if cleaned != "" else default

def _env_bool(name: str, default: bool) -> bool:
    cleaned = _env_clean(os.getenv(name, "true" if default else "false")).lower()
    if cleaned in ("1", "true", "yes", "y", "on"):
        return True
    if cleaned in ("0", "false", "no", "n", "off"):
        return False
    return default

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, None)
    if raw is None:
        return default
    cleaned = _env_clean(raw)
    if cleaned == "":
        return default
    return float(cleaned)

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, None)
    if raw is None:
        return default
    cleaned = _env_clean(raw)
    if cleaned == "":
        return default
    return int(cleaned)

def _env_stripped_lower(name: str, default: str) -> str:
    return _env_str(name, default).strip().lower()

def _env_path(name: str, default: Path, base_dir: Path) -> Path:
    """Read a directory path from env.

    - Empty/unset -> default
    - Relative -> resolved relative to base_dir
    - Absolute -> used as-is
    """
    raw = os.getenv(name, None)
    if raw is None:
        return default
    cleaned = _env_clean(raw)
    if cleaned == "":
        return default
    try:
        p = Path(cleaned).expanduser()
    except Exception:
        return default
    return p if p.is_absolute() else (base_dir / p)


def _env_optional_path(name: str, base_dir: Path) -> Path | None:
    """Read an optional directory path from env.

    - Unset/empty -> None
    - Relative -> resolved relative to base_dir
    - Absolute -> used as-is
    """
    raw = os.getenv(name, None)
    if raw is None:
        return None
    cleaned = _env_clean(raw)
    if cleaned == "":
        return None
    try:
        p = Path(cleaned).expanduser()
    except Exception:
        return None
    return p if p.is_absolute() else (base_dir / p)


def _dotenv_last_assignment(dotenv_path: Path, key: str) -> str | None:
    """Return the last KEY=VALUE assignment from a .env file.

    This is a robustness fallback for cases where the dotenv loader
    gets tripped up by malformed lines like a bare 'KEY' with no '='.
    """
    try:
        text = dotenv_path.read_text(encoding="utf-8")
    except Exception:
        return None
    last: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        k, v = stripped.split("=", 1)
        if k.strip() != key:
            continue
        last = _env_clean(v)
    return last


# Directory settings
BASE_DIR = Path(__file__).parent

# All folders can be overridden via .env.
# Tip: use relative paths to keep the project portable (resolved relative to BASE_DIR).
UPLOAD_DIR = _env_path("UPLOAD_DIR", BASE_DIR / "uploads", BASE_DIR)

# JSON output folder (new preferred name).
# Backward compatible: if JSON_OUTPUT_DIR is not set, OUTPUT_DIR is used.
_legacy_output_dir = _env_path("OUTPUT_DIR", BASE_DIR / "outputs", BASE_DIR)
JSON_OUTPUT_DIR = _env_path("JSON_OUTPUT_DIR", _legacy_output_dir, BASE_DIR)

# Backward-compatible alias: existing code may still import OUTPUT_DIR.
OUTPUT_DIR = JSON_OUTPUT_DIR

# Optional separate folder for CSV outputs (backward compatibility / table outputs).
# Defaults to JSON_OUTPUT_DIR so existing behavior stays unchanged.
CSV_OUTPUT_DIR = _env_path("CSV_OUTPUT_DIR", JSON_OUTPUT_DIR, BASE_DIR)

# Optional separate folder for MinerU sidecar JSON outputs.
# If unset/empty, sidecars may be written next to the selected MinerU JSON (legacy behavior).
SIDECAR_OUTPUT_DIR = _env_optional_path("SIDECAR_OUTPUT_DIR", BASE_DIR)
if SIDECAR_OUTPUT_DIR is None:
    _v = _dotenv_last_assignment(BASE_DIR / ".env", "SIDECAR_OUTPUT_DIR")
    if _v:
        try:
            _p = Path(_v).expanduser()
            SIDECAR_OUTPUT_DIR = _p if _p.is_absolute() else (BASE_DIR / _p)
        except Exception:
            SIDECAR_OUTPUT_DIR = None

# Whether to generate a template-based CSV (templates/template.csv) automatically
# after JSON extraction.
GENERATE_TEMPLATE_CSV = _env_bool("GENERATE_TEMPLATE_CSV", False)

MINERU_OUTPUT_DIR = _env_path("MINERU_OUTPUT_DIR", BASE_DIR / "mineru_outputs", BASE_DIR)
SHAREPOINT_DOWNLOAD_DIR = _env_path("SHAREPOINT_DOWNLOAD_DIR", BASE_DIR / "sharepoint_downloads", BASE_DIR)
STATIC_DIR = _env_path("STATIC_DIR", BASE_DIR / "static", BASE_DIR)

# Searchable PDF output folder (PDFs with an embedded text layer)
SEARCHABLE_PDF_OUTPUT_DIR = _env_path("SEARCHABLE_PDF_OUTPUT_DIR", JSON_OUTPUT_DIR, BASE_DIR)

# Searchable PDF text layer behavior
# Some PDF viewers behave inconsistently with truly "invisible" text.
# Default uses a near-transparent overlay to improve search reliability.
SEARCHABLE_PDF_TEXT_OPACITY = float(_env_str("SEARCHABLE_PDF_TEXT_OPACITY", "0.01"))

# Optional: embed a specific font to improve Unicode search/copy in some viewers.
# Examples:
#   Windows: C:\\Windows\\Fonts\\arial.ttf
#   Linux: /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
SEARCHABLE_PDF_FONT_PATH = _env_optional_path("SEARCHABLE_PDF_FONT_PATH", BASE_DIR)

# Create directories
for dir_path in [UPLOAD_DIR, JSON_OUTPUT_DIR, CSV_OUTPUT_DIR, MINERU_OUTPUT_DIR, SHAREPOINT_DOWNLOAD_DIR, STATIC_DIR, SEARCHABLE_PDF_OUTPUT_DIR]:
    dir_path.mkdir(exist_ok=True)

if SIDECAR_OUTPUT_DIR is not None:
    Path(SIDECAR_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================================
# Configuration (edit defaults here, or override via .env)
# ============================================================================

# AI service selection: azure_openai or ollama
AI_SERVICE = _env_str("AI_SERVICE", "azure_openai").lower()  # Default: Azure OpenAI

# Azure OpenAI settings
# Note: For secrets like API keys, use a .env file and do not hardcode them here.
AZURE_OPENAI_ENDPOINT = _env_str("AZURE_OPENAI_ENDPOINT", "")  # Default: empty
AZURE_OPENAI_API_KEY = _env_str("AZURE_OPENAI_API_KEY", "")  # Default: empty
AZURE_OPENAI_API_VERSION = _env_str("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
AZURE_OPENAI_DEPLOYMENT_NAME = _env_str("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")

# Ollama settings (local model)
OLLAMA_BASE_URL = _env_str("OLLAMA_BASE_URL", "http://localhost:11434")  # Ollama base URL
OLLAMA_MODEL = _env_str("OLLAMA_MODEL", "gemma3:1b")  # Ollama model name
OLLAMA_TIMEOUT = _env_int("OLLAMA_TIMEOUT", 30000)  # Ollama API timeout (seconds), default 30000 (~8.3 hours)

# Model parameter settings
# temperature controls output randomness (0-2):
# - 0: most deterministic/consistent (recommended for structured extraction)
# - 1: balanced creativity/consistency (typical default)
# - 2: most random/creative
AZURE_OPENAI_TEMPERATURE = _env_float("AZURE_OPENAI_TEMPERATURE", 0.0)  # Default 0 for consistent formatting
OLLAMA_TEMPERATURE = _env_float("OLLAMA_TEMPERATURE", 0.0)  # Ollama temperature, default 0

# Azure OpenAI custom prompt settings
# You can set a custom prompt to control output formatting.
# Method 1: set prompt text directly (use \n for newlines)
# Method 2: prompt file paths are disabled; use .env variables below or override via frontend
# If unset, the default prompt is used.
AZURE_OPENAI_CUSTOM_PROMPT = _env_str("AZURE_OPENAI_CUSTOM_PROMPT", "")  # Default: empty
# Default system/user prompts can be set in .env (can be overridden by frontend-provided prompts)
AZURE_OPENAI_SYSTEM_PROMPT = _decode_dotenv_newlines(_env_str("AZURE_OPENAI_SYSTEM_PROMPT", ""))
AZURE_OPENAI_USER_PROMPT = _decode_dotenv_newlines(_env_str("AZURE_OPENAI_USER_PROMPT", ""))

# Azure OpenAI multimodal options
# If enabled, PDF processing will send BOTH:
# - the prompt text (including raw MinerU JSON when enabled)
# - all PDF page images (as image_url data URLs)
AZURE_OPENAI_INCLUDE_IMAGES = _env_bool("AZURE_OPENAI_INCLUDE_IMAGES", True)
# 0 means "no limit" (send all pages). Set e.g. 5 to send only first 5 pages.
AZURE_OPENAI_IMAGE_MAX_PAGES = _env_int("AZURE_OPENAI_IMAGE_MAX_PAGES", 0)
# Resize images so the longest side is at most this many pixels (0 disables resizing).
AZURE_OPENAI_IMAGE_MAX_SIDE = _env_int("AZURE_OPENAI_IMAGE_MAX_SIDE", 1280)
# Image format for data URLs: jpeg or png
AZURE_OPENAI_IMAGE_FORMAT = _env_stripped_lower("AZURE_OPENAI_IMAGE_FORMAT", "jpeg")

# Whether to append the full raw MinerU JSON into the user prompt.
AZURE_OPENAI_INCLUDE_RAW_MINERU_JSON = _env_bool("AZURE_OPENAI_INCLUDE_RAW_MINERU_JSON", True)
# 0 means "no limit". If >0, raw JSON will be truncated to this many characters.
AZURE_OPENAI_RAW_JSON_MAX_CHARS = _env_int("AZURE_OPENAI_RAW_JSON_MAX_CHARS", 0)

# Note: Output format is fixed to JSON (frontend/env no longer offers a CSV option)

# SharePoint settings
SHAREPOINT_SITE_URL = _env_str("SHAREPOINT_SITE_URL", "")
SHAREPOINT_CLIENT_ID = _env_str("SHAREPOINT_CLIENT_ID", "")
SHAREPOINT_CLIENT_SECRET = _env_str("SHAREPOINT_CLIENT_SECRET", "")
SHAREPOINT_TENANT_ID = _env_str("SHAREPOINT_TENANT_ID", "")

# MinerU settings
MINERU_CMD = _env_str("MINERU_CMD", "mineru")
MINERU_METHOD = _env_str("MINERU_METHOD", "auto")  # Options: auto, txt, ocr
# Note: MinerU creates subfolders based on the -m option:
# -m auto -> hybrid_auto/
# -m ocr  -> hybrid_ocr/
# -m txt  -> hybrid_txt/

# MinerU output selection: JSON or PDF
# Options: json (_content_list.json, default), layout_pdf (_layout.pdf), span_pdf (_span.pdf), both_pdf (both PDFs)
MINERU_OUTPUT_SOURCE = _env_str("MINERU_OUTPUT_SOURCE", "json").lower()  # Default: JSON

# MinerU JSON file variant selection (when MINERU_OUTPUT_SOURCE=json)
# Options:
# - content_list (default) -> *_content_list.json
# - middle -> *_middle.json
# - model -> *_model.json
# You can also pass explicit suffixes like _content_list.json / _middle.json / _model.json.
MINERU_JSON_VARIANT = _env_str("MINERU_JSON_VARIANT", "content_list").lower()

# MinerU header zone handling: some layouts mark headers/footers as discarded; this option converts discarded blocks
# in the header zone to text to avoid losing important content downstream.
MINERU_INCLUDE_HEADER_ZONES = _env_bool("MINERU_INCLUDE_HEADER_ZONES", True)
# Header zone as a fraction of page height (estimated via bbox y2/max_y2). For example, 0.12 means top 12%.
MINERU_HEADER_ZONE_RATIO = _env_float("MINERU_HEADER_ZONE_RATIO", 0.12)

# MinerU can sometimes miss selectable PDF text (text layer), especially when a page is classified as a table.
# This option additionally extracts the first few lines from the PDF text layer and merges them into a header
# text block to avoid losing document headers.
MINERU_INCLUDE_PDF_TEXT_LAYER = _env_bool("MINERU_INCLUDE_PDF_TEXT_LAYER", True)
# Note: Some PDFs split each character into its own line in the text layer, producing many lines.
# Keep this default reasonably high to avoid missing headers.
MINERU_PDF_TEXT_LAYER_MAX_LINES = _env_int("MINERU_PDF_TEXT_LAYER_MAX_LINES", 300)

# MinerU vLLM settings (using vllm-async-engine)
USE_VLLM_ASYNC = _env_bool("USE_VLLM_ASYNC", True)  # Whether to use vllm-async-engine
MINERU_MODEL_NAME = _env_str("MINERU_MODEL_NAME", "opendatalab/MinerU2.5-2509-1.2B")
MINERU_DEVICE = _env_str("MINERU_DEVICE", "cuda")  # cuda, cpu, cuda:0, etc.
# Whether to initialize MinerU on application startup (if False, it initializes on first use)
INIT_MINERU_ON_STARTUP = _env_bool("INIT_MINERU_ON_STARTUP", True)  # Default True (preload model)

# Application settings
APP_HOST = _env_str("APP_HOST", "localhost")
APP_PORT = _env_int("APP_PORT", 8000)
DEBUG = _env_bool("DEBUG", False)  # Default False
FORCE_OCR = _env_bool("FORCE_OCR", False)  # Default False

# Whether to generate a searchable PDF (invisible text layer) when OCR is performed.
# If False, OCR output is still produced as JSON, but the PDF is not rewritten.
GENERATE_SEARCHABLE_PDF = _env_bool("GENERATE_SEARCHABLE_PDF", True)

# Searchable PDF generation (OCRmyPDF) settings
# These control how `services/ocr_app.py` is invoked from the main workflow.
OCR_LANGUAGE = _env_str("OCR_LANGUAGE", "eng")
OCR_ROTATE_PAGES = _env_bool("OCR_ROTATE_PAGES", True)
OCR_DESKEW = _env_bool("OCR_DESKEW", False)
OCR_JOBS = _env_int("OCR_JOBS", 1)
OCR_OPTIMIZE = _env_int("OCR_OPTIMIZE", 0)
# Allow forcing OCRmyPDF even if the PDF already has text (in addition to FORCE_OCR)
OCR_FORCE_REDO = _env_bool("OCR_FORCE_REDO", False)

# Optional: explicitly point to a tessdata directory (or its parent) when Tesseract
# can't find configs like 'hocr'/'txt' on Windows.
OCR_TESSDATA_PREFIX = _env_str("OCR_TESSDATA_PREFIX", "")
# Backend concurrency/queue settings: default to a single worker processing one job at a time
# to avoid running multiple MinerU/AI jobs in parallel and exhausting resources.
MAX_CONCURRENT_JOBS = _env_int("MAX_CONCURRENT_JOBS", 1)
# Whether to allow returning mocked MinerU/OCR output when MinerU fails (testing only; disable in production)
ALLOW_MOCK_MINERU_OUTPUT = _env_bool("ALLOW_MOCK_MINERU_OUTPUT", False)
# JSON generation/extraction method (historical names still use *_csv):
# - mineru_csv: extract directly from MinerU JSON (no AI)
# - azure_openai_csv: AI-based extraction
# - program_csv: programmatic flattening extraction
# Prefer JSON_GENERATION_METHOD; fall back to legacy CSV_GENERATION_METHOD if unset (backward compatible).
JSON_GENERATION_METHOD = _env_str("JSON_GENERATION_METHOD", "").lower()
if not JSON_GENERATION_METHOD:
    JSON_GENERATION_METHOD = _env_str("CSV_GENERATION_METHOD", "azure_openai_csv").lower()

# Normalize method value names: allow *_json while keeping internal legacy *_csv handling.
_METHOD_ALIASES = {
    # new preferred names
    "mineru_json": "mineru_csv",
    "azure_openai_json": "azure_openai_csv",
    "program_json": "program_csv",
    # legacy/older names
    "simple": "mineru_csv",
    "azure_openai": "azure_openai_csv",
}
JSON_GENERATION_METHOD = _METHOD_ALIASES.get(JSON_GENERATION_METHOD, JSON_GENERATION_METHOD)

# Backward-compat alias: older code/env may still reference this.
CSV_GENERATION_METHOD = JSON_GENERATION_METHOD

# =========================================================================
# Usage:
#
# Method 1: edit defaults directly in config.py
#   Example: DEBUG = True
#
# Method 2: use a .env file (recommended for secrets)
#   Create a .env file with:
#   DEBUG=True
#   FORCE_OCR=False
#   AZURE_OPENAI_API_KEY=your_key_here
#
# Method 3: use system environment variables
#   export DEBUG=True  (Linux/Mac)
#   set DEBUG=True     (Windows)
# ============================================================================

