# PDF OCR and Data Extraction App

This is a PDF processing application that uses MinerU 2.5 and Azure AI. It can read PDFs from multiple sources, run OCR, and produce output files.

Documentation lives in [docs/README.md](docs/README.md).

For converting OpenAI JSON dumps (or extracted JSON) to CSV locally, see `scripts/openai_json_to_csv.py` and [docs/JSON_TO_CSV_LOCATION.md](docs/JSON_TO_CSV_LOCATION.md).

## Features

- **PDF Upload**: Upload PDFs directly
- **SharePoint Integration**: Read PDFs from a SharePoint repository
- **Windows Folder**: Read PDFs from a local Windows folder
- **OCR Processing**: Use MinerU 2.5 for OCR
- **Data Extraction**: Generate outputs from MinerU JSON
  - **mineru_json**: Extract directly from MinerU JSON (simple extraction, no AI call; faster and free; legacy: mineru_csv)
  - **azure_openai_json**: Use AI to extract structured data (uses Azure OpenAI or Ollama depending on AI_SERVICE; higher quality; legacy: azure_openai_csv)
  - **program_json**: Deterministic, programmatic extraction (flattens content_list.json into structured data; consistent output; no AI call; legacy: program_csv)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install MinerU 2.5:
```bash
# Please follow the official MinerU documentation for installation.
```

3. Configure environment variables:
Create or edit a `.env` file in the project root and set the following variables.

Important: the `.env` file contains sensitive values and is ignored by `.gitignore`, so it will not be committed.

```
# ============================================================================
# AI service selection
# ============================================================================
AI_SERVICE=azure_openai  # Choose: azure_openai (Azure OpenAI) or ollama (local Ollama)

# ============================================================================
# Azure OpenAI settings (recommended in .env; do not hard-code in code)
# ============================================================================
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_API_VERSION=2023-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_TEMPERATURE=0  # Temperature (0-2). Default 0 to keep formatting consistent

# ============================================================================
# Ollama settings (local model when AI_SERVICE=ollama)
# ============================================================================
OLLAMA_BASE_URL=http://localhost:11434  # Ollama service URL
OLLAMA_MODEL=gemma3:1b  # Model name (e.g., gemma3:1b or deepseek-r1:8b)
OLLAMA_TIMEOUT=30000  # Ollama API timeout (seconds)
OLLAMA_TEMPERATURE=0  # Temperature (0-2). Default 0 to keep formatting consistent

# Custom prompt (optional, for controlling CSV output format)
# Option 1: set prompt text directly (use \n for newlines)
# AZURE_OPENAI_CUSTOM_PROMPT=Extract the following fields from the PDF content: date, amount, description, notes. Return CSV with the header row first.\n\n{content}
# Option 2: set a prompt file path (relative to project root)
# AZURE_OPENAI_CUSTOM_PROMPT_FILE=prompts/csv_extraction.txt
# Note: if you use the {content} placeholder it will be replaced by the PDF content; otherwise the PDF content will be appended to the prompt.

# ============================================================================
# SharePoint settings (optional)
# ============================================================================
SHAREPOINT_SITE_URL=your_sharepoint_site_url
SHAREPOINT_CLIENT_ID=your_client_id
SHAREPOINT_CLIENT_SECRET=your_client_secret
SHAREPOINT_TENANT_ID=your_tenant_id
\
# Notes:
# - SharePoint integration uses Microsoft Graph API with client-credentials (app-only).
# - Ensure your Entra ID app has admin-consented Application permissions, typically:
#   - Sites.Read.All (browse/list)
#   - Files.Read.All (download)
#   Some tenants may require broader access depending on your policy.

# ============================================================================
# App settings
# ============================================================================
DEBUG=False          # Set True to enable debug logs
FORCE_OCR=False      # Set True to force OCR for all PDFs (skip checks)
# Output folders (optional; defaults shown)
# Use relative paths to keep the project portable.
UPLOAD_DIR=uploads
JSON_OUTPUT_DIR=outputs
CSV_OUTPUT_DIR=outputs  # Optional separate folder for CSV/table outputs
MINERU_OUTPUT_DIR=mineru_outputs
SHAREPOINT_DOWNLOAD_DIR=sharepoint_downloads
STATIC_DIR=static
# CSV generation is deprecated; outputs are JSON by default. Use `generate_csv=true` in requests to opt-in to CSV output.
# CSV generation methods:
# - mineru_csv: extract directly from MinerU JSON (simple extraction, no AI)
# - azure_openai_csv: use AI to extract structured data (Azure OpenAI or Ollama depending on AI_SERVICE)
# - program_csv: deterministic flattening of content_list.json into CSV
JSON_GENERATION_METHOD=azure_openai_json  # Default to Azure OpenAI (legacy: azure_openai_csv)

# ============================================================================
# MinerU settings (optional; defaults are provided)
# ============================================================================
MINERU_CMD=mineru
MINERU_METHOD=auto
USE_VLLM_ASYNC=True
MINERU_MODEL_NAME=opendatalab/MinerU2.5-2509-1.2B
MINERU_DEVICE=cuda
# Initialize MinerU at app startup:
# - True (default): preload models on startup; faster first request
# - False: lazy init on first use; faster startup
INIT_MINERU_ON_STARTUP=True
# MinerU output source selection (for CSV generation):
# - json (default): use _content_list.json
# - layout_pdf: use _layout.pdf (includes layout info)
# - span_pdf: use _span.pdf (includes span info)
# - both_pdf: use both PDFs (if present)
MINERU_OUTPUT_SOURCE=json

## New Settings (Important)

This project now prefers environment variables and frontend prompt overrides, and no longer requires `prompts/csv_extraction.txt`.

- New environment variables (can be set in `.env`):
  - `AZURE_OPENAI_SYSTEM_PROMPT`: override the system prompt in `.env` (frontend input takes precedence).
  - `AZURE_OPENAI_USER_PROMPT`: override the user prompt in `.env` (frontend input takes precedence).

- Removed requirement: `AZURE_OPENAI_CUSTOM_PROMPT_FILE` / `prompts/csv_extraction.txt` are no longer mandatory. If the file exists, it can be safely deleted or ignored. Prompt resolution order: frontend input > `.env` values > built-in defaults.

- The frontend supports entering `System Prompt` and `User Prompt` directly in the UI. If the frontend does not provide prompts, the backend uses values from `.env`.

### JSON Support

The backend now outputs JSON (array of objects). The frontend and `.env` no longer expose a CSV option.

### Testing and Quick Verification (Recommended)

If your local environment is unstable due to MinerU/vLLM initialization, you can temporarily disable preloading MinerU during local testing to speed up startup and validation.

1. Edit `.env` in the project root:

```powershell
INIT_MINERU_ON_STARTUP=False
```

2. Activate the conda environment (this project uses `mineru2.5`):

```powershell
conda activate mineru2.5
```

3. Start the app (example uses port 8001 to avoid port 8000 conflicts):

```powershell
uvicorn main:app --reload --port 8001
```

4. Test an upload using curl or PowerShell (PowerShell example):

```powershell
curl -F "file=@C:\path\to\test_sample.pdf" `
  -F "system_prompt=Return JSON with fields: date, amount, description" `
  -F "user_prompt=Extract the fields from the following content" `
  http://127.0.0.1:8001/api/upload
```

After processing, the API returns a download URL or a file path (depending on the UI/settings).

```

Notes:
- All Azure OpenAI parameters should live in `.env`, especially `AZURE_OPENAI_API_KEY` (sensitive)
- AI service selection:
  - `AI_SERVICE=azure_openai`: use Azure OpenAI (requires Azure OpenAI credentials)
  - `AI_SERVICE=ollama`: use local Ollama (requires Ollama installed and running)
    - You can choose a model via `OLLAMA_MODEL`: `gemma3:1b` or `deepseek-r1:8b` (default: `gemma3:1b`)
- Generation/extraction methods (JSON output; prefer *_json names; legacy *_csv still supported):
  - `JSON_GENERATION_METHOD=mineru_json`: extract directly from MinerU JSON (no AI; faster and free)
  - `JSON_GENERATION_METHOD=azure_openai_json`: AI-based structured extraction (Azure OpenAI or Ollama depending on `AI_SERVICE`)
  - `JSON_GENERATION_METHOD=program_json`: deterministic programmatic extraction (no AI)
- If AI is not configured and `JSON_GENERATION_METHOD=azure_openai_json`, the app falls back to `mineru_json`
- All parameters can be set in `.env`, or you can see defaults in `config.py`

### Custom CSV Format (Optional)

You can control the CSV output format by setting a custom prompt.

Option 1: use an environment variable (good for short prompts)
```env
AZURE_OPENAI_CUSTOM_PROMPT=Extract the following fields from the PDF content: date, amount, description. Return CSV with the header row first.\n\n{content}
```

Option 2: use a prompt file (recommended for complex prompts)
1. Create `prompts/csv_extraction.txt`
2. Write your prompt in the file (you can use `{content}` as a placeholder)
3. Set in `.env`:
```env
AZURE_OPENAI_CUSTOM_PROMPT_FILE=prompts/csv_extraction.txt
```

Prompt examples:
- `prompts/example_csv_extraction.txt` - basic example
- `prompts/example_multi_placeholder.txt` - multiple built-in placeholders
- `prompts/example_semantic_placeholders.txt` - semantic placeholders (recommended)

Supported placeholders:

Built-in placeholders (replaced with actual content):
- `{content}` - full PDF text content (all pages merged)
- `{tables}` - all table content
- `{page_1}`, `{page_2}`, `{page_3}`, ... - content from specific pages
  - Example: `{page_1}` is page 1 content
  - Example: `{page_2}` is page 2 content

Semantic placeholders (interpreted by the model):
You can use any custom semantic placeholder, for example:
- `{invoice date}` - invoice date
- `{invoice number}` - invoice number
- `{company name}` - company name
- `{amount}` - amount
- `{description}` - description
- or any field name you need

Semantic placeholders remain in the prompt so the model can infer and extract the corresponding information from `{content}`.

Placeholder usage example:
```
Extract invoice information from the following content:

Page 1 (often includes the title):
{page_1}

Full content:
{content}

Tables:
{tables}
```

Notes:
- If the prompt contains placeholders, they will be replaced with the corresponding content
- If the prompt contains no placeholders, the full PDF content will be appended to the prompt
- If a placeholder has no corresponding content (e.g., PDF has only 2 pages but you use `{page_5}`), the placeholder remains as-is

If no custom prompt is configured, the default prompt will be used.

## Run

Using the startup script:
```bash
python run.py
```

Or run uvicorn directly:
```bash
uvicorn main:app --reload
```

The app starts at `http://localhost:8000`.

## Docker (Recommended for deployment)

This project can be deployed in a reproducible way using Docker. Since vLLM mainly depends on NVIDIA/CUDA, this repo provides a CPU-default compose configuration to avoid failures on machines without NVIDIA.

### Option A: Web/UI only (default, lightest)

```bash
docker compose up --build
```

Then open `http://localhost:8000`.

Note: this mode may not include the real MinerU OCR CLI (depending on whether you installed the MinerU CLI during build). If the container does not have the `mineru` command, the app may run the fallback/simulated path (UI/API still runs).

### Option B: Include MinerU CLI in the image (more complete, heavier build)

Set `INSTALL_MINERU` to `"1"` in `docker-compose.yml`, then rebuild:

```bash
docker compose up --build
```

### Common environment variables (override via compose `environment` or `.env`)

- `USE_VLLM_ASYNC=False`: CPU default (avoid accidental vLLM usage)
- `MINERU_DEVICE=cpu`
- `JSON_GENERATION_METHOD=mineru_json | azure_openai_json | program_json` (legacy: `mineru_csv | azure_openai_csv | program_csv`)
- `AI_SERVICE=azure_openai | ollama`

## Usage

### 1. Upload a PDF
- Open the home page
- Click the "Upload" tab
- Select or drag-and-drop a PDF
- Click the "Process PDF" button
- Wait for processing to complete, then download the output

### 2. Process from SharePoint
- Click the "SharePoint" tab
- Enter the SharePoint file URL or path
- (Optional) Enter a folder path
- Click the "Process SharePoint PDF" button

### 3. Process from a Windows folder
- Click the "Windows Folder" tab
- Enter the folder path (e.g., `C:\Documents\PDFs`)
- Click the "Process All PDFs in Folder" button
- The system will process all PDFs in the folder

## API Endpoints

- `GET /` - Home page (web UI)
- `POST /api/upload` - Upload a PDF or JSON file
  - Request: multipart/form-data
    - `file` (required): the uploaded PDF or JSON file
    - `generate_csv` (optional): whether to generate CSV
      - `"true"`, `"1"`, `"yes"`, `"on"` enable
      - `"false"`, `"0"`, `"no"`, `"off"` disable
      - If omitted, CSV is disabled by default (use `generate_csv=true` to enable)
  - Response: JSON including `file_id`, `filename`, `download_url`, etc.
- `POST /api/process-sharepoint` - Process a PDF from SharePoint
  - Request: form-data
    - `file_url` (required): SharePoint file URL or relative path
    - `folder_path` (optional): SharePoint folder path
    - `generate_csv` (optional): whether to generate CSV (same as above)
  - Response: JSON with the processing result
- `POST /api/process-folder` - Process PDFs from a Windows folder
  - Request: form-data
    - `folder_path` (required): Windows folder path
    - `generate_csv` (optional): whether to generate CSV (same as above)
  - Response: JSON list of all processing results
- `GET /api/download/{filename}` - Download the generated output (prefers JSON; returns JSON if present, otherwise CSV)
  - Parameter: `filename` is the CSV filename (without extension)
  - Response: CSV download

## Processing Flow

1. PDF upload/read: get the PDF from the selected source
2. OCR check: determine whether OCR is needed
3. OCR (if needed): run MinerU 2.5 OCR and generate JSON output
4. Data extraction: use Azure OpenAI to analyze the JSON and extract structured data
5. CSV generation: convert extracted data to CSV
6. Download: provide a download link

## Project Structure

```
mineruocr/
├── main.py                 # Main application
├── config.py               # Configuration
├── run.py                  # Startup script
├── requirements.txt        # Python dependencies
├── README.md               # Documentation
├── .env.example            # Environment variable example
├── docs/                   # Docs (installation / quickstart / troubleshooting)
│   ├── INSTALLATION.md
│   ├── QUICKSTART.md
│   ├── MINERU_VLLM_SETUP.md
│   ├── SETUP_STATUS.md
│   └── JSON_TO_CSV_LOCATION.md
├── services/               # Service modules
│   ├── pdf_processor.py    # PDF processing service
│   ├── mineru_service.py   # MinerU integration service
│   ├── azure_ai_service.py # AI service
│   ├── sharepoint_service.py # SharePoint service
│   └── folder_service.py   # Folder service
├── templates/              # HTML templates
│   └── index.html          # Home page
├── uploads/                # Uploads
├── outputs/                # Output directory
├── mineru_outputs/         # MinerU outputs
└── sharepoint_downloads/   # SharePoint downloads
```

## Tests

Run a test script:
```bash
python tests/test_app.py
```

## Notes

- Ensure MinerU 2.5 is installed and available in your shell
- Azure OpenAI requires a valid API key and endpoint
- SharePoint features require an Azure AD app registration and permissions
- Large PDFs may take a long time to process
- CSV output format may vary based on the PDF content

## Troubleshooting

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for troubleshooting guidance.

