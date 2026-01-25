# Installation Guide

## Quick Start (5 Minutes)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

At minimum, set these in `.env`:
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_DEPLOYMENT_NAME`

### 3. Start the application

```bash
python run.py
```

### 4. Test

1. Open `http://localhost:8000` in your browser
2. Upload a PDF
3. Wait for processing to finish
4. Download the output

---

## System Requirements

- Python 3.8 or newer
- Windows 10/11 (for the Windows folder feature)
- MinerU 2.5 (for OCR)
- An Azure OpenAI account (for AI-based data extraction)

## Detailed Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install MinerU 2.5

Please follow the official MinerU documentation:

1. Visit the MinerU GitHub repository
2. Follow the official installation guide for MinerU 2.5
3. Ensure the `mineru` command is available in your shell

If MinerU is not installed, the application can run using simulated data for testing.

### Step 3: Configure Environment Variables

Edit `.env` and fill in your credentials:

```env
# Azure OpenAI settings
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_API_VERSION=2023-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4

# SharePoint settings (optional)
SHAREPOINT_SITE_URL=https://yourtenant.sharepoint.com/sites/yoursite
SHAREPOINT_CLIENT_ID=your_client_id
SHAREPOINT_CLIENT_SECRET=your_client_secret
SHAREPOINT_TENANT_ID=your_tenant_id

# MinerU settings (optional)
MINERU_CMD=mineru

# App settings (optional)
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=False
```

**SharePoint Notes:**
- The app uses Microsoft Graph API with client-credentials (app-only).
- In your Entra ID app registration, grant admin-consented Application permissions:
  - `Sites.Read.All` (browse/list libraries and folders)
  - `Files.Read.All` (download files)

### Step 4: Install Additional Dependencies (If Needed)

#### PDF to Image (for OCR)

If you use `pdf2image`, you need Poppler.

**Windows:**
- Download Poppler for Windows
- Add Poppler's `bin` directory to your system PATH

**Linux:**
```bash
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

### Step 5: Start the Application

```bash
python run.py
```

Or start directly with uvicorn:

```bash
uvicorn main:app --reload
```

The app will be available at `http://localhost:8000`.

---

## Feature Testing

### Test Upload
- Prepare a PDF
- In the UI, click the "Upload" tab
- Drag-and-drop or select the PDF
- Click "Process"

### Test Windows Folder
- Prepare a folder containing PDFs
- In the UI, click the "Windows Folder" tab
- Enter a folder path (e.g., `C:\Users\YourName\Documents\PDFs`)
- Click "Process All PDFs in Folder"

### Test SharePoint (requires setup)
- Ensure SharePoint environment variables are configured
- In the UI, click the "SharePoint" tab
- Enter a SharePoint file path
- Click "Process SharePoint PDF"

---

## Troubleshooting

### MinerU Not Found

If you see a warning like "MinerU processing failed":
- Verify MinerU is installed correctly
- Verify the `mineru` command runs from your shell
- Or set the `MINERU_CMD` environment variable to the correct path

### Azure OpenAI Errors

If you hit Azure OpenAI related errors:
- Verify the API key is correct
- Verify the endpoint URL is correct
- Verify the deployment name is correct
- Verify the API version is supported

### SharePoint Connectivity Issues

If SharePoint features do not work:
- Confirm your app is registered in Azure AD
- Verify client ID, secret, and tenant ID
- Confirm the app has the required Microsoft Graph Application permissions (and admin consent)

### FAQ

**Q: The app won't start?**
A: Verify all dependencies are installed and Python >= 3.8.

**Q: Azure AI processing failed?**
A: Check your Azure OpenAI settings in `.env`.

**Q: MinerU processing failed?**
A: This can be expected if MinerU is not installed; the system uses simulated data. To use real OCR, install MinerU 2.5.

**Q: Can't download output?**
A: Ensure the `outputs/` directory exists and the app has write permissions.
