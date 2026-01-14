# Installation Guide

## System Requirements

- Python 3.8 or newer
- Windows 10/11 (for the Windows folder feature)
- MinerU 2.5 (for OCR)
- An Azure OpenAI account (for AI-based data extraction)

## Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Install MinerU 2.5

Please follow the official MinerU documentation:

1. Visit the MinerU GitHub repository
2. Follow the official installation guide for MinerU 2.5
3. Ensure the `mineru` command is available in your shell

If MinerU is not installed, the application can run using simulated data for testing.

## Step 3: Configure Environment Variables

1. Copy `.env.example` to `.env`
2. Edit `.env` and fill in your Azure OpenAI and SharePoint credentials:

```env
# Azure OpenAI settings
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_API_VERSION=2023-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4

# SharePoint settings (optional; omit if you do not use SharePoint)
SHAREPOINT_SITE_URL=https://yourtenant.sharepoint.com/sites/yoursite
SHAREPOINT_CLIENT_ID=your_client_id
SHAREPOINT_CLIENT_SECRET=your_client_secret
SHAREPOINT_TENANT_ID=your_tenant_id

# SharePoint (Microsoft Graph) notes:
# - The app uses Microsoft Graph API with client-credentials (app-only).
# - In your Entra ID app registration, grant admin-consented Application permissions such as:
#   - Sites.Read.All (browse/list libraries and folders)
#   - Files.Read.All (download files)
# - If your tenant restricts app-only access to SharePoint, you may need additional tenant policy changes.

# MinerU settings (optional)
MINERU_CMD=mineru

# App settings (optional)
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=False
```

## Step 4: Install Additional Dependencies (If Needed)

### PDF to Image (for OCR)

If you use `pdf2image`, you need Poppler.

Windows:
- Download Poppler for Windows
- Add Poppler's `bin` directory to your system PATH

Linux:
```bash
sudo apt-get install poppler-utils
```

macOS:
```bash
brew install poppler
```

## Step 5: Start the Application

```bash
python run.py
```

Or start directly with uvicorn:

```bash
uvicorn main:app --reload
```

The app will be available at `http://localhost:8000`.

## Verify Installation

1. Open `http://localhost:8000` in your browser
2. Try uploading a PDF
3. Confirm it can process successfully and download the output

## Troubleshooting

### MinerU Not Found

If you see a warning like "MinerU processing failed", make sure:
- MinerU is installed correctly
- The `mineru` command runs from your shell
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

