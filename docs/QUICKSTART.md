# Quickstart Guide

## 5-Minute Setup

### 1. Install dependencies (about 2 minutes)

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables (about 1 minute)

Copy `.env.example` to `.env` and fill in your Azure OpenAI settings:

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

At minimum, set:
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_DEPLOYMENT_NAME`

### 3. Start the application (about 30 seconds)

```bash
python run.py
```

### 4. Test the application (about 1 minute)

1. Open `http://localhost:8000` in your browser
2. Upload a PDF
3. Wait for processing to finish
4. Download the output

## Feature Checks

### Test upload
- Prepare a PDF
- In the UI, click the "Upload" tab
- Drag-and-drop or select the PDF
- Click "Process PDF"

### Test Windows folder
- Prepare a folder containing PDFs
- In the UI, click the "Windows Folder" tab
- Enter a folder path (e.g., `C:\Users\YourName\Documents\PDFs`)
- Click "Process All PDFs in Folder"

### Test SharePoint (requires setup)
- Ensure SharePoint environment variables are configured
- In the UI, click the "SharePoint" tab
- Enter a SharePoint file path
- Click "Process SharePoint PDF"

## FAQ

Q: The app won't start?
A: Verify all dependencies are installed and Python >= 3.8.

Q: Azure AI processing failed?
A: Check your Azure OpenAI settings in `.env`.

Q: MinerU processing failed?
A: This can be expected if MinerU is not installed; the system uses simulated data. To use real OCR, install MinerU 2.5.

Q: Can't download output?
A: Ensure the `outputs/` directory exists and the app has write permissions.

## Next Steps

- Read [README.md](../README.md) for full features
- Read [INSTALLATION.md](INSTALLATION.md) for detailed setup
- Customize Azure AI prompts to improve extraction
- Install MinerU 2.5 for better OCR quality

