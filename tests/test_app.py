"""Simple application test script."""
import asyncio
import sys
from pathlib import Path

# Ensure repo root is importable when running from tests/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.pdf_processor import PDFProcessor
from services.azure_ai_service import AzureAIService
from services.folder_service import FolderService


async def test_pdf_processor():
    """Test the PDF processor."""
    print("Testing PDF processor...")
    processor = PDFProcessor()

    # Use a test PDF if present
    test_pdf = REPO_ROOT / "test.pdf"
    if test_pdf.exists():
        needs_ocr = await processor.needs_ocr(test_pdf)
        print(f"  PDF needs OCR: {needs_ocr}")
    else:
        print("  Test PDF file not found")


async def test_folder_service():
    """Test the folder service."""
    print("Testing folder service...")
    service = FolderService()

    # Test repo root
    current_dir = REPO_ROOT
    if service.validate_folder_path(str(current_dir)):
        pdf_files = service.get_pdf_files(str(current_dir))
        print(f"  Found {len(pdf_files)} PDF files")
    else:
        print("  Failed to validate folder path")


async def test_azure_ai_service():
    """Test the Azure AI service."""
    print("Testing Azure AI service...")
    try:
        service = AzureAIService()
        print("  Azure AI service initialized successfully")
    except Exception as e:
        print(f"  Azure AI service initialization failed: {e}")


async def main():
    """Run all tests."""
    print("Starting component tests...\n")

    await test_pdf_processor()
    await test_folder_service()
    await test_azure_ai_service()

    print("\nTests completed!")


if __name__ == "__main__":
    asyncio.run(main())
