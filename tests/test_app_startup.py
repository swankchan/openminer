"""Test application startup and basic functionality."""
import asyncio
import sys
from pathlib import Path

# Ensure repo root is importable when running from tests/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.mineru_service import MinerUService
from services.pdf_processor import PDFProcessor
from services.azure_ai_service import AzureAIService
from config import DEBUG, USE_VLLM_ASYNC


async def test_services():
    """Test initialization of each service."""
    print("=" * 60)
    print("Testing application service initialization")
    print("=" * 60)

    # 1. Test PDFProcessor
    print("\n1. Testing PDFProcessor:")
    try:
        PDFProcessor()
        print("   ✓ PDFProcessor initialized successfully")
    except Exception as e:
        print(f"   ✗ PDFProcessor initialization failed: {e}")
        return False

    # 2. Test MinerUService
    print("\n2. Testing MinerUService:")
    try:
        mineru_service = MinerUService()
        await mineru_service._initialize()
        print("   ✓ MinerUService initialized successfully")
        if USE_VLLM_ASYNC:
            print("   - Using vllm-async-engine mode")
        else:
            print("   - Using CLI mode")
    except Exception as e:
        print(f"   ✗ MinerUService initialization failed: {e}")
        return False

    # 3. Test AzureAIService
    print("\n3. Testing AzureAIService:")
    try:
        azure_ai_service = AzureAIService()
        print("   ✓ AzureAIService initialized successfully")
        if azure_ai_service.client:
            print("   - Azure OpenAI client connected")
        else:
            print("   - Using basic extraction (no Azure OpenAI credentials)")
    except Exception as e:
        print(f"   ✗ AzureAIService initialization failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ All services initialized successfully!")
    print("=" * 60)
    return True


async def test_config():
    """Test configuration."""
    print("\nConfiguration check:")
    print(f"  DEBUG: {DEBUG}")
    print(f"  USE_VLLM_ASYNC: {USE_VLLM_ASYNC}")
    from config import FORCE_OCR
    print(f"  FORCE_OCR: {FORCE_OCR}")


async def main():
    """Main test function."""
    print("\nStarting application test...\n")

    # Test configuration
    await test_config()

    # Test services
    success = await test_services()

    if success:
        print("\n✓ Application is ready and should work normally!")
        print("\nHints:")
        print("  - Visit http://localhost:8000 to use the web UI")
        print("  - Or use the API endpoint to upload a PDF")
    else:
        print("\n✗ Application initialization failed; please check the error output")

    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
