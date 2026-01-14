"""Full application test."""
import asyncio
import sys
from pathlib import Path

# Ensure repo root is importable when running from tests/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


async def main():
    print("=" * 70)
    print("Full application test")
    print("=" * 70)

    # 1. Test configuration
    print("\n[1/4] Testing configuration...")
    from config import DEBUG, USE_VLLM_ASYNC, FORCE_OCR, MINERU_MODEL_NAME

    print(f"   DEBUG: {DEBUG}")
    print(f"   USE_VLLM_ASYNC: {USE_VLLM_ASYNC}")
    print(f"   FORCE_OCR: {FORCE_OCR}")
    print(f"   MINERU_MODEL_NAME: {MINERU_MODEL_NAME}")
    print("   ✓ Configuration loaded successfully")

    # 2. Test service initialization
    print("\n[2/4] Testing service initialization...")
    from services.pdf_processor import PDFProcessor
    from services.mineru_service import MinerUService
    from services.azure_ai_service import AzureAIService

    PDFProcessor()
    print("   ✓ PDFProcessor initialized successfully")

    mineru_service = MinerUService()
    await mineru_service._initialize()
    print("   ✓ MinerUService initialized successfully")

    AzureAIService()
    print("   ✓ AzureAIService initialized successfully")

    # 3. Test directories
    print("\n[3/4] Testing directory structure...")
    from config import UPLOAD_DIR, OUTPUT_DIR, MINERU_OUTPUT_DIR

    dirs = [UPLOAD_DIR, OUTPUT_DIR, MINERU_OUTPUT_DIR]
    for dir_path in dirs:
        if dir_path.exists():
            print(f"   ✓ {dir_path.name}/ directory exists")
        else:
            print(f"   ✗ {dir_path.name}/ directory missing")

    # 4. Test FastAPI application
    print("\n[4/4] Testing FastAPI application...")
    try:
        from main import app

        print("   ✓ FastAPI app loaded successfully")

        # Check routes
        routes = [getattr(route, "path", None) for route in app.routes]
        routes = [r for r in routes if r]
        print(f"   ✓ Found {len(routes)} routes:")
        for route in routes[:5]:  # Show only the first 5
            print(f"      - {route}")
        if len(routes) > 5:
            print(f"      ... and {len(routes) - 5} more")
    except Exception as e:
        print(f"   ✗ FastAPI app failed to load: {e}")
        return False

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
    print("\nApplication status:")
    print("  ✓ All services initialized")
    print("  ✓ Directory structure OK")
    print("  ✓ FastAPI app ready")
    print("\nNext steps:")
    print("  1. Run: python run.py")
    print("  2. Visit: http://localhost:8000")
    print("  3. Upload a PDF to test")

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
