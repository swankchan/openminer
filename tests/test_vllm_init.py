"""Test vllm-async-engine initialization."""
import asyncio
import sys
from pathlib import Path

# Ensure repo root is importable when running from tests/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.mineru_service import MinerUService


async def test():
    print("Testing MinerUService initialization...")
    service = MinerUService()
    await service._initialize()
    print("✓ MinerUService initialized successfully")


if __name__ == "__main__":
    asyncio.run(test())
