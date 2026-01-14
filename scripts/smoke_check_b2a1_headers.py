from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any


def _count_header_markers(data: Any) -> tuple[int, int, int]:
    """Return (promoted_header_count, promoted_from_discarded_count, discarded_nonempty_count)."""
    if not isinstance(data, list):
        return 0, 0, 0

    promoted = 0
    promoted_from_discarded = 0
    discarded_nonempty = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("zone") == "header":
            promoted += 1
        if item.get("original_type") == "discarded":
            promoted_from_discarded += 1
        if item.get("type") == "discarded" and (item.get("text") or "").strip():
            discarded_nonempty += 1
    return promoted, promoted_from_discarded, discarded_nonempty


def _contains(data: Any, needle: str) -> bool:
    if not needle:
        return False
    if not isinstance(data, list):
        return False
    needle_u = needle.upper()
    for it in data:
        if not isinstance(it, dict):
            continue
        text = (it.get("text") or "")
        if needle_u in text.upper():
            return True
    return False


async def main() -> None:
    # Ensure repo root is importable when running as a script.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    pdf_path = Path("sample") / "B2A1.pdf"
    if not pdf_path.exists():
        raise SystemExit(f"Missing: {pdf_path}")

    # Lazy import so dotenv/config is loaded as in app runtime.
    from services.mineru_service import MinerUService

    service = MinerUService()

    print(f"PDF: {pdf_path}")
    data = await service.process_pdf(pdf_path)

    promoted, promoted_from_discarded, discarded_nonempty = _count_header_markers(data)
    print(f"Result type: {type(data).__name__}")
    print(f"Promoted header blocks (zone=header): {promoted}")
    print(f"Promoted blocks (original_type=discarded): {promoted_from_discarded}")
    print(f"Remaining discarded blocks with text: {discarded_nonempty}")
    print(f"Contains 'JEBSEN': {_contains(data, 'JEBSEN')}")

    # Print a couple examples for quick eyeballing.
    if isinstance(data, list):
        examples = [
            it
            for it in data
            if isinstance(it, dict)
            and it.get("zone") == "header"
            and (it.get("text") or "").strip()
        ][:5]
        for idx, it in enumerate(examples, 1):
            bbox = it.get("bbox")
            text = (it.get("text") or "").strip().replace("\n", " ")
            print(f"Header sample {idx}: bbox={bbox} text={text[:120]}")


if __name__ == "__main__":
    asyncio.run(main())
