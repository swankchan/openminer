"""Run the sample PDF through the same /api/upload pipeline (in-process).

This avoids having to start `python run.py` + POST multipart from PowerShell.
It still exercises the exact same FastAPI route + processing queue.

Outputs:
- JSON output in outputs/
- Searchable PDF in outputs/ (when GENERATE_SEARCHABLE_PDF=True)

Usage:
  python scripts/run_sample_searchable_pdf.py

Optional:
  python scripts/run_sample_searchable_pdf.py sample/SomeOther.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (repo_root / "sample" / "B2A1_Case1.pdf")
    pdf_path = pdf_path if pdf_path.is_absolute() else (repo_root / pdf_path)

    if not pdf_path.exists():
        print(f"ERROR: input file not found: {pdf_path}")
        return 2

    # Import the app after resolving paths so config loads from repo root.
    from main import app  # noqa: WPS433 (runtime import)

    with TestClient(app) as client:
        with pdf_path.open("rb") as f:
            files = {"file": (pdf_path.name, f, "application/pdf")}
            resp = client.post("/api/upload", files=files)

    print(f"HTTP {resp.status_code}")
    try:
        payload = resp.json()
    except Exception:
        print(resp.text)
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    sp = payload.get("searchable_pdf_path")
    if sp:
        sp_path = Path(sp)
        # Quick sanity check: does the PDF have extractable text?
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(sp_path))
            extracted_parts = []
            for i, page in enumerate(reader.pages[:3]):
                extracted_parts.append(page.extract_text() or "")
            extracted = "".join(extracted_parts)

            extracted = extracted.strip()
            print(f"\nSearchable PDF: {sp_path}")
            print(f"Extracted text chars (first 3 pages): {len(extracted)}")
        except Exception as e:
            print(f"\nSearchable PDF: {sp_path}")
            print(f"(Could not extract text for validation: {e})")
    else:
        print("\nNo searchable PDF path returned (GENERATE_SEARCHABLE_PDF may be off, or OCR failed).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
