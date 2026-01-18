# Use ocrmypdf to read a PDF file (or image file), run OCRmyPDF to add a text layer, and write a searchable PDF.
from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys
import shutil
import subprocess
from typing import Optional


def _is_valid_tessdata_dir(tessdata_dir: Path) -> bool:
    """Return True if the directory looks like a usable tessdata dir.

    OCRmyPDF invokes tesseract with the 'hocr' and 'txt' configs. Those config
    files live under tessdata/configs/.
    """

    try:
        t = Path(tessdata_dir)
        if not t.exists() or not t.is_dir():
            return False

        has_configs = (t / "configs" / "hocr").exists() and (t / "configs" / "txt").exists()
        if not has_configs:
            return False

        # Ensure some language data exists. Don't require 'eng' specifically.
        has_any_lang = any(p.suffix.lower() == ".traineddata" for p in t.glob("*.traineddata"))
        return bool(has_any_lang)
    except Exception:
        return False


def _coerce_to_tessdata_dir(prefix: Path) -> Optional[Path]:
    """Accept either a tessdata directory or a parent containing tessdata/."""

    p = Path(prefix)
    if _is_valid_tessdata_dir(p):
        return p
    if _is_valid_tessdata_dir(p / "tessdata"):
        return p / "tessdata"
    return None


def _ensure_tessdata_prefix(explicit: Optional[Path] = None) -> None:
    """Ensure Tesseract can locate tessdata (and configs/hocr + configs/txt).

    Sets TESSDATA_PREFIX if it's missing or points somewhere unusable.
    """

    def _try_set(candidate: Path) -> bool:
        tessdata_dir = _coerce_to_tessdata_dir(candidate)
        if tessdata_dir is None:
            return False
        os.environ["TESSDATA_PREFIX"] = str(tessdata_dir)
        return True

    # 1) Explicit override wins.
    if explicit is not None:
        if _try_set(explicit):
            return

    # 2) If env is set and valid, keep it; if set but invalid, replace.
    existing = os.environ.get("TESSDATA_PREFIX")
    if existing:
        if _coerce_to_tessdata_dir(Path(existing)) is not None:
            return

    # 3) Ask tesseract itself (best signal) if it's available on PATH.
    try:
        completed = subprocess.run(
            ["tesseract", "--print-tessdata-dir"],
            check=True,
            capture_output=True,
            text=True,
        )
        td = (completed.stdout or "").strip()
        if td:
            if _try_set(Path(td)):
                return
    except Exception:
        pass

    # 4) Common locations: current Python env (conda/venv), then tesseract install dir.
    prefix = Path(sys.prefix)
    candidates = [
        prefix / "Library" / "share" / "tessdata",
        prefix / "share" / "tessdata",
        prefix,
    ]

    tesseract_exe = shutil.which("tesseract")
    if tesseract_exe:
        tbase = Path(tesseract_exe).resolve().parent
        candidates.extend([tbase / "tessdata", tbase])

    # 5) Windows installers often live here.
    candidates.extend(
        [
            Path("C:/Program Files/Tesseract-OCR/tessdata"),
            Path("C:/Program Files (x86)/Tesseract-OCR/tessdata"),
        ]
    )

    for cand in candidates:
        if _try_set(cand):
            return


def ocr_to_searchable_pdf(
    input_pdf: Path,
    output_pdf: Path,
    *,
    language: str = "eng",
    redo: bool = False,
    deskew: bool = False,
    rotate: bool = False,
    jobs: int = 1,
    optimize: int = 0,
    image_dpi: Optional[int] = None,
    tessdata_prefix: Optional[Path] = None,
) -> Path:
    """Run OCRmyPDF to produce a searchable (text-layer) PDF.

    This is the single supported searchable-PDF generator in this repo.
    """

    input_path = Path(input_pdf).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")

    output_path = Path(output_pdf).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import ocrmypdf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'ocrmypdf'. Install with: python -m pip install ocrmypdf"
        ) from exc

    # Help Tesseract find traineddata + configs (hocr/txt) used by OCRmyPDF.
    _ensure_tessdata_prefix(tessdata_prefix)

    # If the PDF already contains text, default behavior is to keep it and skip OCR.
    # Use redo=True to force OCR.
    skip_text = not bool(redo)

    ocrmypdf.ocr(
        input_file=str(input_path),
        output_file=str(output_path),
        language=str(language or "eng"),
        skip_text=skip_text,
        force_ocr=bool(redo),
        deskew=bool(deskew),
        rotate_pages=bool(rotate),
        optimize=int(optimize),
        jobs=max(1, int(jobs)),
        image_dpi=int(image_dpi) if image_dpi is not None else None,
        progress_bar=False,
    )

    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR a PDF and output a searchable (text-layer) PDF."
    )
    parser.add_argument(
        "input_pdf",
        nargs="?",
        default="B2A1_Case1.pdf",
        help="Input PDF path (default: B2A1_Case1.pdf)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output PDF path (default: <SEARCHABLE_PDF_OUTPUT_DIR>/<input>_searchable.pdf)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder (default: SEARCHABLE_PDF_OUTPUT_DIR when available)",
    )
    parser.add_argument(
        "--tessdata-prefix",
        default=None,
        help=(
            "Tesseract tessdata directory (or a parent containing tessdata/). "
            "Overrides TESSDATA_PREFIX if needed."
        ),
    )
    parser.add_argument(
        "-l",
        "--language",
        default="eng",
        help="Tesseract language(s), e.g. 'eng' or 'eng+spa' (default: eng)",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Force OCR even if the PDF already has text.",
    )
    parser.add_argument(
        "--deskew",
        action="store_true",
        help="Deskew pages (useful for scanned documents).",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Auto-rotate pages.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel jobs (default: 1)",
    )
    parser.add_argument(
        "--optimize",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help=(
            "Optimization level 0-3 (default: 0). "
            "Levels 2-3 may require external tools like pngquant."
        ),
    )
    parser.add_argument(
        "--image-dpi",
        type=int,
        default=None,
        help=(
            "Assume this DPI when input pages lack DPI metadata (OCRmyPDF image_dpi). "
            "Use when scanned images/PDFs have unknown/incorrect DPI."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Align defaults with the app's folder conventions when possible.
    try:
        from config import UPLOAD_DIR, SEARCHABLE_PDF_OUTPUT_DIR
    except Exception:  # pragma: no cover
        UPLOAD_DIR = None
        SEARCHABLE_PDF_OUTPUT_DIR = None

    raw_input = Path(args.input_pdf).expanduser()
    if not raw_input.is_absolute() and UPLOAD_DIR is not None:
        candidate = Path(UPLOAD_DIR) / raw_input
        input_path = candidate.resolve() if candidate.exists() else (Path.cwd() / raw_input).resolve()
    else:
        input_path = raw_input.resolve()
    if not input_path.exists():
        raise SystemExit(f"Input PDF not found: {input_path}")

    if args.output is not None:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_dir = None
        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser().resolve()
        elif SEARCHABLE_PDF_OUTPUT_DIR is not None:
            output_dir = Path(SEARCHABLE_PDF_OUTPUT_DIR)
        else:
            output_dir = input_path.parent

        output_path = Path(output_dir) / f"{input_path.stem}_searchable.pdf"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ocr_to_searchable_pdf(
            input_path,
            output_path,
            language=args.language,
            redo=args.redo,
            deskew=args.deskew,
            rotate=args.rotate,
            jobs=args.jobs,
            optimize=args.optimize,
            image_dpi=args.image_dpi,
            tessdata_prefix=Path(args.tessdata_prefix).expanduser() if args.tessdata_prefix else None,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Wrote searchable PDF: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
