# python jsoncsv_app.py --data data.json --sidecar sidecar.json --template template.csv
import argparse
import csv
import json
import re
import sys
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Optional

# Allow running this file directly (python services/jsoncsv_app.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import BASE_DIR, JSON_OUTPUT_DIR, CSV_OUTPUT_DIR


_PLACEHOLDER_RE = re.compile(r"\[([^\[\]]+)\]")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _get_by_path(obj: Any, path: str) -> Any:
    """Fetch a value from nested dicts using dot notation; falls back to direct key."""
    if not path:
        return None

    if isinstance(obj, dict) and path in obj:
        return obj[path]

    current: Any = obj
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float, bool)):
        return str(value)
    return str(value)


def _extract_last_folder_name_from_windows_path(path_str: str) -> str:
    if not path_str:
        return ""
    try:
        p = PureWindowsPath(path_str)
        return p.parent.name or ""
    except Exception:
        return ""


def _replace_placeholders_in_cell(cell: str, context: Dict[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        value = _get_by_path(context, key)
        return _to_str(value)

    return _PLACEHOLDER_RE.sub(repl, cell)


def _row_has_placeholders(row: List[str]) -> bool:
    return any(_PLACEHOLDER_RE.search(cell or "") for cell in row)


def _is_item_table_placeholder_row(row: List[str]) -> bool:
    # Heuristic: the template.csv uses a row of pure placeholders like [PONumber],[PartNumber],...
    has_any = False
    for cell in row:
        cell = (cell or "").strip()
        if not cell:
            continue
        if not (cell.startswith("[") and cell.endswith("]")):
            return False
        has_any = True
    return has_any


def render_from_template(
    template_rows: List[List[str]],
    header_context: Dict[str, Any],
    items: List[Dict[str, Any]],
) -> List[List[str]]:
    output: List[List[str]] = []

    for row in template_rows:
        if _is_item_table_placeholder_row(row):
            # Expand per item
            if not items:
                # Emit one empty row (placeholders become empty)
                output.append([_replace_placeholders_in_cell(cell, {}) for cell in row])
                continue

            for item in items:
                filled = [_replace_placeholders_in_cell(cell, item) for cell in row]
                output.append(filled)
            continue

        if _row_has_placeholders(row):
            output.append([_replace_placeholders_in_cell(cell, header_context) for cell in row])
        else:
            output.append(row)

    return output


def _read_csv_rows(path: Path) -> List[List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        return [row for row in reader]


def _write_csv_rows(path: Path, rows: Iterable[List[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        for row in rows:
            writer.writerow(row)


def _read_src_dst_hint(sidecar_path: Path) -> Optional[Dict[str, str]]:
    """Read src/dst hint from src_dst.txt near the sidecar or common locations.

    Returns a dict with keys: source_path, processed_url; or None if not found.
    """
    candidates: List[Path] = []
    try:
        # 1) Same folder as sidecar
        candidates.append(sidecar_path.parent / "src_dst.txt")
        # 2) Configured sidecar output dir
        from config import SIDECAR_OUTPUT_DIR  # type: ignore
        if SIDECAR_OUTPUT_DIR:
            candidates.append(Path(SIDECAR_OUTPUT_DIR) / "src_dst.txt")
    except Exception:
        pass
    try:
        # 3) Common outputs folder under project
        candidates.append(BASE_DIR / "outputs" / "src_dst.txt")
    except Exception:
        pass

    for cand in candidates:
        try:
            if cand.exists():
                obj = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    return {
                        "source_path": str(obj.get("source_path") or ""),
                        "processed_url": str(obj.get("processed_url") or ""),
                    }
        except Exception:
            continue
    return None


def generate_template_csv_file(
    *,
    data_json_path: Path,
    sidecar_json_path: Path,
    template_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate a CSV file from the fixed template (templates/template.csv).

    This is the programmatic entrypoint used by the FastAPI pipeline.
    """

    data_path = Path(data_json_path)
    sidecar_path = Path(sidecar_json_path)

    template_csv_path = Path(template_path) if template_path else (BASE_DIR / "templates" / "template.csv")
    if not template_csv_path.is_absolute():
        candidate = BASE_DIR / template_csv_path
        if candidate.exists():
            template_csv_path = candidate

    out_path = Path(output_path) if output_path else (CSV_OUTPUT_DIR / f"{data_path.stem}_output.csv")

    data = _load_json(data_path)
    sidecar = _load_json(sidecar_path)

    confidence_overall = _get_by_path(sidecar, "confidence.overall")
    if confidence_overall is None:
        confidence_overall = _get_by_path(sidecar, "confidence")

    # Source/destination from src_dst.txt exclusively
    srcdst = _read_src_dst_hint(sidecar_path)

    header_context: Dict[str, Any] = {}
    if isinstance(data, dict):
        header_context.update(data)

    header_context["confidence"] = confidence_overall
    header_context["SourceURL"] = (srcdst or {}).get("source_path", "")
    header_context["ProcessedURL"] = (srcdst or {}).get("processed_url", "")
    # Derive [OU] from grandparent folder of source_path (third-from-end path component).
    try:
        src_path_val = (srcdst or {}).get("source_path", "") or ""
        ou_val = ""
        if src_path_val:
            parts = [p for p in src_path_val.split("/") if p]
            if len(parts) >= 3:
                grandparent = parts[-3]
                # If grandparent contains spaces, take first token
                ou_val = str(grandparent).split()[0] if isinstance(grandparent, str) else ""
        header_context["OU"] = ou_val
    except Exception:
        header_context["OU"] = ""
    # Expose processed SharePoint info for template placeholders (e.g., [ProcessedURL])
    try:
        processed_obj = None
        if isinstance(sidecar, dict):
            processed_obj = sidecar.get("processed")
        # Prefer processed.server_relative_path; fallback to sidecar['ProcessedURL']
        processed_url = ""
        processed_folder = ""
        processed_filename = ""
        if processed_obj and isinstance(processed_obj, dict):
            processed_url = processed_obj.get("server_relative_path") or ""
            processed_folder = processed_obj.get("folder") or ""
            processed_filename = processed_obj.get("filename") or ""
        if not processed_url and isinstance(sidecar, dict):
            processed_url = sidecar.get("ProcessedURL") or ""
        header_context["ProcessedURL"] = processed_url
        header_context["ProcessedFolder"] = processed_folder
        header_context["ProcessedFilename"] = processed_filename
    except Exception:
        header_context["ProcessedURL"] = ""
        header_context["ProcessedFolder"] = ""
        header_context["ProcessedFilename"] = ""

    items: List[Dict[str, Any]] = []
    if isinstance(data, dict) and isinstance(data.get("ItemTable"), list):
        for item in data["ItemTable"]:
            if isinstance(item, dict):
                items.append(item)

    template_rows = _read_csv_rows(template_csv_path)
    rendered_rows = render_from_template(template_rows, header_context, items)
    _write_csv_rows(out_path, rendered_rows)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fill a CSV template using a data JSON + a _sidecar.json file."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the main data JSON (invoice fields + ItemTable).",
    )
    parser.add_argument(
        "--sidecar",
        required=True,
        help="Path to the sidecar JSON (confidence + source_pdf).",
    )
    parser.add_argument(
        "--template",
        default=str(BASE_DIR / "templates" / "template.csv"),
        help="Path to the template CSV with [placeholders] (default: templates/template.csv).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: <data_stem>_output.csv in the same folder).",
    )

    args = parser.parse_args(argv)

    data_path = Path(args.data)
    sidecar_path = Path(args.sidecar)
    template_path = Path(args.template)

    if not template_path.is_absolute():
        candidate = (BASE_DIR / template_path)
        if candidate.exists():
            template_path = candidate

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = JSON_OUTPUT_DIR / f"{data_path.stem}_output.csv"

    data = _load_json(data_path)
    sidecar = _load_json(sidecar_path)

    confidence_overall = _get_by_path(sidecar, "confidence.overall")
    if confidence_overall is None:
        confidence_overall = _get_by_path(sidecar, "confidence")

    # Source/destination from src_dst.txt exclusively
    srcdst = _read_src_dst_hint(sidecar_path)

    header_context: Dict[str, Any] = {}
    if isinstance(data, dict):
        header_context.update(data)

    header_context["confidence"] = confidence_overall
    header_context["SourceURL"] = (srcdst or {}).get("source_path", "")
    header_context["ProcessedURL"] = (srcdst or {}).get("processed_url", "")
    # Derive OU from grandparent folder of source_path (third-from-end path component).
    try:
        src_path_val = (srcdst or {}).get("source_path", "") or ""
        ou_val = ""
        if src_path_val:
            parts = [p for p in src_path_val.split("/") if p]
            if len(parts) >= 3:
                grandparent = parts[-3]
                ou_val = str(grandparent).split()[0] if isinstance(grandparent, str) else ""
        header_context["OU"] = ou_val
    except Exception:
        header_context["OU"] = ""

    items: List[Dict[str, Any]] = []
    if isinstance(data, dict) and isinstance(data.get("ItemTable"), list):
        for item in data["ItemTable"]:
            if isinstance(item, dict):
                items.append(item)

    template_rows = _read_csv_rows(template_path)
    rendered_rows = render_from_template(template_rows, header_context, items)
    _write_csv_rows(out_path, rendered_rows)

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
