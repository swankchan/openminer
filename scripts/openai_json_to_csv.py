"""Convert OpenAI JSON (raw responses or structured outputs) to CSV.

Supports:
- Raw OpenAI Chat Completions style payloads ("choices" -> "message" -> "content")
- Raw OpenAI Responses API style payloads ("output" blocks -> output_text)
- Already-extracted structured JSON (e.g., invoice fields + ItemTable)

Examples (PowerShell):
  python scripts/openai_json_to_csv.py --input outputs/B2A1_Case1_20260114_063325.json --output outputs/B2A1_Case1_20260114_063325.csv
  python scripts/openai_json_to_csv.py --input my_openai_dump.json --output out.csv --mode openai_response
  python scripts/openai_json_to_csv.py --input extracted.json --output out.csv --explode ItemTable
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_records(input_path: Path) -> List[Any]:
    """Load JSON records.

    - .json: dict or list
    - .jsonl: one JSON per line
    """
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        records: List[Any] = []
        with input_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_no} in {input_path}: {e}") from e
        return records

    data = json.loads(_read_text(input_path))
    if isinstance(data, list):
        return data
    return [data]


def _extract_text_from_chat_completions(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, None
    first = choices[0] if isinstance(choices[0], dict) else None
    if not first:
        return None, None
    message = first.get("message") if isinstance(first.get("message"), dict) else None
    content = message.get("content") if message else None
    finish_reason = first.get("finish_reason")
    if isinstance(content, str):
        return content, str(finish_reason) if finish_reason is not None else None
    return None, str(finish_reason) if finish_reason is not None else None


def _extract_text_from_responses_api(payload: Dict[str, Any]) -> Optional[str]:
    # Responses API: output -> [{content: [{type: 'output_text', text: '...'}]}]
    output = payload.get("output")
    if not isinstance(output, list):
        return None

    text_chunks: List[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "output_text" and isinstance(block.get("text"), str):
                text_chunks.append(block["text"])
            # Some dumps may store as {type: 'text', text: '...'}
            elif block.get("type") in {"text", "output_text"} and isinstance(block.get("text"), str):
                text_chunks.append(block["text"])

    combined = "\n".join(chunk for chunk in text_chunks if chunk)
    return combined if combined else None


def detect_payload_kind(record: Any) -> str:
    """Return one of: 'chat_completions', 'responses', 'structured'."""
    if not isinstance(record, dict):
        return "structured"
    if "choices" in record and "model" in record:
        return "chat_completions"
    if "output" in record and "id" in record:
        return "responses"
    return "structured"


def try_parse_json_from_text(text: str) -> Optional[Any]:
    """Try to parse JSON from a model-produced string."""
    candidate = text.strip()
    if not candidate:
        return None

    # Common: fenced code block ```json ... ```
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            # Drop first and last fence if present.
            if len(lines) >= 2 and lines[-1].startswith("```"):
                candidate = "\n".join(lines[1:-1]).strip()
            else:
                candidate = "\n".join(lines[1:]).strip()

    # Basic heuristic: must start with { or [
    if not (candidate.startswith("{") or candidate.startswith("[")):
        return None

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def flatten_json(value: Any, prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested JSON into a single-level dict.

    - Dict keys become `prefix.key`
    - Lists are serialized as JSON strings (unless the caller explodes them)
    """
    items: Dict[str, Any] = {}

    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}{sep}{k}" if prefix else str(k)
            items.update(flatten_json(v, prefix=key, sep=sep))
        return items

    if isinstance(value, list):
        # Keep lists as a single cell to avoid row explosion unless requested.
        items[prefix or "value"] = json.dumps(value, ensure_ascii=False)
        return items

    items[prefix or "value"] = value
    return items


def flatten_structured_records(
    records: List[Any],
    explode_path: Optional[str] = None,
    explode_auto: bool = True,
) -> List[Dict[str, Any]]:
    """Flatten structured records to a DataFrame.

    - If explode_path is provided and points to a list of dicts, explode it into one row per entry.
    - If explode_auto is True, try to find a single top-level list-of-dicts key (e.g., ItemTable) and explode it.
    """

    # Normalize to list of dicts where possible; otherwise store raw JSON string
    normalized: List[Dict[str, Any]] = []
    for rec in records:
        if isinstance(rec, dict):
            normalized.append(rec)
        else:
            normalized.append({"value": rec})

    # Explode: apply to each record, then normalize
    if explode_path:
        exploded_rows: List[Dict[str, Any]] = []
        for rec in normalized:
            value = rec.get(explode_path)
            if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                base = {k: v for k, v in rec.items() if k != explode_path}
                for item in value:
                    row = dict(base)
                    # merge item dict under its own columns
                    row.update(item)
                    exploded_rows.append(row)
            else:
                exploded_rows.append(rec)
        return [flatten_json(r) for r in exploded_rows]

    if explode_auto:
        # Find candidate keys that are list-of-dicts across (most) records.
        preferred_keys = [
            "ItemTable",
            "items",
            "line_items",
            "lineItems",
            "rows",
            "records",
            "data",
        ]

        def is_list_of_dicts(v: Any) -> bool:
            return isinstance(v, list) and bool(v) and all(isinstance(x, dict) for x in v)

        # Prefer known keys if present
        for key in preferred_keys:
            if any(is_list_of_dicts(rec.get(key)) for rec in normalized):
                return flatten_structured_records(records, explode_path=key, explode_auto=False)

        # Otherwise, if exactly one top-level key looks like list-of-dicts, use it.
        candidate_keys: List[str] = []
        keys = set().union(*(rec.keys() for rec in normalized))
        for key in keys:
            if any(is_list_of_dicts(rec.get(key)) for rec in normalized):
                candidate_keys.append(key)
        if len(candidate_keys) == 1:
            return flatten_structured_records(records, explode_path=candidate_keys[0], explode_auto=False)

    return [flatten_json(r) for r in normalized]


def openai_records_to_rows(records: List[Any], include_raw: bool = False) -> List[Dict[str, Any]]:
    """Convert OpenAI raw response dumps to simple rows."""
    rows: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            rows.append({"kind": "unknown", "raw": json.dumps(rec, ensure_ascii=False)})
            continue

        kind = detect_payload_kind(rec)
        model = rec.get("model")
        created = rec.get("created") or rec.get("created_at")
        usage = rec.get("usage") if isinstance(rec.get("usage"), dict) else {}

        response_text: Optional[str] = None
        finish_reason: Optional[str] = None
        if kind == "chat_completions":
            response_text, finish_reason = _extract_text_from_chat_completions(rec)
        elif kind == "responses":
            response_text = _extract_text_from_responses_api(rec)

        parsed_json = try_parse_json_from_text(response_text or "") if response_text else None

        row: Dict[str, Any] = {
            "kind": kind,
            "id": rec.get("id"),
            "model": model,
            "created": created,
            "finish_reason": finish_reason,
            "response_text": response_text,
            "usage.prompt_tokens": usage.get("prompt_tokens"),
            "usage.completion_tokens": usage.get("completion_tokens"),
            "usage.total_tokens": usage.get("total_tokens"),
        }

        # If model returned JSON, flatten it into columns with prefix parsed.
        if isinstance(parsed_json, dict):
            flat = flatten_json(parsed_json)
            for k, v in flat.items():
                row[f"parsed.{k}"] = v
        elif isinstance(parsed_json, list):
            row["parsed"] = json.dumps(parsed_json, ensure_ascii=False)

        if include_raw:
            row["raw"] = json.dumps(rec, ensure_ascii=False)

        rows.append(row)

    return rows


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    if not rows:
        # still write header-less CSV
        output_path.write_text("", encoding="utf-8-sig")
        return

    # Stable header order: common columns first, then the rest.
    preferred = [
        "kind",
        "id",
        "model",
        "created",
        "finish_reason",
        "response_text",
        "usage.prompt_tokens",
        "usage.completion_tokens",
        "usage.total_tokens",
    ]
    all_keys = set().union(*(r.keys() for r in rows))
    header = [k for k in preferred if k in all_keys] + sorted(k for k in all_keys if k not in preferred)

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert OpenAI JSON dumps or extracted JSON into CSV")
    parser.add_argument("--input", required=True, help="Input .json or .jsonl file")
    parser.add_argument("--output", required=True, help="Output .csv file")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "openai_response", "structured"],
        help="How to interpret the JSON",
    )
    parser.add_argument(
        "--explode",
        default=None,
        help="Structured mode: explode this top-level list-of-dicts key into one CSV row per entry (e.g., ItemTable)",
    )
    parser.add_argument(
        "--no-auto-explode",
        action="store_true",
        help="Structured mode: disable auto-detection of an item table to explode",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="OpenAI response mode: include raw JSON in a column",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    records = load_records(input_path)

    # Decide mode
    mode = args.mode
    if mode == "auto":
        # If any record looks like OpenAI, use openai_response; otherwise structured.
        looks_openai = any(detect_payload_kind(r) in {"chat_completions", "responses"} for r in records)
        mode = "openai_response" if looks_openai else "structured"

    if mode == "openai_response":
        rows = openai_records_to_rows(records, include_raw=bool(args.include_raw))
    else:
        rows = flatten_structured_records(
            records,
            explode_path=args.explode,
            explode_auto=not bool(args.no_auto_explode),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_csv(rows, output_path)

    print(f"Wrote {len(rows)} row(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
