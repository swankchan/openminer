from __future__ import annotations

import argparse
from pathlib import Path


def _env_clean(value: str | None) -> str:
    if value is None:
        return ""
    cleaned = value.split("#", 1)[0].strip()
    if (len(cleaned) >= 2) and (cleaned[0] == cleaned[-1]) and cleaned[0] in ("'", '"'):
        cleaned = cleaned[1:-1]
    return cleaned.strip()


def _parse_env(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = _env_clean(val)
        if key:
            out[key] = val
    return out


def _update_env_file(dst_path: Path, updates: dict[str, str]) -> tuple[int, list[str]]:
    """Update (or append) KEY=VALUE lines in dst_path.

    Returns: (number_of_keys_updated_or_added, list_of_keys_updated_or_added)
    """
    original = dst_path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)

    updated_keys: set[str] = set()

    def replace_line(line: str, key: str, value: str) -> str:
        # Preserve newline style from the original line.
        newline = "\n" if line.endswith("\n") else ""
        if line.endswith("\r\n"):
            newline = "\r\n"
        return f"{key}={value}{newline}"

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            lines[i] = replace_line(line, key, updates[key])
            updated_keys.add(key)

    # Append missing keys at end (ensure file ends with newline)
    if lines and not (lines[-1].endswith("\n") or lines[-1].endswith("\r\n")):
        lines[-1] = lines[-1] + "\n"

    missing = [k for k in updates.keys() if k not in updated_keys]
    if missing:
        if lines and not (lines[-1].endswith("\n") or lines[-1].endswith("\r\n")):
            lines[-1] = lines[-1] + "\n"
        if lines and lines[-1].strip() != "":
            lines.append("\n")
        lines.append("# Synced SharePoint credentials\n")
        for k in missing:
            lines.append(f"{k}={updates[k]}\n")
            updated_keys.add(k)

    new_text = "".join(lines)
    dst_path.write_text(new_text, encoding="utf-8")
    return len(updated_keys), sorted(updated_keys)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sync SharePoint/Graph app credentials between .env files")
    ap.add_argument("--src", required=True, help="Source .env path (known-good)")
    ap.add_argument("--dst", required=True, help="Destination .env path (this repo)")
    args = ap.parse_args()

    src_path = Path(args.src)
    dst_path = Path(args.dst)

    src = _parse_env(src_path.read_text(encoding="utf-8"))

    # Source may use either SHAREPOINT_* keys or legacy AZURE_* keys.
    tenant = src.get("SHAREPOINT_TENANT_ID") or src.get("SHAREPOINT_TENANT") or src.get("AZURE_TENANT_ID") or ""
    client = src.get("SHAREPOINT_CLIENT_ID") or src.get("AZURE_CLIENT_ID") or ""
    secret = src.get("SHAREPOINT_CLIENT_SECRET") or src.get("SHAREPOINT_SECRET") or src.get("AZURE_CLIENT_SECRET") or ""

    if not (tenant and client and secret):
        missing = []
        if not tenant:
            missing.append("tenant")
        if not client:
            missing.append("client")
        if not secret:
            missing.append("secret")
        raise SystemExit(f"Source .env is missing required fields: {', '.join(missing)}")

    # Apply to destination as SHAREPOINT_* only.
    updates = {
        "SHAREPOINT_TENANT_ID": tenant,
        "SHAREPOINT_CLIENT_ID": client,
        "SHAREPOINT_CLIENT_SECRET": secret,
    }

    count, keys = _update_env_file(dst_path, updates)

    # Safe output only
    print("[OK] Synced env keys")
    print(f"- dst: {dst_path}")
    print(f"- updated_keys: {count}")
    print(f"- client_secret_len: {len(secret)}")
    print(f"- keys: {', '.join(keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
