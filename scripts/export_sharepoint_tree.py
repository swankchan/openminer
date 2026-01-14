from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


def _render_tree(root: dict, *, include_files: bool) -> list[str]:
    lines: list[str] = []

    def walk(node: dict, indent: str) -> None:
        name = str(node.get("name") or "").strip() or "(unnamed)"
        path = str(node.get("path") or "").strip()
        suffix = "/" if path else "/"
        lines.append(f"{indent}- {name}{suffix}")

        next_indent = indent + "  "

        if include_files:
            for f in list(node.get("files") or []):
                fname = str(f.get("name") or "").strip() or "(unnamed)"
                lines.append(f"{next_indent}- {fname}")

        for child in list(node.get("folders") or []):
            walk(child, next_indent)

        if node.get("truncated"):
            lines.append(f"{next_indent}- (truncated)")

    walk(root, "")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export SharePoint folder/file hierarchy (via Microsoft Graph) to Markdown"
    )
    ap.add_argument(
        "--out",
        default="docs/SHAREPOINT_TREE.md",
        help="Output markdown path (default: docs/SHAREPOINT_TREE.md)",
    )
    ap.add_argument(
        "--folder",
        default=None,
        help="Server-relative folder path to start from (default: SHAREPOINT_FOLDER or '/')",
    )
    ap.add_argument("--depth", type=int, default=4, help="Folder recursion depth (default: 4)")
    ap.add_argument(
        "--include-files",
        action="store_true",
        help="Include files in the output (default: false)",
    )
    ap.add_argument(
        "--max-nodes",
        type=int,
        default=2000,
        help="Max folders/files visited (default: 2000)",
    )
    args = ap.parse_args()

    # Load .env from repo root when launched from anywhere.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    load_dotenv(repo_root / ".env")

    from services.sharepoint_service import SharePointService

    sp = SharePointService()

    folder = args.folder
    if folder is None:
        folder = (sp._env_clean(__import__("os").getenv("SHAREPOINT_FOLDER")) or "/").strip()  # noqa: SLF001

    result = sp.get_folder_tree(
        folder,
        depth=int(args.depth),
        include_files=bool(args.include_files),
        max_nodes=int(args.max_nodes),
    )

    root = dict(result.get("root") or {})
    visited = int(result.get("visited") or 0)

    # Safe header: no secrets.
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    site_url = (sp.site_url or "").strip()

    out_path = (repo_root / str(args.out)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md_lines: list[str] = []
    md_lines.append("# SharePoint Hierarchy\n")
    md_lines.append(f"Generated: {now}  ")
    md_lines.append(f"Site: {site_url or '(not set)'}  ")
    md_lines.append(f"Start folder: {folder}  ")
    md_lines.append(
        f"Params: depth={int(args.depth)}, include_files={bool(args.include_files)}, max_nodes={int(args.max_nodes)}  "
    )
    md_lines.append(f"Visited nodes: {visited}  \n")

    if root.get("truncated"):
        md_lines.append(
            "> Note: output was truncated due to max_nodes limit. Increase `--max-nodes` or reduce `--depth`.\n"
        )

    md_lines.append("## Tree\n")
    md_lines.extend(_render_tree(root, include_files=bool(args.include_files)))
    md_lines.append("")

    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[OK] Wrote SharePoint hierarchy to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
