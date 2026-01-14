from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import quote


# Ensure repo root is importable even if the script is run with a different CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.sharepoint_service import SharePointService  # noqa: E402


def _pick_drive_by_name(drives: list[dict], drive_name: str | None) -> dict:
    if not drives:
        raise RuntimeError("No drives found")

    if drive_name:
        target = drive_name.strip().lower()
        for d in drives:
            if str(d.get("name") or "").strip().lower() == target:
                return d

    # Default preference: Documents / Shared Documents
    for d in drives:
        name = str(d.get("name") or "").strip().lower()
        if name in ("documents", "shared documents"):
            return d

    return drives[0]


def cmd_info(svc: SharePointService) -> int:
    site = svc._get_site()
    print(f"Site: {site.get('displayName')}  id={site.get('id')}")

    drives = svc._list_drives()
    print("Drives:")
    for d in drives:
        print(f"- {d.get('name')}  id={d.get('id')}")
    return 0


def cmd_validate(svc: SharePointService, folder: str | None) -> int:
    result = svc.validate_connection(folder_path=folder)
    print(result)
    return 0 if result.get("ok") else 1


def cmd_list_root(svc: SharePointService, drive_name: str | None) -> int:
    site = svc._get_site()
    drives = svc._list_drives()
    drive = _pick_drive_by_name(drives, drive_name)
    print(f"Using drive: {drive.get('name')}  id={drive.get('id')}")

    drive_id = str(drive.get("id") or "")
    items = svc._graph_request(
        "GET",
        f"/drives/{drive_id}/root/children?$select=id,name,webUrl,file,folder,parentReference",
    ).get("value", [])

    if not items:
        print("(No items in drive root)")
        return 0

    for it in items:
        kind = "folder" if "folder" in it else "file"
        print(f"- [{kind}] {it.get('name')}")

    return 0


def cmd_list_path(svc: SharePointService, drive_name: str | None, folder_path: str) -> int:
    site = svc._get_site()
    drives = svc._list_drives()
    drive = _pick_drive_by_name(drives, drive_name)
    print(f"Using drive: {drive.get('name')}  id={drive.get('id')}")
    print(f"Path: /{folder_path.lstrip('/')}" if folder_path else "Path: /(root)")

    drive_id = str(drive.get("id") or "")
    p = (folder_path or "").strip().lstrip("/")
    if not p:
        return cmd_list_root(svc, drive_name)

    encoded = quote(p, safe="/")
    items = svc._graph_request(
        "GET",
        f"/drives/{drive_id}/root:/{encoded}:/children?$select=id,name,webUrl,file,folder,parentReference",
    ).get("value", [])

    if not items:
        print("(No items)")
        return 0

    for it in items:
        kind = "folder" if "folder" in it else "file"
        print(f"- [{kind}] {it.get('name')}")

    return 0


def cmd_search_pdf(svc: SharePointService, drive_name: str | None, limit: int) -> int:
    site = svc._get_site()
    drives = svc._list_drives()
    drive = _pick_drive_by_name(drives, drive_name)
    print(f"Using drive: {drive.get('name')}  id={drive.get('id')}")

    drive_id = str(drive.get("id") or "")
    # Escape single quotes for OData literal
    q = ".pdf".replace("'", "''")
    payload = svc._graph_request(
        "GET",
        f"/drives/{drive_id}/root/search(q='{q}')?$top={int(limit)}&$select=id,name,webUrl,file,folder,parentReference",
    )
    results = list((payload or {}).get("value") or [])

    if not results:
        print("No PDF results found.")
        return 0

    for r in results:
        web_url = r.get("webUrl", "")
        print(f"- {r.get('name')}\n  {web_url}")

    return 0


def _paged_get(svc: SharePointService, url_or_path: str) -> list[dict]:
    items: list[dict] = []
    next_url: str | None = url_or_path
    while next_url:
        payload = svc._graph_request("GET", next_url)
        value = payload.get("value") if isinstance(payload, dict) else None
        items.extend(list(value or []))
        next_url = payload.get("@odata.nextLink") if isinstance(payload, dict) else None
    return items


def _list_children_by_drive_path(svc: SharePointService, drive_id: str, folder_path: str) -> list[dict]:
    p = (folder_path or "").strip().lstrip("/")
    if not p:
        return _paged_get(
            svc,
            f"/drives/{drive_id}/root/children?$select=id,name,webUrl,file,folder,parentReference",
        )
    encoded = quote(p, safe="/")
    return _paged_get(
        svc,
        f"/drives/{drive_id}/root:/{encoded}:/children?$select=id,name,webUrl,file,folder,parentReference",
    )


def cmd_tree(
    svc: SharePointService,
    drive_name: str | None,
    start_path: str,
    depth: int,
    include_files: bool,
) -> int:
    drives = svc._list_drives()
    drive = _pick_drive_by_name(drives, drive_name)
    drive_id = str(drive.get("id") or "")

    start_path_norm = (start_path or "").strip().strip("/")
    display_root = "/" + start_path_norm if start_path_norm else "/"

    print(f"Using drive: {drive.get('name')}  id={drive.get('id')}")
    print(display_root)

    visited: set[str] = set()

    def walk(folder_path: str, level: int) -> None:
        if level >= depth:
            return

        children = _list_children_by_drive_path(svc, drive_id=drive_id, folder_path=folder_path)
        # Stable-ish output: folders first, then files; both by name.
        def sort_key(it: dict) -> tuple[int, str]:
            is_folder = 0 if "folder" in it else 1
            return (is_folder, str(it.get("name") or "").lower())

        for it in sorted(children, key=sort_key):
            name = str(it.get("name") or "")
            if not name:
                continue

            is_folder = "folder" in it
            if (not include_files) and (not is_folder):
                continue

            prefix = "  " * (level + 1)
            kind = "folder" if is_folder else "file"
            print(f"{prefix}- [{kind}] {name}")

            if is_folder:
                item_id = str(it.get("id") or "")
                if item_id:
                    if item_id in visited:
                        continue
                    visited.add(item_id)

                next_path = f"{folder_path.strip('/')}" if folder_path else ""
                next_path = f"{next_path}/{name}".strip("/")
                walk(next_path, level + 1)

    walk(start_path_norm, 0)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SharePoint via SharePointService (Microsoft Graph)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Resolve site and list drives")

    p_val = sub.add_parser("validate", help="Validate token + site resolution (+ optional folder probe)")
    p_val.add_argument("--folder", default=None, help="Server-relative folder path (e.g. '/sites/APOCR/Shared Documents')")

    p_list = sub.add_parser("list-root", help="List items in drive root")
    p_list.add_argument("--drive", default=None, help="Drive name override (e.g. Documents)")

    p_listp = sub.add_parser("list-path", help="List items in a folder path")
    p_listp.add_argument("path", help="Folder path relative to drive root (e.g. 'General/Invoices')")
    p_listp.add_argument("--drive", default=None, help="Drive name override (e.g. Documents)")

    p_search = sub.add_parser("search-pdf", help="Search for PDFs in a drive")
    p_search.add_argument("--drive", default=None, help="Drive name override (e.g. Documents)")
    p_search.add_argument("--limit", type=int, default=50, help="Max results")

    p_tree = sub.add_parser("tree", help="Show drive folder structure (recursive)")
    p_tree.add_argument("--drive", default=None, help="Drive name override (e.g. Documents)")
    p_tree.add_argument("--path", default="", help="Start folder path relative to drive root (e.g. 'Production')")
    p_tree.add_argument("--depth", type=int, default=4, help="Max recursion depth")
    p_tree.add_argument("--include-files", action="store_true", help="Include files in output")

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    svc = SharePointService()

    if args.command == "info":
        return cmd_info(svc)
    if args.command == "validate":
        return cmd_validate(svc, folder=args.folder)
    if args.command == "list-root":
        return cmd_list_root(svc, drive_name=args.drive)
    if args.command == "list-path":
        return cmd_list_path(svc, drive_name=args.drive, folder_path=args.path)
    if args.command == "search-pdf":
        return cmd_search_pdf(svc, drive_name=args.drive, limit=args.limit)
    if args.command == "tree":
        return cmd_tree(
            svc,
            drive_name=args.drive,
            start_path=args.path,
            depth=max(int(args.depth), 0),
            include_files=bool(args.include_files),
        )

    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
