"""SharePoint service integration via Microsoft Graph API."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse, parse_qs, unquote, quote

import requests
from dotenv import load_dotenv

load_dotenv()


class SharePointService:
    """SharePoint integration helper (Microsoft Graph)."""

    GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"

    def __init__(self):
        self._token_cache: Dict[str, Any] = {}
        self._site_cache: Dict[str, Any] = {}
        self._drives_cache: Dict[str, Any] = {}

        self.reload_from_env()

        download_dir = os.getenv("SHAREPOINT_DOWNLOAD_DIR") or "sharepoint_downloads"
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)

    @staticmethod
    def _env_clean(value: Optional[str]) -> str:
        """Clean a value read from environment/.env.

        Removes trailing inline comments, trims whitespace, and strips common quote wrappers.
        """
        if value is None:
            return ""
        cleaned = value.split("#", 1)[0].strip()
        if (len(cleaned) >= 2) and (cleaned[0] == cleaned[-1]) and cleaned[0] in ("'", '"'):
            cleaned = cleaned[1:-1].strip()
        return cleaned

    @staticmethod
    def _derive_site_url_from_link(link: str) -> str:
        if not link:
            return ""
        try:
            parsed = urlparse(link)
            if not parsed.scheme or not parsed.netloc:
                return ""
            parts = [p for p in (parsed.path or "").split("/") if p]
            if len(parts) >= 2 and parts[0].lower() == "sites":
                return f"{parsed.scheme}://{parsed.netloc}/sites/{parts[1]}"
            return f"{parsed.scheme}://{parsed.netloc}"
        except Exception:
            return ""

    def reload_from_env(self) -> None:
        """Reload credentials/settings from environment variables."""
        # Prefer legacy keys, but support the newer keys used in this repo's .env.
        self.client_id = self._env_clean(os.getenv("SHAREPOINT_CLIENT_ID"))
        secret_from_primary = self._env_clean(os.getenv("SHAREPOINT_CLIENT_SECRET"))
        if secret_from_primary:
            self.client_secret = secret_from_primary
            self.client_secret_source = "SHAREPOINT_CLIENT_SECRET"
        else:
            self.client_secret = self._env_clean(os.getenv("SHAREPOINT_SECRET"))
            self.client_secret_source = "SHAREPOINT_SECRET"
        self.tenant_id = self._env_clean(os.getenv("SHAREPOINT_TENANT_ID")) or self._env_clean(os.getenv("SHAREPOINT_TENANT"))

        self.site_url = self._env_clean(os.getenv("SHAREPOINT_SITE_URL"))
        if not self.site_url:
            # Use SHAREPOINT_LINK as a fallback and derive the site root.
            self.site_url = self._derive_site_url_from_link(self._env_clean(os.getenv("SHAREPOINT_LINK")) or "")

        # Reset caches when configuration changes
        self._token_cache = {}
        self._site_cache = {}
        self._drives_cache = {}

    @staticmethod
    def _normalize_server_relative_path(file_url_or_path: str) -> str:
        """Accept either a server-relative path or a full SharePoint URL and return a server-relative path."""
        raw = (file_url_or_path or "").strip()
        if not raw:
            return ""

        # Already server-relative
        if raw.startswith("/") and not raw.lower().startswith("/http"):
            return raw

        # Try full URL
        try:
            parsed = urlparse(raw)
            if parsed.scheme and parsed.netloc:
                # Common share links contain `?id=/sites/...`.
                qs = parse_qs(parsed.query or "")
                if "id" in qs and qs["id"]:
                    return unquote(qs["id"][0])

                path = unquote(parsed.path or "")
                # Handle special share link patterns like /:b:/r/sites/... or /:f:/r/sites/...
                if path.startswith("/:") and "/sites/" in path:
                    path = path[path.index("/sites/"):]
                return path if path.startswith("/") else f"/{path}"
        except Exception:
            pass

        # Best-effort: treat as relative and prefix '/'
        return raw if raw.startswith("/") else f"/{raw}"

    def _acquire_graph_token(self) -> str:
        """Acquire an Azure AD access token for Microsoft Graph via client credentials."""
        cached = self._token_cache or {}
        access_token = cached.get("access_token")
        expires_at = cached.get("expires_at", 0)
        now = int(time.time())
        if access_token and now < int(expires_at) - 60:
            return str(access_token)

        tenant = (self.tenant_id or "").strip()
        if not tenant:
            raise Exception("Missing tenant id (SHAREPOINT_TENANT_ID or SHAREPOINT_TENANT)")

        # Prefer MSAL if available (matches the known-working connector scripts).
        # Note: MSAL won't fix invalid secrets, but it avoids edge cases around encoding.
        try:
            import msal  # type: ignore

            authority = f"https://login.microsoftonline.com/{tenant}"
            app = msal.ConfidentialClientApplication(
                str(self.client_id or ""),
                authority=authority,
                client_credential=str(self.client_secret or ""),
            )
            result_any = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
            if not isinstance(result_any, dict):
                raise Exception(str(result_any))

            if "access_token" not in result_any:
                raise Exception(str(result_any))

            token = result_any.get("access_token")
            expires_in = int(result_any.get("expires_in") or 0)
            self._token_cache = {
                "access_token": token,
                "expires_at": int(time.time()) + max(expires_in, 0),
                "raw": result_any,
            }
            return str(token)
        except ModuleNotFoundError:
            pass

        token_url = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
        scope = "https://graph.microsoft.com/.default"

        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
            "scope": scope,
        }
        resp = requests.post(token_url, data=data, timeout=30)
        try:
            payload = resp.json()
        except Exception:
            payload = {"error": "invalid_response", "error_description": resp.text}

        if resp.status_code >= 400:
            # Keep message machine-readable for the validator (it extracts AADSTS codes)
            raise Exception(str(payload))

        token = payload.get("access_token")
        if not token:
            raise Exception(str(payload))

        expires_in = int(payload.get("expires_in") or 0)
        self._token_cache = {
            "access_token": token,
            "expires_at": int(time.time()) + max(expires_in, 0),
            "raw": payload,
        }
        return str(token)

    def _graph_headers(self) -> Dict[str, str]:
        token = self._acquire_graph_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

    def _graph_request(self, method: str, path: str, *, stream: bool = False) -> Any:
        url = path if path.lower().startswith("http") else f"{self.GRAPH_BASE_URL}{path}"
        resp = requests.request(method.upper(), url, headers=self._graph_headers(), timeout=60, stream=stream)

        if stream:
            if resp.status_code >= 400:
                try:
                    payload = resp.json()
                except Exception:
                    payload = {"error": "invalid_response", "error_description": resp.text}
                raise Exception(str(payload))
            return resp

        try:
            payload = resp.json() if resp.text else {}
        except Exception:
            payload = {"error": "invalid_response", "error_description": resp.text}

        if resp.status_code >= 400:
            raise Exception(str(payload))

        return payload

    def _get_site_ref(self) -> Tuple[str, str]:
        """Return (hostname, site_path) for Graph site lookup."""
        parsed = urlparse(self.site_url or "")
        if not parsed.scheme or not parsed.netloc:
            raise Exception("Invalid SHAREPOINT_SITE_URL")
        hostname = parsed.netloc
        site_path = parsed.path or "/"
        if not site_path.startswith("/"):
            site_path = f"/{site_path}"
        return hostname, site_path

    def _get_site(self) -> Dict[str, Any]:
        """Get Graph site object (cached)."""
        cache_key = (self.site_url or "").strip().lower()
        if cache_key and cache_key in self._site_cache:
            return dict(self._site_cache[cache_key])

        hostname, site_path = self._get_site_ref()
        if site_path in ("", "/"):
            path = f"/sites/{hostname}:/?$select=id,displayName,webUrl"
        else:
            # e.g. /sites/{hostname}:/sites/{siteName}
            path = f"/sites/{hostname}:{site_path}?$select=id,displayName,webUrl"

        site = self._graph_request("GET", path)
        self._site_cache[cache_key] = site
        return dict(site)

    def _list_drives(self) -> List[Dict[str, Any]]:
        """List document libraries (drives) for the configured site (cached)."""
        cache_key = (self.site_url or "").strip().lower()
        cached = self._drives_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        site = self._get_site()
        site_id = site.get("id")
        if not site_id:
            raise Exception("Unable to resolve SharePoint site id via Graph")

        drives = self._graph_request("GET", f"/sites/{site_id}/drives?$select=id,name,webUrl")
        value = drives.get("value") if isinstance(drives, dict) else None
        drive_list = list(value or [])
        self._drives_cache[cache_key] = drive_list
        return drive_list

    @staticmethod
    def _share_id_from_url(url: str) -> str:
        """Encode a sharing URL into a Graph shareId (u!base64url)."""
        raw = (url or "").encode("utf-8")
        b64 = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        return f"u!{b64}"

    @staticmethod
    def _server_relative_from_web_url(web_url: str) -> str:
        try:
            parsed = urlparse(web_url or "")
            return unquote(parsed.path or "") or ""
        except Exception:
            return ""

    def _pick_default_drive(self, drives: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not drives:
            return None
        for d in drives:
            name = str(d.get("name") or "").strip().lower()
            if name in ("documents", "shared documents"):
                return d
        for d in drives:
            name = str(d.get("name") or "").strip().lower()
            if "document" in name:
                return d
        return drives[0]

    def _map_server_relative_to_drive(self, server_relative_path: str) -> Tuple[str, str]:
        """Map a server-relative path to (drive_id, drive_relative_path)."""
        p = (server_relative_path or "").strip() or "/"
        if not p.startswith("/"):
            p = f"/{p}"

        drives = self._list_drives()
        default_drive = self._pick_default_drive(drives)
        if not drives or not default_drive:
            raise Exception("No document libraries (drives) found for the site")

        # Prefer matching by drive webUrl path prefix.
        for d in drives:
            web_url = str(d.get("webUrl") or "")
            root_path = self._server_relative_from_web_url(web_url).rstrip("/")
            if not root_path:
                continue
            if p == root_path:
                return str(d.get("id")), ""
            if p.startswith(root_path + "/"):
                rel = p[len(root_path) + 1 :]
                return str(d.get("id")), rel

        # Fallback: common default library URL segment is "Shared Documents".
        low = p.lower()
        marker = "/shared documents/"
        if marker in low:
            idx = low.index(marker)
            rel = p[idx + len(marker) :]
            return str(default_drive.get("id")), rel

        # If user passed root, use default drive root.
        if p in ("/", str(urlparse(self.site_url or "").path or "/").rstrip("/")):
            return str(default_drive.get("id")), ""

        # Last resort: try using the whole path (minus leading slash) as drive-relative.
        return str(default_drive.get("id")), p.lstrip("/")

    def _resolve_drive_item_from_server_relative(self, server_relative_path: str) -> Dict[str, Any]:
        drive_id, drive_rel = self._map_server_relative_to_drive(server_relative_path)
        if drive_rel:
            encoded = quote(drive_rel, safe="/")
            item = self._graph_request(
                "GET",
                f"/drives/{drive_id}/root:/{encoded}?$select=id,name,webUrl,file,folder,parentReference",
            )
        else:
            item = self._graph_request(
                "GET",
                f"/drives/{drive_id}/root?$select=id,name,webUrl,file,folder,parentReference",
            )

        item["_drive_id"] = drive_id
        return item

    def _resolve_drive_item(self, file_url: str, folder_path: Optional[str] = None) -> Dict[str, Any]:
        raw = (file_url or "").strip()
        if not raw:
            raise Exception("Missing SharePoint file URL/path")

        # If a folder_path is provided and file_url looks like a relative name, join them.
        if folder_path and not raw.startswith("/"):
            parsed = urlparse(raw)
            if not (parsed.scheme and parsed.netloc):
                folder_norm = self._normalize_server_relative_path(folder_path)
                raw = f"{folder_norm.rstrip('/')}/{raw.lstrip('/')}"

        parsed = urlparse(raw)

        # If it's a URL, try resolving via /shares first (works well for sharing links).
        if parsed.scheme and parsed.netloc:
            try:
                share_id = self._share_id_from_url(raw)
                item = self._graph_request(
                    "GET",
                    f"/shares/{share_id}/driveItem?$select=id,name,webUrl,file,folder,parentReference",
                )
                parent = item.get("parentReference") or {}
                drive_id = parent.get("driveId")
                if drive_id:
                    item["_drive_id"] = drive_id
                return item
            except Exception:
                # Fall back to server-relative mapping.
                pass

        server_rel = self._normalize_server_relative_path(raw)
        return self._resolve_drive_item_from_server_relative(server_rel)
    
    async def download_file(
        self, 
        file_url: str, 
        folder_path: Optional[str] = None
    ) -> Path:
        """
        Download a PDF file from SharePoint.
        
        Args:
            file_url: SharePoint file URL or server-relative path
            folder_path: SharePoint folder path (optional)
        
        Returns:
            Local file path
        """
        try:
            item = self._resolve_drive_item(file_url, folder_path)
            drive_id = item.get("_drive_id")
            item_id = item.get("id")
            name = item.get("name")
            if not drive_id or not item_id:
                raise Exception("Unable to resolve drive item for the provided URL/path")

            # Download file content
            resp = self._graph_request("GET", f"/drives/{drive_id}/items/{item_id}/content", stream=True)

            local_filename = str(name or "").strip() or Path(str(file_url)).name
            local_path = self.download_dir / local_filename

            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
            
            return local_path
            
        except Exception as e:
            raise Exception(f"Error downloading file from SharePoint: {str(e)}")

    def browse_folder(self, folder_path: str = "/") -> Dict[str, Any]:
        """List subfolders and files for a given server-relative folder path."""
        try:
            folder_path_norm = self._normalize_server_relative_path(folder_path)
            item = self._resolve_drive_item_from_server_relative(folder_path_norm)
            drive_id = item.get("_drive_id")
            item_id = item.get("id")

            if not drive_id or not item_id:
                raise Exception("Unable to resolve SharePoint folder via Graph")

            if item.get("file") is not None:
                raise Exception("Provided path appears to be a file, not a folder")

            children = self._graph_request(
                "GET",
                f"/drives/{drive_id}/items/{item_id}/children?$select=name,webUrl,file,folder",
            )
            values = list((children or {}).get("value") or [])

            folder_items: List[Dict[str, Any]] = []
            file_items: List[Dict[str, Any]] = []
            for child in values:
                name = child.get("name")
                web_url = child.get("webUrl")
                server_rel = self._server_relative_from_web_url(str(web_url or ""))

                if child.get("folder") is not None:
                    folder_items.append({
                        "name": name,
                        "server_relative_url": server_rel,
                    })
                else:
                    is_pdf = bool(name and str(name).lower().endswith(".pdf"))
                    file_items.append({
                        "name": name,
                        "server_relative_url": server_rel,
                        "is_pdf": is_pdf,
                    })

            # Sort folders/files for nicer UX
            folder_items.sort(key=lambda x: (x.get("name") or "").lower())
            file_items.sort(key=lambda x: (not bool(x.get("is_pdf")), (x.get("name") or "").lower()))

            return {
                "path": folder_path_norm,
                "folders": folder_items,
                "files": file_items,
            }
        except Exception as e:
            raise Exception(f"Error browsing SharePoint folder: {str(e)}")

    def _paged_children(
        self,
        drive_id: str,
        item_id: str,
        *,
        select: str = "id,name,webUrl,file,folder",
        page_size: int = 200,
    ) -> List[Dict[str, Any]]:
        """Return all children of a drive item, following @odata.nextLink if present."""
        if not drive_id or not item_id:
            return []

        items: List[Dict[str, Any]] = []
        url: str = f"/drives/{drive_id}/items/{item_id}/children?$select={select}&$top={int(page_size)}"
        while url:
            payload = self._graph_request("GET", url)
            values = (payload or {}).get("value") if isinstance(payload, dict) else None
            items.extend(list(values or []))
            next_link = (payload or {}).get("@odata.nextLink") if isinstance(payload, dict) else None
            url = str(next_link) if next_link else ""

        return items

    def get_folder_tree(
        self,
        folder_path: str = "/",
        *,
        depth: int = 4,
        include_files: bool = False,
        max_nodes: int = 1500,
    ) -> Dict[str, Any]:
        """Build a folder tree (server-relative paths) for frontend display.

        - Uses depth-limited recursion.
        - Follows Graph paging (@odata.nextLink).
        - Caps total visited nodes to avoid huge responses.
        """
        if depth < 0:
            depth = 0
        if depth > 25:
            depth = 25
        if max_nodes < 50:
            max_nodes = 50
        if max_nodes > 20000:
            max_nodes = 20000

        folder_path_norm = self._normalize_server_relative_path(folder_path)
        root_item = self._resolve_drive_item_from_server_relative(folder_path_norm)
        drive_id = str(root_item.get("_drive_id") or "")
        item_id = str(root_item.get("id") or "")
        if not drive_id or not item_id:
            raise Exception("Unable to resolve SharePoint folder via Graph")
        if root_item.get("file") is not None:
            raise Exception("Provided path appears to be a file, not a folder")

        visited = 0

        def build(item_name: str, server_rel_path: str, drive_id_local: str, item_id_local: str, level: int) -> Dict[str, Any]:
            nonlocal visited
            visited += 1
            if visited > max_nodes:
                return {
                    "name": item_name,
                    "path": server_rel_path,
                    "folders": [],
                    "files": [],
                    "truncated": True,
                }

            node: Dict[str, Any] = {
                "name": item_name,
                "path": server_rel_path,
                "folders": [],
                "files": [],
            }

            if level <= 0:
                return node

            children = self._paged_children(drive_id_local, item_id_local)

            folders: List[Dict[str, Any]] = []
            files: List[Dict[str, Any]] = []
            for child in children:
                name = child.get("name")
                web_url = child.get("webUrl")
                child_server_rel = self._server_relative_from_web_url(str(web_url or ""))
                if child.get("folder") is not None:
                    folders.append({
                        "name": name,
                        "path": child_server_rel,
                        "drive_id": drive_id_local,
                        "item_id": child.get("id"),
                    })
                else:
                    if include_files:
                        is_pdf = bool(name and str(name).lower().endswith(".pdf"))
                        files.append({
                            "name": name,
                            "path": child_server_rel,
                            "is_pdf": is_pdf,
                        })

            folders.sort(key=lambda x: (str(x.get("name") or "").lower()))
            files.sort(key=lambda x: (not bool(x.get("is_pdf")), str(x.get("name") or "").lower()))

            node["files"] = files
            node_folders: List[Dict[str, Any]] = []
            for f in folders:
                if visited > max_nodes:
                    node["truncated"] = True
                    break
                node_folders.append(
                    build(
                        str(f.get("name") or ""),
                        str(f.get("path") or ""),
                        str(f.get("drive_id") or ""),
                        str(f.get("item_id") or ""),
                        level - 1,
                    )
                )

            node["folders"] = node_folders
            return node

        return {
            "root": build(str(root_item.get("name") or ""), folder_path_norm, drive_id, item_id, depth),
            "visited": visited,
            "depth": depth,
            "include_files": bool(include_files),
            "max_nodes": max_nodes,
        }

    def validate_connection(self, folder_path: Optional[str] = None) -> Dict[str, Any]:
        """Validate credentials and basic site access.

        Returns a structured dict suitable for UI display.
        """
        # Basic config presence checks
        missing = []
        if not self.client_id:
            missing.append("SHAREPOINT_CLIENT_ID")
        if not self.client_secret:
            missing.append("SHAREPOINT_CLIENT_SECRET or SHAREPOINT_SECRET")
        if not self.tenant_id:
            missing.append("SHAREPOINT_TENANT_ID or SHAREPOINT_TENANT")
        if not self.site_url:
            missing.append("SHAREPOINT_SITE_URL or SHAREPOINT_LINK")

        secret_val = self.client_secret or ""
        secret_has_whitespace = any(ch.isspace() for ch in secret_val)
        secret_looks_like_guid = False
        s = secret_val.strip()
        if len(s) == 36 and s.count("-") == 4:
            secret_looks_like_guid = True

        auth_mode = "graph"

        if missing:
            return {
                "ok": False,
                "site_url": self.site_url or "",
                "auth_mode": auth_mode,
                "config": {
                    "client_id": self.client_id or "",
                    "tenant": self.tenant_id or "",
                    "secret_source": getattr(self, "client_secret_source", ""),
                    "secret_length": len(secret_val),
                    "secret_has_whitespace": secret_has_whitespace,
                    "secret_looks_like_guid": secret_looks_like_guid,
                },
                "error": {
                    "message": "Missing required SharePoint settings",
                    "missing": missing,
                },
            }

        try:
            site = self._get_site()
            title = site.get("displayName")
            url = site.get("webUrl")

            folder_test_path = None
            if folder_path:
                folder_test_path = self._normalize_server_relative_path(folder_path)
            elif os.getenv("SHAREPOINT_FOLDER"):
                folder_test_path = self._normalize_server_relative_path(os.getenv("SHAREPOINT_FOLDER") or "")

            folder_probe = None
            if folder_test_path:
                try:
                    # resolve folder and list children as a probe
                    folder_item = self._resolve_drive_item_from_server_relative(folder_test_path)
                    if folder_item.get("file") is not None:
                        raise Exception("Path resolves to a file")
                    folder_probe = {
                        "path": folder_test_path,
                        "exists": True,
                        "name": folder_item.get("name"),
                    }
                except Exception as fe:
                    folder_probe = {
                        "path": folder_test_path,
                        "exists": False,
                        "error": str(fe),
                    }

            return {
                "ok": True,
                "site_url": self.site_url,
                "auth_mode": auth_mode,
                "config": {
                    "client_id": self.client_id or "",
                    "tenant": self.tenant_id or "",
                    "secret_source": getattr(self, "client_secret_source", ""),
                    "secret_length": len(secret_val),
                    "secret_has_whitespace": secret_has_whitespace,
                    "secret_looks_like_guid": secret_looks_like_guid,
                },
                "web": {
                    "title": title,
                    "url": url,
                },
                "folder_probe": folder_probe,
            }
        except Exception as e:
            return {
                "ok": False,
                "site_url": self.site_url or "",
                "auth_mode": auth_mode,
                "config": {
                    "client_id": self.client_id or "",
                    "tenant": self.tenant_id or "",
                    "secret_source": getattr(self, "client_secret_source", ""),
                    "secret_length": len(secret_val),
                    "secret_has_whitespace": secret_has_whitespace,
                    "secret_looks_like_guid": secret_looks_like_guid,
                },
                "error": {
                    "message": str(e),
                },
            }
    
    def list_pdf_files(self, folder_path: str = "/") -> list:
        """
        List all PDF files in a SharePoint folder.
        
        Args:
            folder_path: SharePoint folder path
        
        Returns:
            List of PDF file URLs
        """
        try:
            data = self.browse_folder(folder_path)
            files = list((data or {}).get("files") or [])
            return [
                f.get("server_relative_url")
                for f in files
                if f.get("is_pdf") and f.get("server_relative_url")
            ]
            
        except Exception as e:
            raise Exception(f"Error listing SharePoint files: {str(e)}")

