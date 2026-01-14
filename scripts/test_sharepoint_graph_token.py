from __future__ import annotations

import base64
import json
import os
import sys
from datetime import datetime, timezone

from pathlib import Path

from dotenv import load_dotenv


# Ensure repo root is importable even if the script is run with a different CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _b64url_decode(segment: str) -> bytes:
    segment = segment.strip()
    # Add base64 padding
    segment += "=" * ((4 - (len(segment) % 4)) % 4)
    return base64.urlsafe_b64decode(segment.encode("utf-8"))


def _safe_jwt_payload(access_token: str) -> dict:
    parts = (access_token or "").split(".")
    if len(parts) < 2:
        return {}
    try:
        return json.loads(_b64url_decode(parts[1]).decode("utf-8"))
    except Exception:
        return {}


def main() -> int:
    # Ensure we load the repo .env, and allow environment variables to override.
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=True)

    from services.sharepoint_service import SharePointService  # local import after dotenv

    svc = SharePointService()

    # Acquire token (will use MSAL if installed)
    token = svc._acquire_graph_token()
    payload = _safe_jwt_payload(token)

    exp = payload.get("exp")
    exp_dt = None
    if isinstance(exp, int):
        exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc)

    # Print only safe metadata (no token string)
    print("[OK] Acquired Microsoft Graph access token")
    print(f"- token_length: {len(token)}")
    print(f"- tenant: {payload.get('tid') or ''}")
    print(f"- app_id: {payload.get('appid') or payload.get('azp') or ''}")
    print(f"- audience (aud): {payload.get('aud') or ''}")
    if exp_dt:
        print(f"- expires_utc: {exp_dt.isoformat()}")

    roles = payload.get("roles")
    if isinstance(roles, list) and roles:
        print(f"- roles: {', '.join([str(r) for r in roles])}")

    scp = payload.get("scp")
    if isinstance(scp, str) and scp.strip():
        print(f"- scopes: {scp}")

    # If site URL is configured, show that we can also resolve the site.
    if svc.site_url:
        site = svc._get_site()
        print("[OK] Resolved SharePoint site via Graph")
        print(f"- site_display_name: {site.get('displayName') or ''}")
        print(f"- site_web_url: {site.get('webUrl') or ''}")
    else:
        print("[INFO] SHAREPOINT_SITE_URL not set; token test only.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
