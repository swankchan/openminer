# Deployment (Make It Usable Online)

This app is a FastAPI (ASGI) service served by Uvicorn. The easiest way to run it online is to deploy the Docker container and put it behind an HTTPS reverse proxy.

## Important security note

This app currently has no login/authentication. If you expose it to the public internet, anyone who can reach the URL can upload documents and trigger processing.

Recommended approaches:
- Keep it private (VPN / corporate network / IP allow-list).
- Or put a reverse proxy in front with authentication (Basic Auth / SSO).

## Option 1: Deploy with Docker on a VM (recommended)

### 1) Provision a server

Any Linux VM works (e.g., Ubuntu). Open ports:
- 80 and 443 (for HTTPS)

### 2) Install Docker

Follow Docker’s official installation steps for your distro.

### 3) Clone and configure

- Clone your repo onto the server.
- Create a `.env` file (do not commit it). You can start from `.env.example`.

At minimum for Azure OpenAI mode:
- `AI_SERVICE=azure_openai`
- `AZURE_OPENAI_ENDPOINT=...`
- `AZURE_OPENAI_API_KEY=...`
- `AZURE_OPENAI_DEPLOYMENT_NAME=...`

If using SharePoint integration, also set:
- `SHAREPOINT_SITE_URL=...`
- `SHAREPOINT_CLIENT_ID=...`
- `SHAREPOINT_CLIENT_SECRET=...`
- `SHAREPOINT_TENANT_ID=...`

### 4) Run the container

From the repo root:

- `docker compose up -d --build`

This uses the existing docker-compose file and exposes port 8000 in the container to `OPENMINER_HOST_PORT` on the host (default 8001).

### 5) Add HTTPS (reverse proxy)

Run a reverse proxy in front (Caddy / Nginx / Traefik) and route a domain to your container.

Key requirements:
- Proxy WebSockets (for progress updates).
- Keep a single backend instance (the app uses in-memory progress; scaling requires shared state).

## Option 2: Deploy to a Docker hosting provider

Most providers can deploy directly from your GitHub repo using the Dockerfile.

What you’ll need:
- Set environment variables in the provider UI (copy from `.env.example`).
- Expose container port 8000.
- Ensure WebSocket support is enabled.

Notes:
- MinerU can make the image large if `INSTALL_MINERU=1` is used during build.
- If you only want to demo the UI/API without MinerU installed in the image, build without that arg.

## Verifying after deployment

- Open the site URL and load the UI.
- Check the runtime status subtitle (it calls the backend).
- Try a small PDF upload.
- If SharePoint is enabled, validate the SharePoint connection from the UI.
