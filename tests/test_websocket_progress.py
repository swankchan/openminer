"""Test WebSocket progress functionality."""
import asyncio
import json
import sys
import uuid
import time
from pathlib import Path

import requests
import websockets

# Ensure repo root is importable when running from tests/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


async def test_websocket_progress():
    """Test WebSocket progress updates."""
    # Generate a task ID
    task_id = f"test_{uuid.uuid4().hex[:8]}"
    print(f"[Test] Task ID: {task_id}")

    # Connect to WebSocket
    uri = f"ws://localhost:8000/ws/progress/{task_id}"
    print(f"[Test] Connecting to WebSocket: {uri}")

    try:
        async with websockets.connect(uri) as websocket:
            print("[Test] WebSocket connected")

            # Send ping to verify connectivity
            await websocket.send("ping")
            response = await websocket.recv()
            print(f"[Test] Ping response: {response}")

            # Upload a file in another task
            async def upload_file():
                await asyncio.sleep(1)  # Wait for WebSocket connection to stabilize
                print("[Test] Starting file upload...")

                pdf_path = REPO_ROOT / "sample" / "B2J1.pdf"
                if not pdf_path.exists():
                    print(f"[Error] File does not exist: {pdf_path}")
                    return

                with open(pdf_path, "rb") as f:
                    files = {"file": ("B2J1.pdf", f, "application/pdf")}
                    data = {
                        "generate_csv": "true",
                        "task_id": task_id,
                    }
                    try:
                        # requests is synchronous; calling it directly blocks the asyncio event loop,
                        # which can prevent WebSocket from receiving real-time progress updates.
                        response = await asyncio.to_thread(
                            requests.post,
                            "http://localhost:8000/api/upload",
                            files=files,
                            data=data,
                            timeout=600,
                        )
                        print(f"[Test] Upload finished, status: {response.status_code}")
                        if response.status_code == 200:
                            result = response.json()
                            print(f"[Test] Processing result: {json.dumps(result, indent=2, ensure_ascii=False)}")
                        else:
                            print(f"[Test] Error: {response.text}")
                    except Exception as e:
                        print(f"[Test] Upload error: {e}")

            # Start upload task
            upload_task = asyncio.create_task(upload_file())

            # Receive progress updates
            progress_count = 0
            try:
                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=300)
                        data = json.loads(message)
                        progress_count += 1
                        percentage = data.get("percentage", 0)
                        message_text = data.get("message", "")
                        stage = data.get("stage", "")

                        print(f"[Progress #{progress_count}] {percentage}% - {message_text} (stage: {stage})")

                        if percentage >= 100:
                            print("[Test] Processing complete!")
                            break
                    except asyncio.TimeoutError:
                        print("[Test] Timed out waiting for progress updates")
                        break
            except websockets.exceptions.ConnectionClosed:
                print("[Test] WebSocket connection closed")

            # Wait for the upload task to finish
            await upload_task

    except Exception as e:
        print(f"[Error] WebSocket test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("WebSocket progress test")
    print("=" * 60)
    print()

    # Check whether the server is running
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        print(f"[Check] Server is running (status: {response.status_code})")
    except Exception as e:
        print(f"[Error] Cannot connect to server: {e}")
        print("[Hint] Please run first: python run.py")
        raise SystemExit(1)

    print()
    asyncio.run(test_websocket_progress())
