"""Application startup script."""
import uvicorn
from config import APP_HOST, APP_PORT, DEBUG

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=DEBUG
    )

