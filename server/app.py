"""
server/app.py — OpenEnv required entry point for multi-mode deployment.
"""
import os
import uvicorn
from app.main import app  # noqa: F401


def main():
    """Main entry point. Must be named main() and guarded by __name__ == '__main__'."""
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":
    main()