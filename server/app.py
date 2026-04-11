"""
server/app.py — OpenEnv required entry point.
Imports and re-exports the FastAPI app for multi-mode deployment.
"""
from app.main import app, main  # noqa: F401

__all__ = ["app", "main"]