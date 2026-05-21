from __future__ import annotations

from fastapi import APIRouter

from .admin import router as admin_router
from .chat import router as chat_router
from .gmail import router as gmail_router
from .meta import router as meta_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(meta_router)
api_router.include_router(chat_router)
api_router.include_router(gmail_router)
api_router.include_router(admin_router)

__all__ = ["api_router"]
