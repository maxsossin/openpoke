from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .logging_config import configure_logging, logger
from .routes import api_router
from .services import get_important_email_watcher, get_knowledge_graph_watcher, get_trigger_scheduler
from .services.triggers import get_trigger_service


# Register global exception handlers for consistent error responses across the API
def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.debug("validation error", extra={"errors": exc.errors(), "path": str(request.url)})
        return JSONResponse(
            {"ok": False, "error": "Invalid request", "detail": exc.errors()},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException):
        logger.debug(
            "http error",
            extra={"detail": exc.detail, "status": exc.status_code, "path": str(request.url)},
        )
        detail = exc.detail
        if not isinstance(detail, str):
            detail = json.dumps(detail)
        return JSONResponse({"ok": False, "error": detail}, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error", extra={"path": str(request.url)})
        return JSONResponse(
            {"ok": False, "error": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


configure_logging()
_settings = get_settings()

app = FastAPI(
    title=_settings.app_name,
    version=_settings.app_version,
    docs_url=_settings.resolved_docs_url,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(api_router)


_CLAIM_VERIFIER_AGENT = "claim-verifier"

# Every 6 hours. Balances freshness against LLM cost: at ~10 LLM calls per run
# (one execution agent turn evaluating 10 claims), 4 runs/day = ~40 assessments/day.
_CLAIM_VERIFIER_RRULE = "FREQ=HOURLY;INTERVAL=6"

_CLAIM_VERIFIER_PAYLOAD = (
    "Run the periodic claim verification pass.\n\n"
    "Evaluate pending newsletter claims against available evidence in the knowledge "
    "graph. Process up to 10 claims per run. For each claim, gather evidence and "
    "record your verdict following your system prompt exactly.\n\n"
    "Use fetch_pending_claims first, then query_claim_evidence for each claim "
    "(all simultaneously), then record_claim_verdict for each claim "
    "(all simultaneously). Write a structured summary as your final message."
)


def _bootstrap_claim_verifier_trigger() -> None:
    """Ensure the claim-verifier trigger exists and is active.

    Idempotent: if an active trigger already exists for this agent, this is a
    no-op. Called synchronously from the startup event before the scheduler
    starts so the trigger is visible on the first poll.
    """
    service = get_trigger_service()
    existing = service.list_triggers(agent_name=_CLAIM_VERIFIER_AGENT)
    non_completed = [t for t in existing if t.status != "completed"]
    if non_completed:
        logger.debug(
            "Claim verifier trigger already exists",
            extra={"trigger_id": non_completed[0].id, "status": non_completed[0].status},
        )
        return

    record = service.create_trigger(
        agent_name=_CLAIM_VERIFIER_AGENT,
        payload=_CLAIM_VERIFIER_PAYLOAD,
        recurrence_rule=_CLAIM_VERIFIER_RRULE,
        status="active",
    )
    logger.info(
        "Claim verifier trigger bootstrapped",
        extra={"trigger_id": record.id, "next_trigger": record.next_trigger},
    )


@app.on_event("startup")
# Initialize background services (trigger scheduler, email watcher, KG watcher) when the app starts
async def _start_trigger_scheduler() -> None:
    _bootstrap_claim_verifier_trigger()
    scheduler = get_trigger_scheduler()
    await scheduler.start()
    watcher = get_important_email_watcher()
    await watcher.start()
    kg_watcher = get_knowledge_graph_watcher()
    await kg_watcher.start()


@app.on_event("shutdown")
# Gracefully shutdown background services when the app stops
async def _stop_trigger_scheduler() -> None:
    scheduler = get_trigger_scheduler()
    await scheduler.stop()
    watcher = get_important_email_watcher()
    await watcher.stop()
    kg_watcher = get_knowledge_graph_watcher()
    await kg_watcher.stop()


__all__ = ["app"]
