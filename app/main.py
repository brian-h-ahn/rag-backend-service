import time
import uuid
import logging
from fastapi import FastAPI, Request
from app.api.routes import router
from app.core.config import settings

logger = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title=settings.app_name)
app.include_router(router)

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    try:
        response = await call_next(request)
        latency_ms = int((time.perf_counter() - start) * 1000)

        logger.info(
            "request_id=%s method=%s path=%s status=%s latency_ms=%s",
            request_id, request.method, request.url.path, response.status_code, latency_ms,
        )

        # Helpful for debugging across client <-> server
        response.headers["X-Request-ID"] = request_id
        return response

    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.exception(
            "request_id=%s method=%s path=%s latency_ms=%s error=%s",
            request_id, request.method, request.url.path, latency_ms, str(e)
        )
        raise