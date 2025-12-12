# main.py
"""
NGAI DeepThink Backend (Hybrid: Polling + WebSocket)
- Single-file FastAPI app
- Step-by-step thinking (single AI call) + streamed final answer (local chunk streaming)
- OpenRouter integration (single non-streaming call for steps + final answer)
- TTLCache for memory with auto-cleanup
- API key validation via env, rate limiting, Pydantic validation
- Shared httpx.AsyncClient, CORS, request-id, process time, logging, heartbeat, cleanup
"""

import os
import re
import uuid
import json
import time
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx
from fastapi import (
    FastAPI,
    Request,
    HTTPException,
    Depends,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from cachetools import TTLCache
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

# ---------------------------
# Logging / Config
# ---------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("ngai")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
API_KEYS = os.getenv("API_KEYS", "")  # comma-separated client keys
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is required in environment")
if not API_KEYS:
    raise RuntimeError("API_KEYS is required in environment")
API_KEYS = [k.strip() for k in API_KEYS.split(",") if k.strip()]

# Free models you want to allow (update as available)
FREE_MODELS = ["openassistant-medium", "openassistant-large", "openassistant-mini"]
DEFAULT_MODEL = FREE_MODELS[0]

# OpenRouter endpoint
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ---------------------------
# App + Rate limiter + CORS
# ---------------------------
app = FastAPI(title="NGAI DeepThink Backend (Hybrid WebSocket + Polling)")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# CORS - lock down origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Middleware: Request ID & process time
# ---------------------------
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.time()
    resp = await call_next(request)
    resp.headers["X-Process-Time"] = str(time.time() - start)
    return resp


# ---------------------------
# Shared HTTP client
# ---------------------------
@app.on_event("startup")
async def startup_event():
    app.state.http_client = httpx.AsyncClient(timeout=40.0)
    logger.info("HTTP client started")


@app.on_event("shutdown")
async def shutdown_event():
    await app.state.http_client.aclose()
    logger.info("HTTP client closed")


# ---------------------------
# In-memory caches (TTL)
# ---------------------------
THINKING_CACHE_TTL = 60 * 60  # 1 hour
thinking_cache: TTLCache = TTLCache(maxsize=2000, ttl=THINKING_CACHE_TTL)
user_history: TTLCache = TTLCache(maxsize=10000, ttl=THINKING_CACHE_TTL)


# ---------------------------
# Pydantic models
# ---------------------------
class ChatStartRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, max_length=5000)
    model: Optional[str] = Field(default=DEFAULT_MODEL)


class ChatStartResponse(BaseModel):
    task_id: str


class ChatStatusResponse(BaseModel):
    status: str  # thinking|analyzing|reasoning|complete
    step: int
    message: str
    progress: int
    timestamp: str


class ChatResultResponse(BaseModel):
    thinking_process: List[ChatStatusResponse]
    final_answer: str
    confidence: float = 0.95
    citations: List[str] = []
    suggested_followups: List[str] = []


class RegenerateRequest(BaseModel):
    user_id: str
    last_task_id: str


# ---------------------------
# Security helpers
# ---------------------------
async def validate_api_key(request: Request):
    key = request.headers.get("x-api-key")
    if not key or key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")


def validate_model(model: str) -> str:
    if model not in FREE_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model. Choose from: {FREE_MODELS}")
    return model


def filter_toxic_content(message: str) -> str:
    banned = ["badword1", "badword2"]
    lower = (message or "").lower()
    for b in banned:
        if b in lower:
            raise HTTPException(status_code=400, detail="Message contains inappropriate content")
    return message


# ---------------------------
# Thinking step utilities
# ---------------------------
THINKING_STATES = [
    "receiving",
    "analyzing",
    "researching",
    "reasoning",
    "formulating",
    "reviewing",
    "complete",
]


def create_thinking_step(step_idx: int, message: str) -> Dict:
    progress = int(((step_idx + 1) / len(THINKING_STATES)) * 100)
    return {
        "status": THINKING_STATES[min(step_idx, len(THINKING_STATES) - 1)],
        "step": step_idx + 1,
        "message": message,
        "progress": progress,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------
# AI integration helpers
# ---------------------------
RE_STEP = re.compile(r"STEP\s*\d+[:\-]\s*(.+)", re.IGNORECASE)
RE_FINAL = re.compile(r"FINAL[:\-]\s*(.+)", re.IGNORECASE)


def parse_thinking_steps(content: str) -> Dict:
    steps = RE_STEP.findall(content)
    final_match = RE_FINAL.search(content)
    final = final_match.group(1).strip() if final_match else ""
    if not steps:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
        if paragraphs:
            steps = paragraphs[:-1] if len(paragraphs) > 1 else paragraphs
        else:
            steps = [content.strip()]
    if not final:
        final = content.strip().splitlines()[-1] if content.strip() else "Answer generated by AI."
    return {"steps": steps, "final_answer": final}


async def get_ai_thinking_steps(message: str, model: str) -> Dict:
    prompt = f"""
Think step by step and output labelled steps in this format:

STEP 1: [initial analysis]
STEP 2: [key concepts]
STEP 3: [logical reasoning]
STEP 4: [synthesis]
FINAL: [concise final answer]

User Question: {message}
"""
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    client: httpx.AsyncClient = app.state.http_client
    try:
        resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)
        if resp.status_code == 429:
            raise HTTPException(status_code=429, detail="AI service rate-limited")
        resp.raise_for_status()
    except httpx.TimeoutException:
        logger.exception("AI service timeout (steps)")
        raise HTTPException(status_code=504, detail="AI service timeout")
    except httpx.HTTPError as e:
        logger.exception("AI service error (steps)")
        raise HTTPException(status_code=502, detail=f"AI service error: {str(e)}")

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        content = json.dumps(data)
    return parse_thinking_steps(content)


# ---------------------------
# Thinking simulation combining AI steps + local streaming final answer
# ---------------------------
async def simulate_thinking_full(task_id: str, user_message: str, model: str, websocket: Optional[WebSocket] = None):
    """
    Single AI call approach:
    1) Call the non-streaming endpoint to get structured thinking steps AND a final_answer.
    2) Emit those thinking steps (with delays).
    3) Stream the final_answer character-by-character (no second API call).
    """
    try:
        thinking_data = await get_ai_thinking_steps(user_message, model)
    except HTTPException as e:
        if websocket:
            await websocket.send_json({"type": "error", "detail": str(e.detail)})
        thinking_cache[task_id]["complete"] = True
        thinking_cache[task_id]["final_answer"] = f"Error generating thinking steps: {e.detail}"
        return

    steps = thinking_data.get("steps", [])
    final_answer = thinking_data.get("final_answer", "")

    # 1) Emit the thinking steps
    for idx, step_text in enumerate(steps):
        step_obj = create_thinking_step(idx, step_text)
        thinking_cache[task_id]["thinking_steps"].append(step_obj)
        if websocket:
            await websocket.send_json({"type": "thinking_step", "task_id": task_id, "step": step_obj})
        await asyncio.sleep(0.6 + (idx * 0.15))

    # small review pause
    review_step = create_thinking_step(len(THINKING_STATES) - 2, "Reviewing the response for quality...")
    thinking_cache[task_id]["thinking_steps"].append(review_step)
    if websocket:
        await websocket.send_json({"type": "thinking_step", "task_id": task_id, "step": review_step})
    await asyncio.sleep(0.6)

    # 2) Stream the final_answer char-by-char using the SAME response
    partial = ""
    chunk_size = 4  # tune for UI smoothness vs event frequency
    for i in range(0, len(final_answer), chunk_size):
        # detect websocket disconnect (best-effort)
        if websocket and websocket.client_state.name != "CONNECTED":
            thinking_cache[task_id]["cancelled"] = True
            thinking_cache[task_id]["cancelled_at"] = datetime.now(timezone.utc).isoformat()
            logger.info(f"WebSocket disconnected mid-stream for task {task_id}; marking cancelled.")
            return

        chunk = final_answer[i : i + chunk_size]
        partial += chunk
        thinking_cache[task_id]["partial_answer"] = partial
        if websocket:
            await websocket.send_json({"type": "answer_chunk", "task_id": task_id, "chunk": chunk})
        await asyncio.sleep(0.01)

    # 3) Finish up
    thinking_cache[task_id]["final_answer"] = final_answer or "No answer generated."
    thinking_cache[task_id]["complete"] = True
    thinking_cache[task_id]["completed_at"] = datetime.now(timezone.utc).isoformat()

    if websocket:
        await websocket.send_json(
            {"type": "complete", "task_id": task_id, "final_answer": thinking_cache[task_id]["final_answer"]}
        )


# ---------------------------
# HTTP Endpoints (polling-friendly)
# ---------------------------

@app.post("/chat/start", dependencies=[Depends(validate_api_key)])
@limiter.limit("10/minute")
async def chat_start(req: ChatStartRequest):
    model = req.model or DEFAULT_MODEL
    validate_model(model)
    message = filter_toxic_content(req.message)

    task_id = str(uuid.uuid4())
    thinking_cache[task_id] = {
        "thinking_steps": [],
        "final_answer": "",
        "partial_answer": "",
        "model": model,
        "complete": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # store user history
    hist = user_history.get(req.user_id, [])
    hist.append({"task_id": task_id, "message": message, "model": model, "ts": datetime.now(timezone.utc).isoformat()})
    user_history[req.user_id] = hist[-10:]

    # start background simulation
    asyncio.create_task(simulate_thinking_full(task_id, message, model))

    logger.info(f"Started thinking task {task_id} model={model}")
    return {"task_id": task_id}


@app.get("/chat/status/{task_id}", dependencies=[Depends(validate_api_key)])
@limiter.limit("20/minute")
async def chat_status(task_id: str):
    task = thinking_cache.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["complete"]:
        return {
            "status": "complete",
            "message": "Thinking complete",
            "step": len(THINKING_STATES),
            "progress": 100,
            "timestamp": task.get("completed_at", datetime.now(timezone.utc).isoformat()),
        }
    steps = task["thinking_steps"]
    if not steps:
        return {
            "status": "thinking",
            "step": 0,
            "message": "Queued",
            "progress": 0,
            "timestamp": task["created_at"],
        }
    last = steps[-1]
    return last


@app.get("/chat/result/{task_id}", response_model=ChatResultResponse, dependencies=[Depends(validate_api_key)])
@limiter.limit("20/minute")
async def chat_result(task_id: str):
    task = thinking_cache.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if not task["complete"]:
        raise HTTPException(status_code=202, detail="Thinking not complete yet")
    return {
        "thinking_process": task["thinking_steps"],
        "final_answer": task["final_answer"],
        "confidence": 0.95,
        "citations": [],
        "suggested_followups": [],
    }


@app.post("/chat/regenerate", dependencies=[Depends(validate_api_key)])
@limiter.limit("10/minute")
async def chat_regenerate(req: RegenerateRequest):
    last = thinking_cache.get(req.last_task_id)
    if not last:
        raise HTTPException(status_code=404, detail="Last task not found")
    model = last.get("model", DEFAULT_MODEL)
    message = None
    for _, h in user_history.items():
        for entry in h:
            if entry.get("task_id") == req.last_task_id:
                message = entry.get("message")
                break
        if message:
            break
    if not message:
        raise HTTPException(status_code=400, detail="Original message not found for regeneration")

    new_task_id = str(uuid.uuid4())
    thinking_cache[new_task_id] = {
        "thinking_steps": [],
        "final_answer": "",
        "partial_answer": "",
        "model": model,
        "complete": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    asyncio.create_task(simulate_thinking_full(new_task_id, message, model))
    return {"task_id": new_task_id}


@app.get("/models")
async def list_models():
    return {"models": FREE_MODELS}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_tasks": len(thinking_cache),
        "cache_ttl_seconds": THINKING_CACHE_TTL,
    }


# ---------------------------
# WebSocket endpoint (Option A: WebSocket creates task_id)
# ---------------------------
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    api_key = websocket.headers.get("x-api-key")
    if not api_key or api_key not in API_KEYS:
        await websocket.accept()
        await websocket.send_json({"type": "error", "detail": "Invalid API key"})
        await websocket.close()
        return

    await websocket.accept()

    # heartbeat task to keep ws alive and detect disconnects
    heartbeat_task = asyncio.create_task(_ws_heartbeat(websocket))
    task_id = None
    try:
        init = await websocket.receive_json()
    except Exception:
        await websocket.send_json({"type": "error", "detail": "Invalid init payload"})
        await websocket.close()
        heartbeat_task.cancel()
        return

    user_message = init.get("message")
    model = init.get("model", DEFAULT_MODEL)
    try:
        validate_model(model)
        filter_toxic_content(user_message)
    except HTTPException as e:
        await websocket.send_json({"type": "error", "detail": e.detail})
        await websocket.close()
        heartbeat_task.cancel()
        return

    task_id = str(uuid.uuid4())
    thinking_cache[task_id] = {
        "thinking_steps": [],
        "final_answer": "",
        "partial_answer": "",
        "model": model,
        "complete": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    await websocket.send_json({"type": "connected", "task_id": task_id})

    try:
        await simulate_thinking_full(task_id, user_message, model, websocket=websocket)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for task {task_id}")
        if task_id in thinking_cache and not thinking_cache[task_id].get("complete"):
            thinking_cache[task_id]["cancelled"] = True
            thinking_cache[task_id]["cancelled_at"] = datetime.now(timezone.utc).isoformat()
    except Exception:
        logger.exception("Error during simulate_thinking_full via websocket")
        try:
            await websocket.send_json({"type": "error", "detail": "Internal server error"})
        except Exception:
            pass
    finally:
        heartbeat_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass


async def _ws_heartbeat(ws: WebSocket, interval: float = 10.0):
    """Sends periodic pings to keep connection alive; cancels if sending fails."""
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                await ws.send_json({"type": "ping"})
            except Exception:
                break
    except asyncio.CancelledError:
        return


# ---------------------------
# Global exception handler
# ---------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception for request {getattr(request, 'url', '')}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
