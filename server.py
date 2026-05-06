#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
try:
    from whisperx_service import WhisperXSTTService
except ImportError:
    WhisperXSTTService = None

# Load environment variables
load_dotenv(override=True)

from bot_fast_api import run_bot

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles FastAPI startup and shutdown."""
    yield  # Run app


# Initialize FastAPI app with lifespan manager
app = FastAPI(lifespan=lifespan)

# Configure CORS to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/assets", StaticFiles(directory="ui/assets"), name="assets")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted")
    try:
        transportType = websocket.query_params.get("transport", None)
        await run_bot(websocket, transportType)
    except Exception as e:
        print(f"Exception in run_bot: {e}")


@app.post("/connect")
async def bot_connect(request: Request) -> Dict[Any, Any]:
    ws_url = os.getenv("PUBLIC_WS_URL", "wss://rtstt-demo.securityzone.vn/ws")
    return {"ws_url": ws_url}

@app.get("/")
async def serve_index():
    return FileResponse("ui/index.html")

@app.get("/health")
async def serve_index():
    return {"status": "ok"}
    
async def main():
    tasks = []
    try:
        config = uvicorn.Config(app, host="0.0.0.0", port=int(os.getenv("PORT", "9090")))
        server = uvicorn.Server(config)
        tasks.append(server.serve())

        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        print("Tasks cancelled (probably due to shutdown).")


if __name__ == "__main__":
    asyncio.run(main())
