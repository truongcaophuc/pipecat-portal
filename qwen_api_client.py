"""
QwenARS API STT Service - Pipecat client for remote QwenARS API.

Drop-in replacement for QwenARSSTTService that calls remote API instead of local model.
"""

import asyncio
import base64
import json
import os
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import SegmentedSTTService, STTService, WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    raise Exception(f"Missing module: {e}")

class WhisperHallucinationFilter():

    def __init__(self, blacklist_phrases: list[str]):
        self.blacklist = [p.lower() for p in blacklist_phrases]
    
    async def process_frame(self, text):
        if not text:
            return True

        text_lower = text.lower()
        # Kiểm tra nếu chứa phrase không mong muốn
        if any(phrase in text_lower for phrase in self.blacklist):
            print("ignore text frame", text_lower)
            return True

class QwenAPISTTService(SegmentedSTTService):
    """
    Simple QwenARS API STT service using /api/transcribe endpoint.

    Args:
        api_url: Base URL of QwenARS API server (e.g., "http://localhost:8000")
        api_key: Optional API key for authentication
        timeout: HTTP request timeout in seconds
        aiohttp_session: Optional shared aiohttp session
        **kwargs: Additional arguments passed to SegmentedSTTService.
    """

    def __init__(
        self,
        *,
        api_url: str = "http://localhost:8801",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
        **kwargs,
    ):
        super().__init__(
            
            ttfs_p99_latency=1.0,
            **kwargs
        )
        self._api_url = api_url.rstrip('/')
        self._api_key = api_key
        self._timeout = timeout
        self._aiohttp_session = aiohttp_session
        self._owned_session = False
        self._model_name = "qwen-stt"

        self.hallucination_filter = WhisperHallucinationFilter(
            blacklist_phrases=[
            ]
        )

    def can_generate_metrics(self) -> bool:
        return True

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._aiohttp_session is None:
            self._aiohttp_session = aiohttp.ClientSession()
            self._owned_session = True
        return self._aiohttp_session

    async def _close_session(self):
        if self._owned_session and self._aiohttp_session:
            await self._aiohttp_session.close()
            self._aiohttp_session = None
            self._owned_session = False

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:  # type: ignore[override]
        """
        Transcribe audio using /api/transcribe endpoint.

        Args:
            audio: Raw audio bytes (float32 format).

        Yields:
            Frame: Either a TranscriptionFrame or ErrorFrame.
        """
        import time
        import numpy as np

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        t0 = time.perf_counter()
        audio_kb = len(audio) / 1024
        logger.info(f"QwenSTT: sending {audio_kb:.1f}KB to /api/transcribe ...")

        try:
            session = await self._get_session()

            url = f"{self._api_url}/api/transcribe"
            headers = {
                "Content-Type": "application/octet-stream",
            }

            if self._api_key:
                headers["X-API-Key"] = self._api_key

            # asyncio.shield prevents pipeline cancellation from aborting the
            # in-flight HTTP request (race condition: smart turn fires while
            # SegmentedSTTService is still waiting for /api/transcribe response)
            post_coro = session.post(
                url,
                data=audio,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            )
            async with await asyncio.shield(post_coro) as response:
                ttfb = (time.perf_counter() - t0) * 1000
                await self.stop_ttfb_metrics()

                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"QwenARS Simple API error: {response.status} - {error_text}")
                    yield ErrorFrame(error=f"QwenARS API error: {response.status}")
                    return

                result = await response.json()
                total = (time.perf_counter() - t0) * 1000
                text = result.get("text", "").strip()

                await self.stop_processing_metrics()

                if text:
                    await self._handle_transcription(text, True, None)
                    logger.info(f"QwenSTT [{text}] | ttfb={ttfb:.0f}ms total={total:.0f}ms audio={len(audio)/1024:.1f}KB")
                    ishaclu = await self.hallucination_filter.process_frame(text)
                    if not ishaclu:
                        yield TranscriptionFrame(
                            text,
                            self._user_id,
                            time_now_iso8601(),
                            None,
                        )

        except asyncio.CancelledError:
            logger.warning("QwenSTT: request was cancelled (pipeline turn-switch), result discarded")

        except aiohttp.ClientError as e:
            logger.error(f"QwenARS Simple API connection error: {e}")
            yield ErrorFrame(error=f"QwenARS API connection error: {e}")

        except Exception as e:
            import traceback
            logger.error(f"QwenARS Simple API exception: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            yield ErrorFrame(error=f"QwenARS API error: {e}")




class QwenStreamingSTTService(STTService):
    """Streaming Qwen STT service using HTTP session API.

    Giống ElevenLabsRealtimeSTTService: gửi audio liên tục qua run_stt() cho mọi
    frame, không phân biệt đang nói hay không → không bao giờ mất từ đầu tiên.

    Protocol:
        POST /api/start                          → {"session_id": "..."}
        POST /api/chunk?session_id=<id>
             Content-Type: application/octet-stream
             Body: float32 LE PCM @ 16 kHz       → {"language": "vi", "text": "interim..."}
        POST /api/finish?session_id=<id>          → {"language": "vi", "text": "final..."}
    """

    def __init__(
        self,
        *,
        api_url: str = "http://localhost:8801",
        api_key: Optional[str] = None,
        chunk_ms: int = 500,
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
        **kwargs,
    ):
        super().__init__(
            ttfs_p99_latency=0.5,
            **kwargs,
        )
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._chunk_ms = chunk_ms
        self._aiohttp_session = aiohttp_session
        self._owned_session = False

        self._session_id: Optional[str] = None
        self._speaking = False
        self._audio_buf = bytearray()
        self._chunk_bytes = 0

    def can_generate_metrics(self) -> bool:
        return True

    # ─── lifecycle ────────────────────────────────────────────────────

    async def start(self, frame):
        await super().start(frame)
        self._chunk_bytes = int(self._chunk_ms * self.sample_rate / 1000) * 2
        if self._aiohttp_session is None:
            self._aiohttp_session = aiohttp.ClientSession()
            self._owned_session = True
        # Tạo session ngay — audio sẽ được gửi liên tục từ frame đầu tiên
        await self._new_session()

    async def stop(self, frame):
        await super().stop(frame)
        await self._cleanup_session()
        if self._owned_session and self._aiohttp_session:
            await self._aiohttp_session.close()
            self._aiohttp_session = None

    async def cancel(self, frame):
        await super().cancel(frame)
        await self._cleanup_session()

    # ─── frame processing ─────────────────────────────────────────────

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._speaking = True
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._on_user_stopped()

    # ─── run_stt: gọi cho MỌI audio frame (giống ElevenLabs) ─────────

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Buffer audio và gửi chunk mỗi chunk_ms lên /api/chunk."""
        if not self._session_id:
            logger.warning("QwenStreaming: run_stt called but no session_id — audio dropped")
            yield None
            return

        self._audio_buf.extend(audio)

        while len(self._audio_buf) >= self._chunk_bytes:
            chunk = bytes(self._audio_buf[: self._chunk_bytes])
            self._audio_buf = self._audio_buf[self._chunk_bytes :]
            interim = await self._send_chunk(chunk)
            if interim and self._speaking:
                await self.stop_ttfb_metrics()
                await self.push_frame(
                    InterimTranscriptionFrame(interim, self._user_id, time_now_iso8601(), "vi")
                )

        yield None

    # ─── session management ───────────────────────────────────────────

    async def _new_session(self):
        try:
            async with self._aiohttp_session.post(
                f"{self._api_url}/api/start", headers=self._headers()
            ) as r:
                data = await r.json()
                self._session_id = data["session_id"]
                self._audio_buf.clear()
                logger.debug("QwenStreaming: new session {}", self._session_id)
        except Exception as e:
            logger.error("QwenStreaming: /api/start FAILED — bot sẽ không nhận audio: {}", e)
            self._session_id = None

    async def _send_chunk(self, int16_bytes: bytes) -> Optional[str]:
        try:
            import numpy as np
            float32 = np.frombuffer(int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            async with self._aiohttp_session.post(
                f"{self._api_url}/api/chunk",
                params={"session_id": self._session_id},
                data=float32.tobytes(),
                headers={**self._headers(), "Content-Type": "application/octet-stream"},
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    return (data.get("text") or "").strip() or None
        except Exception as e:
            logger.warning("QwenStreaming: /api/chunk error: {}", e)
        return None

    async def _on_user_stopped(self):
        self._speaking = False
        if not self._session_id:
            return
        sid = self._session_id
        self._session_id = None
        try:
            async with self._aiohttp_session.post(
                f"{self._api_url}/api/finish",
                params={"session_id": sid},
                headers=self._headers(),
            ) as r:
                data = await r.json()
                text = (data.get("text") or "").strip()
                lang = data.get("language") or "vi"
                await self.stop_processing_metrics()
                if text:
                    await self._handle_transcription(text, True, lang)
                    logger.debug("QwenStreaming final: [{}]", text)
                    await self.push_frame(
                        TranscriptionFrame(
                            text, self._user_id, time_now_iso8601(), lang, finalized=True
                        )
                    )
        except Exception as e:
            logger.error("QwenStreaming: /api/finish error: {}", e)
        finally:
            # Tạo session mới sẵn cho turn tiếp theo
            await self._new_session()

    async def _cleanup_session(self):
        if self._session_id:
            try:
                await self._aiohttp_session.post(
                    f"{self._api_url}/api/finish",
                    params={"session_id": self._session_id},
                    headers=self._headers(),
                )
            except Exception:
                pass
            self._session_id = None

    def _headers(self) -> dict:
        return {"X-API-Key": self._api_key} if self._api_key else {}

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        pass


class QwenChatSTTService(SegmentedSTTService):
    """
    STT via OpenAI-compatible chat completions with audio_url (Qwen3-ASR style).

    Sends audio as base64 WAV data URL to /v1/chat/completions:
        {"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}}]}
    """

    def __init__(
        self,
        *,
        api_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
        fallback_stt=None,
        **kwargs,
    ):
        super().__init__(
            ttfs_p99_latency=1.0,
            **kwargs,
        )
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._aiohttp_session = aiohttp_session
        self._owned_session = False
        self._fallback_stt = fallback_stt
        self.hallucination_filter = WhisperHallucinationFilter(blacklist_phrases=[])

    def can_generate_metrics(self) -> bool:
        return True

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._aiohttp_session is None:
            self._aiohttp_session = aiohttp.ClientSession()
            self._owned_session = True
        return self._aiohttp_session

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        import time
        import wave
        import io

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        t0 = time.perf_counter()
        logger.info(f"QwenChatSTT: sending {len(audio)/1024:.1f}KB ...")

        # Wrap raw PCM int16 bytes in WAV container for data URL
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(self.sample_rate or 16000)
            wf.writeframes(audio)
        audio_b64 = base64.b64encode(wav_buf.getvalue()).decode()
        data_url = f"data:audio/wav;base64,{audio_b64}"

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "language: vietnamese"},
                        {"type": "audio_url", "audio_url": {"url": data_url}},
                    ],
                }
            ],
        }

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            session = await self._get_session()

            async with session.post(
                f"{self._api_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as response:
                ttfb = (time.perf_counter() - t0) * 1000
                await self.stop_ttfb_metrics()

                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"QwenChatSTT error: {response.status} - {error_text}")
                    yield ErrorFrame(error=f"QwenChatSTT error: {response.status}")
                    return

                result = await response.json()
                total = (time.perf_counter() - t0) * 1000
                raw_content = (result["choices"][0]["message"]["content"] or "").strip()

                # Parse language and <asr_text>
                import re as _re
                lang_m = _re.search(r"language\s+(\w+)", raw_content)
                detected_lang = lang_m.group(1).lower() if lang_m else "vietnamese"

                m = _re.search(r"<asr_text>(.*?)(?:</asr_text>|$)", raw_content, _re.DOTALL)
                text = m.group(1).strip() if m else raw_content

                await self.stop_processing_metrics()
                logger.info(f"QwenChatSTT [{text}] lang={detected_lang} | ttfb={ttfb:.0f}ms total={total:.0f}ms")

                if detected_lang != "vietnamese" and self._fallback_stt:
                    logger.warning(f"QwenChatSTT: non-Vietnamese ({detected_lang}), falling back to WhisperX")
                    async for frame in self._fallback_stt.run_stt(audio):
                        yield frame
                    return

                if text:
                    await self._handle_transcription(text, True, None)
                    if not await self.hallucination_filter.process_frame(text):
                        yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), None)

        except aiohttp.ClientError as e:
            logger.error(f"QwenChatSTT connection error: {e}")
            yield ErrorFrame(error=f"QwenChatSTT connection error: {e}")
        except Exception as e:
            import traceback
            logger.error(f"QwenChatSTT exception: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            yield ErrorFrame(error=f"QwenChatSTT error: {e}")
