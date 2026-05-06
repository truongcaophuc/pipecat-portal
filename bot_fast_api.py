#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import os
import sys
import aiohttp
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.services.openai.llm import OpenAILLMService

# Optional imports - may not be available in light Docker image
try:
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
except ImportError:
    GeminiLiveLLMService = None

try:
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService, ElevenLabsHttpTTSService
    from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService, ElevenLabsSTTService
except ImportError:
    ElevenLabsTTSService = ElevenLabsHttpTTSService = None
    ElevenLabsRealtimeSTTService = ElevenLabsSTTService = None

try:
    from whisperx_service import WhisperXSTTService, WhisperHallucinationFilter
except ImportError:
    WhisperXSTTService = WhisperHallucinationFilter = None

try:
    from whisperx_api_client import WhisperXAPISTTService
except ImportError:
    WhisperXAPISTTService = None

from custom_http_tts_service import CustomHttpTTSService
from omi_tts_service import OmniVoiceTTSService
from n8n_processor import N8NProcessor
from n8n_processor_llm import N8NLLMService, ResponseMode
from qwen_api_client import QwenChatSTTService
from pipecat.processors.aggregators.llm_context import LLMContext as OpenAILLMContext
from pipecat.serializers.twilio import TwilioFrameSerializer

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


SYSTEM_INSTRUCTION = f"""
"You are  Chatbot, a friendly, helpful robot.
Respond to what the user said in a creative and helpful way. Keep your responses brief. One or two sentences at most.
Trả lời ngắn gọn, không dài dòng! Hãy trả lời bằng tiếng việt
"""
'''
# Global model holder
WHISPERX_MODEL = None

def get_whisperx_model():
    global WHISPERX_MODEL
    if WHISPERX_MODEL is None:
        logger.info("Loading WhisperX model globally...")
        import whisperx
        WHISPERX_MODEL = whisperx.load_model(
            "large-v3",
            "cuda",
            compute_type="float16",
            language="vi"
        )
    return WHISPERX_MODEL
'''
async def run_bot(websocket_client, transport_type: Optional[str] = 'websocket'):
    session = aiohttp.ClientSession()
    try:
        ws_transport = None
        init_welcome = False
        
        # VAD config — in pipecat v1.0.0, VAD is a pipeline processor, not a transport param
        vad_params = VADParams(
            confidence=float(os.getenv("VAD_CONFIDENCE", "0.5")),
            start_secs=float(os.getenv("VAD_START_SECS", "0.2")),
            stop_secs=float(os.getenv("VAD_STOP_SECS", "0.8")),
            min_volume=float(os.getenv("VAD_MIN_VOLUME", "0.85")),
        )
        vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(params=vad_params))

        if transport_type == 'twilio':
            init_welcome = True
            ws_transport = FastAPIWebsocketTransport(
                websocket=websocket_client,
                params=FastAPIWebsocketParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    add_wav_header=False,
                    serializer=TwilioFrameSerializer(
                        stream_sid="session_id",
                        call_sid="call_id",
                        account_sid="account_sid",
                        auth_token="auth_token"
                    ),
                ),
            )
        else:
            init_welcome = True
            ws_transport = FastAPIWebsocketTransport(
                websocket=websocket_client,
                params=FastAPIWebsocketParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    add_wav_header=False,
                    serializer=ProtobufFrameSerializer(),
                ),
            )
        print("Start bot", transport_type, init_welcome)
        # Initialize text-to-speech service
        # stt = ElevenLabsSTTService(
        #     api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        #     aiohttp_session=session,
        #     language_code="vi",
        #     model="scribe_v2",
        # )

        # Get shared model instance
        #model_instance = get_whisperx_model()
        
        # Create NEW service instance with SHARED model
        '''
        stt = WhisperXSTTService(
            device="cuda", 
            model="large-v3",
            model_obj=model_instance
        )
        '''
        # WhisperX STT (fallback, only if available)
        whisperx_stt = None
        if WhisperXAPISTTService:
            whisperx_stt = WhisperXAPISTTService(
                api_url=os.getenv("STT_BASE_URL", ""),
                aiohttp_session=session,
            )

        # Qwen Chat STT
        qwen_stt = QwenChatSTTService(
            api_url=os.getenv("QWEN_STT_URL", ""),
            aiohttp_session=session,
            sample_rate=16000,
            fallback_stt=whisperx_stt,
        )

        # Try Qwen, fallback to WhisperX if Qwen fails to initialize
        try:
            stt = qwen_stt
            logger.info("Using QwenChatSTTService")
        except Exception as e:
            logger.warning(f"Qwen STT init failed: {e}, falling back to WhisperX")
            stt = whisperx_stt
        

        # TTS cũ (ElevenLabs) - đã comment
        # tts = ElevenLabsHttpTTSService(
        #     api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        #     voice_id=os.getenv("ELEVENLABS_VOICE_ID", "SAz9YHcvj6GT2YYXdXww"),
        #     aiohttp_session=session,
        # )

        # TTS OmniVoice (OpenAI-compatible /v1/audio/speech)
        tts = OmniVoiceTTSService(
            aiohttp_session=session,
            base_url=os.getenv("OMNIVOICE_TTS_BASE_URL", "http://10.120.80.3:6655"),
            voice_id=os.getenv("OMNIVOICE_TTS_VOICE_ID", "nu_ai"),
            sample_rate=24000,
        )

        # Initialize LLM service
        
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY", ""))
        #llm = N8NProcessor(n8n_webhook_url="https://n8n-test.securityzone.vn/webhook/samsung-support-flow")
        llm = N8NLLMService(
            api_key=os.getenv("OPENAI_API_KEY", ""), 
            n8n_webhook_url=os.getenv("LLM_BASE_URL", ""),
            ragflow_url=os.getenv("RAG_LLM_BASE_URL", ""),
            ragflow_api_key=os.getenv("RAG_API_KEY", ""),
            response_mode=ResponseMode.N8N,
            aiohttp_session=session,
        )
        '''
        context = LLMContext(
            [
                {
                    "role": "user",
                    "content": "Xin chào em tôi là bot",
                }
            ],
        )
        '''
        # Create context with initial messages
        messages = []
        if init_welcome:
            messages.append({"role": "system", "welcome": True, "content": os.getenv("WELCOME_MSG", "")})

        # Create context (messages only)
        context = OpenAILLMContext(messages)

        context_aggregator = LLMContextAggregatorPair(context)

        # RTVI events for Pipecat client UI
        rtvi = RTVIProcessor()

        pipeline = Pipeline(
            [
                ws_transport.input(),
                vad,
                rtvi,
                stt,
                context_aggregator.user(),
                llm,  # LLM
                tts,
                ws_transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)],
        )

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            logger.info("Pipecat client ready.")
            await rtvi.set_bot_ready()
            # Kick off the conversation.
            await task.queue_frames([LLMRunFrame()])

        @ws_transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Pipecat Client connected")

        @ws_transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Pipecat Client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=False)

        await runner.run(task)
    finally:
        # Cleanup: close aiohttp session
        await session.close()
        logger.info("Cleaned up aiohttp session")
