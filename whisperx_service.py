import asyncio
import io
import wave
from typing import AsyncGenerator, Optional

import numpy as np
import whisperx
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame, InterimTranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.processors.frame_processor import FrameProcessor

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



class WhisperXSTTService(SegmentedSTTService):
    """WhisperX-based Speech-to-Text service for Pipecat.
    
    This service uses WhisperX for high-quality speech transcription with support
    for multiple languages and speaker diarization capabilities.
    
    Args:
        device: Device to run inference on ('cuda', 'cpu', or 'auto'). Defaults to 'cuda'.
        batch_size: Batch size for WhisperX transcription. Defaults to 16.
        compute_type: Compute type for inference ('float16', 'int8', etc.). Defaults to 'float16'.
        model: WhisperX model size ('tiny', 'base', 'small', 'medium', 'large'). Defaults to 'base'.
        language: Language code for transcription (e.g., 'en', 'vi'). Defaults to 'en'.
        **kwargs: Additional arguments passed to SegmentedSTTService.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        batch_size: int = 16,
        compute_type: str = "float16",
        model: str = "base",
        language: str = "vi",
        model_obj: Optional[object] = None,  # Accept pre-loaded model
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._device = device
        self._batch_size = batch_size
        self._compute_type = compute_type
        self._model_name = model
        self._language = language
        
        self._settings = {
            "language": language,
            "device": self._device,
            "compute_type": self._compute_type,
            "batch_size": self._batch_size,
        }

        if model_obj:
            self._model = model_obj
            logger.info(f"Using provided WhisperX model instance")
        else:
            self._model = None
            self._load_model()

    def can_generate_metrics(self) -> bool:
        """Indicates whether this service can generate metrics.
        
        Returns:
            bool: True, as this service supports metric generation.
        """
        return True

    def _load_model(self):
        """Load the WhisperX model.
        
        Note:
            If this is the first time this model is being run,
            it will take time to download from the Hugging Face model hub.
        """
        logger.info(f"Loading WhisperX model: {self._model_name} on {self._device}")
        try:
            self._model = whisperx.load_model(
                self._model_name,
                self._device,
                compute_type=self._compute_type,
                language=self._language,
            )
            logger.info("WhisperX model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load WhisperX model: {e}")
            raise

    async def set_language(self, language: Language):
        """Set the language for transcription.
        
        Args:
            language: The Language enum value to use for transcription.
        """
        # Convert Language enum to language code if needed
        lang_code = str(language.value) if hasattr(language, 'value') else str(language)
        logger.info(f"Switching STT language to: [{lang_code}]")
        self._settings["language"] = lang_code
        self._language = lang_code

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe audio data using WhisperX.
        
        Args:
            audio: Raw audio bytes in WAV format (provided by SegmentedSTTService).
            
        Yields:
            Frame: Either a TranscriptionFrame containing the transcribed text
                  or an ErrorFrame if transcription fails.
                  
        Note:
            The parent class (SegmentedSTTService) provides audio in WAV format,
            so we need to extract the raw PCM data and convert it to float32.
        """
        if not self._model:
            logger.error(f"{self} error: WhisperX model not available")
            yield ErrorFrame("WhisperX model not available")
            return

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        try:
            # Extract audio data from WAV bytes
            # The parent class provides audio in WAV format
            with io.BytesIO(audio) as wav_io:
                with wave.open(wav_io, "rb") as wav_file:
                    audio_data = wav_file.readframes(wav_file.getnframes())
                    
            # Convert to float32 numpy array (divide by 32768 for 16-bit audio)
            audio_float = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Run WhisperX transcription in thread pool to avoid blocking
            result = await asyncio.to_thread(
                self._model.transcribe,
                audio_float,
                batch_size=self._batch_size,
            )

            # Extract text from segments
            text = ""
            for segment in result.get("segments", []):
                text += segment.get("text", "") + " "

            text = text.strip()

            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()

            if text:
                await self._handle_transcription(text, True, self._settings["language"])
                logger.debug(f"Transcription: [{text}]")
                yield TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    self._settings["language"],
                )

        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"{self} error: {e}")
