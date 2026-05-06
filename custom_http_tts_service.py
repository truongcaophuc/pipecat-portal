import aiohttp
import base64
import json
import re
from typing import AsyncGenerator, Optional, List, Dict, Any, Callable
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService


# Text normalization rules for TTS
MAP_RULES: List[Dict[str, Any]] = [
    {"type": "simple", "from": "Sacombank", "to": "sa com bank"},
    {"type": "simple", "from": "CRV", "to": "xê rờ vê"},
    {"type": "simple", "from": "eKYC", "to": "y kêy quai xy"},
    {"type": "simple", "from": "ABC", "to": "a bê xê"},
    {"type": "simple", "from": "TV", "to": "ti vi"},
    {"type": "simple", "from": "SSD", "to": "ét ét đi"},
    {"type": "simple", "from": "TNHH ", "to": "trách nhiệm hữu hạn "},
    {"type": "simple", "from": "HDMI", "to": "hắt đi em ai,"},
    {"type": "simple", "from": "Members", "to": "mem bơ,"},
    {"type": "simple", "from": "VAT", "to": "vê a tê "},
    {"type": "regex", "pattern": r"[。、*#\"]+", "to": " "},
    {"type": "regex", "pattern": r"\s+", "to": " "},
    {
        "type": "custom",
        "pattern": r"(1800[\-–—‑]\d{3}[\-–—‑]\d{3})",
        "func": lambda match: " ".join(re.sub(r"[\-–—‑]", "", match.group(0)))
    },
    {
        "type": "regex",
        "pattern": r"(\d{1,2}/\d{1,2}/\d{4})",
        "to": r"\1, "
    },
    {"type": "regex", "pattern": r"(\d+)\s*[\-–—‑]\s*(\d+)", "to": r"\1 đến \2"},
    {
        "type": "custom",
        "pattern": r"(?:^|\s)(0\d{9,10})(?:\s|$)",
        "func": lambda match: " " + " ".join(match.group(0).strip()) + " "
    },
    {"type": "regex", "pattern": r"#", "to": " "},
    {"type": "regex", "pattern": r"[?!]", "to": " ."},
]





class CustomHttpTTSService(TTSService):
    """Custom HTTP-based TTS service for SecurityZone RTTTS API.

    Calls the TTS API at https://rttts-demo.securityzone.vn/tts with configurable
    seed and voice parameters.
    """

    def __init__(
        self,
        *,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = "https://rttts-demo.securityzone.vn",
        seed: int = 10000,
        voice: str = "Giọng nữ 1",
        sample_rate: int = 24000,
        nfe_step: int = 25,
        **kwargs,
    ):
        """Initialize the Custom HTTP TTS service.

        Args:
            aiohttp_session: aiohttp ClientSession for HTTP requests.
            base_url: Base URL for the TTS API.
            seed: Seed parameter for voice generation.
            voice: Voice name to use (e.g., "Giọng nữ 1", "Giọng nam 1").
            sample_rate: Audio sample rate (default: 24000).
            **kwargs: Additional arguments passed to the parent service.
        """
        super().__init__(
            sample_rate=sample_rate,
            **kwargs,
        )

        self._base_url = base_url
        self._seed = seed
        self._voice = voice
        self._session = aiohttp_session
        self._nfe_step = nfe_step

    def apply_text_rules(self, text: str, rules: List[Dict[str, Any]] = None) -> str:
        """Apply text normalization rules for TTS.

        Args:
            text: Input text to normalize
            rules: List of rules to apply (defaults to MAP_RULES)

        Returns:
            Normalized text
        """
        if rules is None:
            rules = MAP_RULES

        result = text

        for rule in rules:
            rule_type = rule.get("type")

            if rule_type == "simple":
                # Simple string replace
                result = result.replace(rule["from"], rule["to"])

            elif rule_type == "regex":
                # Regex replace
                pattern = rule["pattern"]
                replacement = rule["to"]
                result = re.sub(pattern, replacement, result)

            elif rule_type == "custom":
                # Custom function
                pattern = rule["pattern"]
                func = rule.get("func")
                if func and callable(func):
                    result = re.sub(pattern, func, result)

        return result


    def vietnamese_tokenizer(self, text, min_words=3):
        # Pattern to split by:
        # 1. Conjunctions preceded by comma: , và | , nhưng | ... (Priority to capture full conjunction)
        # 2. Safe Dot: . followed by whitespace or end (not in abbreviations like TP.HCM, Q.3)
        # 3. Standard terminators: ? ! ; : \n
        # 4. Safe Comma: , not inside numbers (Fallback)

        # Safe dot rules:
        # - Must be followed by whitespace or end of string
        # - NOT preceded by single uppercase letter (TP., Q., P.)
        # - NOT followed directly by uppercase letter (.HCM)
        pattern = r'([,]\s*(?:và|nhưng|mà|thì|nên|để)\b|(?<![A-Z])[.](?=\s|$)|[?!;:\n]+|(?<!\d)[,]|[,](?!\d))'
        
        parts = re.split(pattern, text, flags=re.IGNORECASE)
        
        sentences = []
        current_sentence = ""
        
        for part in parts:
            if not part:
                continue
                
            # If it's a delimiter (matches the pattern)
            if re.match(pattern, part, re.IGNORECASE):
                current_sentence += part
                
                # Check word count
                word_count = len(current_sentence.strip().split())
                
                if word_count > min_words:
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
            else:
                current_sentence += part
                
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
            
        return sentences
    
    async def run_tts(self, text: str, language: Optional[str] = None) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using the custom TTS API.

        Splits text into sentences using vietnamese_tokenizer and streams each sentence.

        Args:
            text: Text to convert to speech.

        Yields:
            Frame: Audio and control frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        # Skip emotion-only frames - don't read them aloud
        if re.match(r'^\[EMO:\w+\]$', text.strip()):
            return

        # Strip inline emotion tags
        text = re.sub(r'\[EMO:\w+\]\s*', '', text)
        if not text.strip():
            return

        text = self.apply_text_rules(text)
        # Split text into sentences for streaming
        sentences = self.vietnamese_tokenizer(text)

        if not sentences:
            return

        url = f"{self._base_url}/tts"

        try:
            # Send TTS started frame once at the beginning
            yield TTSStartedFrame()

            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue

                logger.debug(f"{self}: TTS sentence {i+1}/{len(sentences)}: [{sentence}]")

                params = {
                    "seed": self._seed,
                    "voice": self._voice,
                    "nfe_step": self._nfe_step,
                    "text": sentence,
                }

                async with self._session.get(url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"{self} error: {error_text}")
                        yield ErrorFrame(error=f"Custom TTS API error: {error_text}")
                        continue

                    # Read the audio content
                    audio_data = await response.read()

                    if not audio_data:
                        logger.warning(f"{self}: Received empty audio data for sentence: {sentence}")
                        continue

                    # Stream audio for this sentence immediately
                    yield TTSAudioRawFrame(audio_data, self.sample_rate, num_channels=1)

            # Send TTS stopped frame when all sentences are done
            yield TTSStoppedFrame()

        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"{self} error: {e}")
            yield TTSStoppedFrame()

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            False, as metrics are not implemented for this custom service.
        """
        return False
