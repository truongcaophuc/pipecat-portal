"""
N8N LLM Service - extends BaseOpenAILLMService from pipecat.

Can use either n8n webhook OR OpenAI LLM for responses.
"""

from typing import List, Optional, Dict, Any, Literal, AsyncGenerator
from enum import Enum
import uuid
import json
import httpx

from pipecat.frames.frames import (
    Frame,
    LLMTextFrame,
)
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.processors.aggregators.llm_context import LLMContext
from omi_tts_service import process_tts_text


class ResponseMode(str, Enum):
    """Mode for how to generate responses."""
    N8N = "n8n"           # Always use n8n webhook
    OPENAI = "openai"     # Always use OpenAI LLM
    N8N_FIRST = "n8n_first"  # Try n8n first, fallback to OpenAI on error
    HYBRID = "hybrid"     # Let n8n decide (n8n can return use_llm=True)


class N8NLLMService(BaseOpenAILLMService):
    """
    N8N LLM Service that extends BaseOpenAILLMService.

    Provides flexibility to:
    - Use n8n webhook for responses
    - Use OpenAI LLM for responses
    - Hybrid mode: n8n decides whether to use LLM or return direct response
    """

    def __init__(
        self,
        *,
        # N8N config
        n8n_webhook_url: str,
        n8n_timeout: float = 30.0,
        session_id: Optional[str] = None,
        response_mode: ResponseMode = ResponseMode.N8N,
        # RAGFlow config
        ragflow_url: Optional[str] = None,
        ragflow_api_key: Optional[str] = None,
        # OpenAI config (inherited)
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize N8N LLM Service.

        Args:
            n8n_webhook_url: The n8n webhook URL to call
            n8n_timeout: HTTP request timeout for n8n calls
            session_id: Session ID for conversation tracking
            response_mode: How to handle responses (n8n, openai, n8n_first, hybrid)
            ragflow_url: RAGFlow chat completions URL
            ragflow_api_key: RAGFlow API key (Bearer token)
            model: OpenAI model to use when using LLM
            api_key: OpenAI API key
            base_url: Custom OpenAI base URL
        """
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
        self._n8n_webhook_url = n8n_webhook_url
        self._n8n_timeout = n8n_timeout
        self._session_id = session_id or str(uuid.uuid4())
        self._response_mode = response_mode
        self._n8n_client: Optional[httpx.AsyncClient] = None
        # RAGFlow
        self._ragflow_url = ragflow_url
        self._ragflow_api_key = ragflow_api_key

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def response_mode(self) -> ResponseMode:
        return self._response_mode

    @response_mode.setter
    def response_mode(self, mode: ResponseMode):
        """Allow changing response mode at runtime."""
        self._response_mode = mode

    async def start(self, frame: Frame):
        """Initialize clients when service starts."""
        await super().start(frame)
        self._n8n_client = httpx.AsyncClient(timeout=self._n8n_timeout)

    async def stop(self, frame: Frame):
        """Cleanup clients when service stops."""
        await super().stop(frame)
        if self._n8n_client:
            await self._n8n_client.aclose()
            self._n8n_client = None

    def _extract_user_text(self, messages: List[Dict]) -> str:
        """Extract the last user message text from messages."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            return part.get("text", "")
                        elif isinstance(part, str):
                            return part
        return ""

    async def _call_n8n(self, messages: List[Dict], user_text: str) -> Dict[str, Any]:
        """
        Call n8n webhook and return response.

        Returns dict with:
            - text: Response text
            - use_llm: (optional) If True, should use OpenAI LLM instead
            - error: (optional) Error message if failed
        """
        if not self._n8n_client:
            self._n8n_client = httpx.AsyncClient(timeout=self._n8n_timeout)

        try:
            response = await self._n8n_client.post(
                self._n8n_webhook_url,
                json={
                    "isGetBothAudioandText": False,
                    "isText": user_text,
                    "sessionId": self._session_id,
                    "messages": messages,
                }
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            return {"error": f"N8N HTTP error: {e.response.status_code}"}
        except httpx.TimeoutException:
            return {"error": "N8N timeout"}
        except Exception as e:
            return {"error": f"N8N error: {str(e)}"}

    async def _stream_n8n_sentences(self, user_text: str) -> AsyncGenerator[str, None]:
        """
        Stream n8n webhook response and yield complete sentences.

        Parses NDJSON format from n8n and yields text when sentence ends (., !, ?).
        """
        if not self._n8n_client:
            self._n8n_client = httpx.AsyncClient(timeout=self._n8n_timeout)

        try:
            async with self._n8n_client.stream(
                "POST",
                self._n8n_webhook_url,
                json={
                    "isGetBothAudioandText": False,
                    "isText": user_text,
                    "sessionId": self._session_id,
                }
            ) as response:
                response.raise_for_status()
                buffer = ""
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        print(">>>>", data)
                        if data.get("type") == "item":
                            content = data.get("content", "")
                            buffer += content
                            # Check for sentence endings
                            while any(sep in buffer for sep in [".", "!", "?"]):
                                for sep in [".", "!", "?"]:
                                    idx = buffer.find(sep)
                                    if idx != -1:
                                        sentence = buffer[:idx + 1].strip()
                                        buffer = buffer[idx + 1:]
                                        if sentence:
                                            print("n8n sentence: >> ", sentence)
                                            yield sentence
                                        break
                        elif data.get("type") == "end":
                            # Yield remaining buffer
                            if buffer.strip():
                                print("n8n final: >> ", buffer.strip())
                                yield buffer.strip()
                            break
                    except json.JSONDecodeError:
                        continue
                # Yield any remaining text
                if buffer.strip():
                    yield buffer.strip()

        except httpx.HTTPStatusError as e:
            yield f"Lỗi: N8N HTTP error: {e.response.status_code}"
        except httpx.TimeoutException:
            yield "Lỗi: N8N timeout"
        except Exception as e:
            yield f"Lỗi: {str(e)}"

    async def _stream_rag_sentences(self, user_text: str) -> AsyncGenerator[str, None]:
        """
        Stream RAGFlow API response and yield complete sentences.

        Parses SSE format from RAGFlow (OpenAI-compatible) and yields text when sentence ends.
        """
        if not self._ragflow_url or not self._ragflow_api_key:
            yield "Lỗi: RAGFlow chưa được cấu hình"
            return

        if not self._n8n_client:
            self._n8n_client = httpx.AsyncClient(timeout=self._n8n_timeout)

        try:
            async with self._n8n_client.stream(
                "POST",
                self._ragflow_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._ragflow_api_key}",
                },
                json={
                    "model": "ragflow",
                    "messages": [{"role": "user", "content": user_text}],
                    "stream": True,
                    "reference": False,
                }
            ) as response:
                print(f"RAGFlow response status: {response.status_code}")
                response.raise_for_status()
                buffer = ""

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    # SSE format: "data: {...}"
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            # Stream finished
                            if buffer.strip():
                                print("RAG final: >> ", buffer.strip())
                                yield buffer.strip()
                            break
                        try:
                            data = json.loads(data_str)
                            #print("RAG chunk: >> ", data)
                            # Extract content from OpenAI format
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    buffer += content
                                    # Check for sentence endings
                                    while any(sep in buffer for sep in [".", "!", "?", "\n"]):
                                        for sep in [".", "!", "?", "\n"]:
                                            idx = buffer.find(sep)
                                            if idx != -1:
                                                sentence = buffer[:idx + 1].strip()
                                                buffer = buffer[idx + 1:]
                                                if sentence:
                                                    #print("RAG sentence: >> ", sentence)
                                                    yield sentence
                                                break
                                # Check finish_reason
                                finish_reason = choices[0].get("finish_reason")
                                if finish_reason:
                                    if buffer.strip():
                                        print("RAG final: >> ", buffer.strip())
                                        yield buffer.strip()
                                        buffer = ""
                        except json.JSONDecodeError:
                            continue

                # Yield any remaining text
                if buffer.strip():
                    yield buffer.strip()

        except httpx.HTTPStatusError as e:
            yield f"Lỗi: RAGFlow HTTP error: {e.response.status_code}"
        except httpx.TimeoutException:
            yield "Lỗi: RAGFlow timeout"
        except Exception as e:
            yield f"Lỗi: {str(e)}"

    async def _process_context_n8n_streaming(self, context: LLMContext | LLMContext):
        """Process context using n8n webhook with streaming."""
        messages = context.get_messages()
        user_text = self._extract_user_text(messages)

        if not user_text:
            return

        await self.start_ttfb_metrics()
        first_sentence = True
        async for sentence in self._stream_n8n_sentences(user_text):
            if first_sentence:
                await self.stop_ttfb_metrics()
                first_sentence = False
            await self.push_frame(LLMTextFrame(sentence))

    async def _process_context_ragflow_streaming(self, user_text: str):
        """Process context using n8n webhook with streaming."""
        if not user_text:
            return

        await self.start_ttfb_metrics()
        first_sentence = True
        async for sentence in self._stream_rag_sentences(user_text):
            if first_sentence:
                await self.stop_ttfb_metrics()
                first_sentence = False
            print("sentence result", sentence)
            await self.push_frame(LLMTextFrame(sentence))        
    
    async def _process_context_n8n(self, context: LLMContext | LLMContext):
        """Process context using n8n webhook with streaming."""
        messages = context.get_messages()
        user_text = self._extract_user_text(messages)

        if not user_text:
            return

        await self.start_ttfb_metrics()
        result = await self._call_n8n(messages, user_text)
        await self.stop_ttfb_metrics()
        print("n8n result", result)

        # Send emotion frame if available
        emotion = result.get("emotion")
        if emotion:
            await self.push_frame(LLMTextFrame(f"[EMO:{emotion}]"))

        need_rag = result.get("need_rag")
        if "error" in result:
            await self.push_frame(LLMTextFrame(f"Có Lỗi, xin kiểm tra lại."))
        elif need_rag:
            await self.push_frame(LLMTextFrame("Xin Anh/Chị vui lòng chờ trong giây lát."))
            await self._process_context_ragflow_streaming(result.get("text"))
        elif result.get("text"):
            for chunk in process_tts_text(result["text"]):
                await self.push_frame(LLMTextFrame(chunk))

    async def _process_context_openai(self, context: LLMContext | LLMContext):
        """Process context using OpenAI LLM (parent implementation)."""
        await super()._process_context(context)

    async def _process_context_hybrid(self, context: LLMContext | LLMContext):
        """
        Hybrid mode: Call n8n first, let n8n decide if LLM should be used.

        n8n can return:
            - {"text": "response"} -> Use this response directly
            - {"use_llm": True} -> Forward to OpenAI LLM
            - {"use_llm": True, "system_prompt": "..."} -> Use LLM with custom prompt
        """
        messages = context.get_messages()
        user_text = self._extract_user_text(messages)

        if not user_text:
            return

        await self.start_ttfb_metrics()
        result = await self._call_n8n(messages, user_text)

        if "error" in result:
            # On n8n error, fallback to OpenAI
            await self._process_context_openai(context)
            return

        if result.get("use_llm", False):
            # n8n says use LLM
            # Optionally update system prompt if provided
            if result.get("system_prompt"):
                messages_copy = list(messages)
                if messages_copy and messages_copy[0].get("role") == "system":
                    messages_copy[0]["content"] = result["system_prompt"]
                else:
                    messages_copy.insert(0, {"role": "system", "content": result["system_prompt"]})
                context = LLMContext(messages_copy)

            await self._process_context_openai(context)
        else:
            # Use n8n response directly
            await self.stop_ttfb_metrics()
            if result.get("text"):
                for chunk in process_tts_text(result["text"]):
                    await self.push_frame(LLMTextFrame(chunk))

    async def _process_context(self, context: LLMContext | LLMContext):
        """
        Process context based on response_mode.

        Overrides BaseOpenAILLMService._process_context to add n8n support.
        """
        messages = context.get_messages()
        if len(messages) == 1:
            await self.push_frame(LLMTextFrame(messages[-1].get("content","Test")))
            #await self._process_context_openai(context)

        elif self._response_mode == ResponseMode.OPENAI:
            await self._process_context_openai(context)

        elif self._response_mode == ResponseMode.N8N:
            #await self._process_context_n8n_streaming(context)
            await self._process_context_n8n(context)

        elif self._response_mode == ResponseMode.N8N_FIRST:
            # Try n8n, fallback to OpenAI on error
            messages = context.get_messages()
            user_text = self._extract_user_text(messages)

            if not user_text:
                return

            await self.start_ttfb_metrics()
            result = await self._call_n8n(messages, user_text)

            if "error" in result:
                # Fallback to OpenAI
                await self._process_context_openai(context)
            else:
                await self.stop_ttfb_metrics()
                if result.get("text"):
                    for chunk in process_tts_text(result["text"]):
                        await self.push_frame(LLMTextFrame(chunk))

        elif self._response_mode == ResponseMode.HYBRID:
            await self._process_context_hybrid(context)

        else:
            # Default to n8n
            await self._process_context_n8n(context)
