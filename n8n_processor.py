from typing import List

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
)
import httpx
import uuid

class N8NProcessor(FrameProcessor):
    def __init__(self, n8n_webhook_url: str):
        super().__init__()
        self.webhook_url = n8n_webhook_url
        self.sessionId = str(uuid.uuid4())

    async def _process_context(self, context, direction):
        # Gọi n8n workflow
        messages: List = context.get_messages()
        print("@traced_llm 11", messages)

        # Lấy message cuối cùng của user
        user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_text = msg.get("content", "")
                break

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.webhook_url,
                json={"isGetBothAudioandText": False, "isText": user_text, "sessionId": self.sessionId}
            )
            #print("n8n call", response)
            result = response.json()
            print("n8n,",result)
           

        # Trả response về pipeline
        await self.push_frame(TextFrame(result["text"]), direction)


    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        # Nhận text từ user
        #print(f"Frame type: {type(frame).__name__}")
        context = None
        if isinstance(frame, LLMContextFrame):
            context = frame.context
            messages: List = context.get_messages()
            if len(messages) <= 1:
                print("return>>", messages[-1].get("content", ""))
                await self.push_frame(TextFrame(messages[-1].get("content", "")), direction)
                return
        else:
            await self.push_frame(frame, direction)

        if context:
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                await self._process_context(context, direction)
            except httpx.TimeoutException:
                await self._call_event_handler("on_completion_timeout")
            finally:
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())    


