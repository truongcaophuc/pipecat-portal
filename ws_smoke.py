"""End-to-end WebSocket smoke test.

Sends RTVI client-ready to trigger welcome TTS, then collects audio frames.
Saves any audio received to welcome.wav so we can confirm OmniVoice produced real audio.
"""
import asyncio
import json
import wave
import websockets
from pipecat.frames.frames import StartFrame, TransportMessageUrgentFrame
from pipecat.serializers.protobuf import ProtobufFrameSerializer


async def main():
    uri = "ws://localhost:7860/ws"
    ser = ProtobufFrameSerializer()
    await ser.setup(StartFrame(audio_in_sample_rate=16000, audio_out_sample_rate=24000))

    print(f"Connecting {uri} ...")
    async with websockets.connect(uri, max_size=None) as ws:
        print("WS connected")

        # Send RTVI client-ready to trigger welcome flow
        ready_msg = TransportMessageUrgentFrame(
            message={"label": "rtvi-ai", "type": "client-ready"}
        )
        payload = await ser.serialize(ready_msg)
        if payload:
            await ws.send(payload)
            print("-> sent rtvi client-ready")

        recv_count = 0
        audio_chunks = []
        sample_rate = 24000
        num_channels = 1
        try:
            async with asyncio.timeout(30):
                async for msg in ws:
                    recv_count += 1
                    try:
                        frame = await ser.deserialize(msg)
                    except Exception as e:
                        print(f"<- #{recv_count}: deserialize err: {e}")
                        continue
                    name = type(frame).__name__ if frame else "<None>"
                    extra = ""
                    if getattr(frame, "audio", None):
                        audio_chunks.append(frame.audio)
                        sr = getattr(frame, "sample_rate", sample_rate) or sample_rate
                        sample_rate = sr
                        extra = f" sr={sr} {len(frame.audio)}B"
                    if getattr(frame, "text", None):
                        extra += f" text={frame.text!r}"
                    print(f"<- #{recv_count}: {name}{extra}")
                    if name == "TTSStoppedFrame" and audio_chunks:
                        break
        except asyncio.TimeoutError:
            print("(timeout)")

        total = sum(len(c) for c in audio_chunks)
        print(f"\nFrames: {recv_count} | audio chunks: {len(audio_chunks)} | total bytes: {total}")
        if audio_chunks:
            with wave.open("welcome.wav", "wb") as w:
                w.setnchannels(num_channels)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(b"".join(audio_chunks))
            print("Saved -> welcome.wav")


if __name__ == "__main__":
    asyncio.run(main())
