#!/usr/bin/env python3
"""Piper TTS via Wyoming protocol. Outputs WAV to stdout."""
import asyncio
import io
import sys
import wave

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize


async def main():
    host = sys.argv[1]
    port = int(sys.argv[2])
    voice = sys.argv[3] if len(sys.argv) > 3 else ""
    text = sys.stdin.read().strip()
    if not text:
        sys.exit(1)

    client = AsyncTcpClient(host, port)
    await asyncio.wait_for(client.connect(), timeout=30)

    try:
        synth = Synthesize(text=text)
        ev = synth.event()
        if voice:
            ev.data["voice"] = {"name": voice}
        await client.write_event(ev)

        pcm = b""
        rate, width, channels = 22050, 2, 1

        while True:
            event = await asyncio.wait_for(client.read_event(), timeout=60)
            if event is None:
                break
            if AudioStart.is_type(event.type):
                start = AudioStart.from_event(event)
                rate, width, channels = start.rate, start.width, start.channels
            elif AudioChunk.is_type(event.type):
                pcm += AudioChunk.from_event(event).audio
            elif AudioStop.is_type(event.type):
                break
    finally:
        await client.disconnect()

    if not pcm:
        sys.exit(1)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(pcm)

    sys.stdout.buffer.write(buf.getvalue())


if __name__ == "__main__":
    asyncio.run(main())
