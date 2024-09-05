import os
from typing import AsyncGenerator

from core.base.parsers.base_parser import AsyncParser
from core.parsers.media.openai_helpers import process_audio_with_openai
from core.telemetry.telemetry_decorator import telemetry_event

class AudioParser(AsyncParser[bytes]):
    """A parser for audio data."""

    def __init__(
        self, api_base: str = "https://api.openai.com/v1/audio/transcriptions"
    ):
        self.api_base = api_base
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")

    @telemetry_event("ingest_audio")
    async def ingest(self, data: bytes) -> AsyncGenerator[str, None]:
        """Ingest audio data and yield a transcription."""
        temp_audio_path = "temp_audio.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(data)
        try:
            transcription_text = process_audio_with_openai(
                open(temp_audio_path, "rb"), self.openai_api_key
            )
            yield transcription_text
        finally:
            os.remove(temp_audio_path)
