"""Request schemas for text and voice query endpoints."""

from typing import Optional
from pydantic import BaseModel, Field


class TextQueryRequest(BaseModel):
    """Payload for POST /api/query."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user query in natural language",
        examples=["What was the Manhattan Project?"]
    )
    language: Optional[str] = Field(
        default="en",
        description="Optional ISO 639-1 language code hint (default: en)",
        examples=["en", "hi", "te", "ta", "bn"]
    )


class VoiceQueryRequest(BaseModel):
    """Payload for POST /api/voice/query metadata when sending base64 or multipart audio."""

    language_hint: Optional[str] = Field(
        default="en-IN",
        description="BCP-47 language code hint for Sarvam STT (default: en-IN)",
        examples=["en-IN", "hi-IN", "te-IN", "ta-IN", "bn-IN"]
    )
    audio_format: Optional[str] = Field(
        default="wav",
        description="Audio format encoding: wav, pcm, mp3, ogg, or webm"
    )
