"""Audio metadata schema for analyze_audio results."""

from pydantic import BaseModel


class AudioMetadata(BaseModel):
    key: str | None = None
    bpm: int | None = None
    duration: float | None = None
