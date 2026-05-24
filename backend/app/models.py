"""Pydantic-схемы для API."""
from pydantic import BaseModel
from typing import Optional


class UploadResponse(BaseModel):
    filename: str
    duration_sec: float
    status: str


class TranscriptSegment(BaseModel):
    speaker: int
    text: str
    start_sec: float
    end_sec: float
    start_ts: str
    end_ts: str


class TranscribeResponse(BaseModel):
    filename: str
    original_filename: str = ""
    duration_sec: float
    num_segments: int
    segments: list[TranscriptSegment]
    transcript_text: str


class ProtocolRequest(BaseModel):
    transcript: str
    template_filename: Optional[str] = None
    template_text: Optional[str] = None
    original_filename: Optional[str] = None


class ProtocolResponse(BaseModel):
    protocol_text: str
    status: str


class ChatRequest(BaseModel):
    message: str
    transcript: str = ""
    history: list[dict] = []


class ChatResponse(BaseModel):
    reply: str
