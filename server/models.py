"""Validated browser/server contracts. Secrets never belong in these models."""

from typing import Annotated, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from src.reasoning import correction_output_token_limit, correction_settings
from src.translation import ENV_FILE

DEFAULT_MODEL_PAIR = "gpt-live-transcribe + gpt-6-luna"
LEGACY_MODEL_PAIR = "Breeze + OpenAI"
REALTIME_MODEL_PAIR = "gpt-realtime-translate"
MODEL_PAIRS = (DEFAULT_MODEL_PAIR, LEGACY_MODEL_PAIR, REALTIME_MODEL_PAIR)
TRANSLATION_TYPES = (
    "Compare all translations", "Fast English", "Corrected English", "Fully reviewed English",
)


def _correction_default(index):
    load_dotenv(ENV_FILE, override=False)
    return correction_settings()[index]


def _token_default():
    load_dotenv(ENV_FILE, override=False)
    return correction_output_token_limit()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class SessionSettings(Contract):
    model_pair: Literal["gpt-live-transcribe + gpt-6-luna", "Breeze + OpenAI", "gpt-realtime-translate"] = DEFAULT_MODEL_PAIR
    translation_type: Literal["Compare all translations", "Fast English", "Corrected English", "Fully reviewed English"] = "Compare all translations"
    speech_enabled: bool = False
    speech_model: Literal["gpt-4o-mini-tts", "tts-1-hd"] = "gpt-4o-mini-tts"
    astra_speech_mode: Literal["Live corrections", "Full review"] = "Live corrections"
    speaker_voices: bool = True
    diarization: bool = True
    corrections_enabled: bool = True
    correction_model: str = Field(default_factory=lambda: _correction_default(0), min_length=1, max_length=120)
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] = Field(default_factory=lambda: _correction_default(1))
    max_output_tokens: int = Field(default_factory=_token_default, ge=512, le=16384)
    confidence_threshold: float = Field(default=0.6, ge=0, le=1, allow_inf_nan=False)
    glossary_path: str = Field(default="", max_length=4096)
    dnt_path: str = Field(default="", max_length=4096)


class ReferenceOptions(Contract):
    format: Literal["auto", "plain", "transcript"] = "auto"
    sheet: str | None = Field(default=None, max_length=255)
    # Browser values are 1-based; zero explicitly means no header.
    header_row: int | None = Field(default=None, ge=0, le=10000)
    source_column: int | None = Field(default=None, ge=0, le=99)
    english_column: int | None = Field(default=None, ge=0, le=99)


class SimpleCommand(Contract):
    type: Literal["stop", "cancel"]


class RetryCommand(Contract):
    type: Literal["retry"]
    provider: Literal["openai", "tencent", "astra"]


class CorrectionsCommand(Contract):
    type: Literal["corrections"]
    enabled: bool


class SpeechCommand(Contract):
    type: Literal["speech"]
    enabled: bool
    model: Literal["gpt-4o-mini-tts", "tts-1-hd"] | None = None
    mode: Literal["Live corrections", "Full review"] | None = None
    speaker_voices: bool | None = None


class TerminologyCommand(Contract):
    type: Literal["terminology"]
    glossary_path: str = Field(default="", max_length=4096)
    dnt_path: str = Field(default="", max_length=4096)


Command = Annotated[
    SimpleCommand | RetryCommand | CorrectionsCommand | SpeechCommand | TerminologyCommand,
    Field(discriminator="type"),
]


class CaptureStart(Contract):
    type: Literal["capture.start"]
    sample_rate: int = Field(ge=8000, le=192000)
    channels: Literal[1]
    format: Literal["pcm_s16le"]


class SpeechAck(Contract):
    type: Literal["speech.ack"]
    speech_session_id: str = Field(min_length=1, max_length=64)
    played: int = Field(ge=0)


class ProviderResult(BaseModel):
    text: str | None = None
    status: str
    error: str | None = None


class SegmentSnapshot(BaseModel):
    id: str
    index: int
    source: str
    english: str | None = None
    status: str = "pending"
    start_s: float | None = None
    end_s: float | None = None
    speaker: str | None = None
    translations: dict[str, ProviderResult] = Field(default_factory=dict)


class SessionSnapshot(BaseModel):
    """Public snapshot schema; provider-specific audit metadata remains intact."""
    model_config = ConfigDict(extra="allow")
    id: str
    revision: int
    kind: Literal["microphone", "upload", "evaluation"]
    name: str | None = None
    status: str
    accepting: bool
    input_finished: bool
    stop_requested: bool = False
    finished: bool
    error: str | None = None
    settings: dict
    providers: list[str] = Field(default_factory=list)
    transcription_label: str = ""
    processing_disclosure: str = ""
    segments: list[SegmentSnapshot] = Field(default_factory=list)
    realtime: dict | None = None
    progress: dict = Field(default_factory=dict)
    correction: dict = Field(default_factory=dict)
    diarization: dict = Field(default_factory=dict)
    evaluation: dict | None = None
    speech: dict | None = None
