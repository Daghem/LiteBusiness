from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    source: str
    excerpt: str | None = None
    chunk_id: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    score: float | None = None


class ChatRequest(BaseModel):
    content: str
    regime_id: str | None = None
    chat_id: str | None = None


class ChatResponse(BaseModel):
    message: str
    sources: List[str] = Field(default_factory=list)
    source_details: List[SourceRef] = Field(default_factory=list)
    confidence_label: str | None = None
    confidence_score: float | None = None
    retrieval_mode: str | None = None
    regime_id: str | None = None
    chat_id: str | None = None


class RegimeOption(BaseModel):
    regime_id: str
    label: str
    is_default: bool = False


class ChatTurnPayload(BaseModel):
    chat_id: str
    regime_id: str | None = None
    user_message: str
    assistant_message: str
    assistant_sources: List[str] = Field(default_factory=list)
    confidence_label: str | None = None
    confidence_score: float | None = None
    retrieval_mode: str | None = None


class ChatMessage(BaseModel):
    role: str
    text: str
    sources: List[str] = Field(default_factory=list)
    created_at: str


class ChatSummary(BaseModel):
    chat_id: str
    title: str
    regime_id: str | None = None
    updated_at: str
    message_count: int


class ChatTranscript(BaseModel):
    chat_id: str
    title: str
    regime_id: str | None = None
    created_at: str
    updated_at: str
    messages: List[ChatMessage] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    chat_id: str | None = None
    message: str
    vote: str
    regime_id: str | None = None
    assistant_sources: List[str] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    status: str


class SimulationRequest(BaseModel):
    regime_id: str = "forfettario"
    ricavi: float
    ateco_code: str | None = None
    coefficiente_redditivita: float | None = None
    aliquota_imposta: float = 0.15
    gestione_previdenziale: str = "nessuna"
    aliquota_contributiva: float | None = None
    riduzione_inps_35: bool = False


class SimulationBreakdownItem(BaseModel):
    label: str
    amount: float


class SimulationResponse(BaseModel):
    regime_id: str
    coefficiente_redditivita: float
    aliquota_imposta: float
    aliquota_contributiva: float
    imponibile_stimato: float
    contributi_stimati: float
    imposta_sostitutiva_stimata: float
    netto_stimato: float
    breakdown: List[SimulationBreakdownItem] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class AdminStats(BaseModel):
    total_chats: int
    total_messages: int
    positive_feedback: int
    negative_feedback: int
    no_result_events: int
    low_confidence_events: int
    top_questions: List[str] = Field(default_factory=list)
    top_regimes: List[str] = Field(default_factory=list)
