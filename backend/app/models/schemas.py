# Pydantic modeli za HTTP teloto — FastAPI avtomatski validira i dava 422 pri greska.
# Limitot za question ide od settings.max_question_chars (ist kako sanitize + widget).
from __future__ import annotations

from pydantic import BaseModel, Field

from app.config import settings


class ChatRequest(BaseModel):
    """Vlez od vidzetot / curl: tekst + opciski session_id od sessionStorage."""
    question: str = Field(..., min_length=1, max_length=settings.max_question_chars)
    # Patternot se sovpagja so SESSION_ID_RE — falsifikat so spezijalen znak se otfrla tuka.
    session_id: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]{8,64}$",
    )


class Source(BaseModel):
    """Eden izvor pokraj odgovorot — studentot da moze da go proveri vo oficijalen dokument."""
    title: str
    url: str | None = None
    article_no: str | None = None
    source: str


class ChatResponse(BaseModel):
    """JSON odgovor od POST /chat (streamot koristi SSE, ne ovoj model)."""
    answer: str
    sources: list[Source]
    session_id: str
