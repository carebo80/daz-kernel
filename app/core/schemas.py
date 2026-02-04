
import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.core.settings import TOP_K_DEFAULT

class AskHybridRequest(BaseModel):
    question: str
    level: Optional[str] = None
    top_k: int = TOP_K_DEFAULT
    text_terms: Optional[List[str]] = None

class AskCitation(BaseModel):
    source: str
    page: Optional[int] = None
    chunk_index: Optional[int] = None
    score: float = 0.0

class AskResponse(BaseModel):
    answer: str
    citations: List[AskCitation] = Field(default_factory=list)

class PlanUnitRequest(BaseModel):
    topic: str
    level: str
    strong_group: bool = True
    time_start: str = "08:45"
    time_end: str = "11:15"
    top_k: int = TOP_K_DEFAULT
    text_terms: Optional[List[str]] = None
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    phase_model: str = "rita"

class PlanUnitResponse(BaseModel):
    unit_title: str
    meta: Dict[str, Any]
    ger: Dict[str, Any]
    language_support: Dict[str, Any] = Field(default_factory=dict)
    phases: List[Dict[str, Any]]
    materials: List[Dict[str, Any]]

class AskRequest(BaseModel):
    question: str
    top_k: int = TOP_K_DEFAULT

class UnitCitationIn(BaseModel):
    chunk_id: int
    score: float = 0.0
    quote: str = ""

class UnitCreateRequest(BaseModel):
    level: str                 # "A2"
    topic: Optional[str] = None  # "Bank" (Titel)
    topic_slug: Optional[str] = None  # "bank" (optional; wenn None -> aus topic abgeleitet)
    time_start: str = ""
    time_end: str = ""
    strong_group: bool = False
    title: str = ""
    notes: str = ""
    plan: Dict[str, Any] = Field(default_factory=dict)             # Feinplanung JSON
    language_support: Dict[str, Any] = Field(default_factory=dict)
    citations: List["UnitCitationIn"] = Field(default_factory=list)

class UnitCitationOut(BaseModel):
    id: str
    chunk_id: int
    score: float
    quote: str
    source: Optional[str] = None
    page: Optional[int] = None
    chunk_index: Optional[int] = None

class UnitResponse(BaseModel):
    id: str
    created_at: str
    updated_at: str
    level: str
    topic: Optional[str] = None
    time_start: str
    time_end: str
    strong_group: bool
    title: str
    notes: str
    plan: Dict[str, Any] = Field(default_factory=dict)
    language_support: Dict[str, Any] = Field(default_factory=dict)
    citations: List["UnitCitationOut"] = Field(default_factory=list)

class TopicOut(BaseModel):
    id: str
    slug: str
    title: str

class TopicWithUnitsOut(BaseModel):
    id: str
    slug: str
    title: str
    unit_count: int

class SearchHit(BaseModel):
    id: int
    source: str
    page: Optional[int] = None
    chunk_index: int
    score: float
    content: str

AskResponse.model_rebuild()
UnitCreateRequest.model_rebuild()
UnitResponse.model_rebuild()