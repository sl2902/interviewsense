from datetime import datetime, timezone
from enum import StrEnum
from typing import Optional
from uuid import uuid4

from pydantic import(
    BaseModel, 
    Field, 
    computed_field,
    ConfigDict,
)

class SessionStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    TERMINATED_EARLY = "terminated_early"

class Recommendation(StrEnum):
    STRONG_HIRE = "strong_hire"
    HIRE = "hire"
    NO_HIRE = "no_hire"

class TurnEvaluation(BaseModel):
    score: int = Field(ge=1, le=5)
    feedback: str
    tags: list[str] = Field(default_factory=list)

class Turn(BaseModel):
    turn_number: int = Field(ge=1)
    candidate_input: str
    interviewer_response: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    evaluation: Optional[TurnEvaluation] = None

class Summary(BaseModel):
    overall_score: float = Field(ge=1.0, le=5.0)
    strengths: list[str]
    improvements: list[str]
    recommendation: Recommendation

class Session(BaseModel):
    role: dict
    domain: dict
    persona: dict
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None
    status: SessionStatus = SessionStatus.IN_PROGRESS
    turns: list[Turn] = Field(default_factory=list)
    summary: Optional[Summary] = None

    def add_turn(self, candidate_input: str, interviewer_response: str) -> Turn:
        turn = Turn(
            turn_number=len(self.turns) + 1,
            candidate_input=candidate_input,
            interviewer_response=interviewer_response
        )
        self.turns.append(turn)
        return turn

    def close(self, status: SessionStatus = SessionStatus.COMPLETED):
        if not isinstance(status, SessionStatus):
            raise ValueError("Invalid status. Must be a SessionStatus enum value")
        self.ended_at = datetime.now(timezone.utc)
        self.status = status

    @computed_field
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.ended_at:
            return (self.ended_at - self.created_at).total_seconds()
        return None

    @computed_field
    @property
    def human_readable_id(self) -> str:
        role_slug = self.role["name"].lower().replace(" ", "_")
        date_slug = self.created_at.strftime("%Y%m%d_%H%M")
        return f"{role_slug}_{date_slug}_{self.session_id[:8]}"
