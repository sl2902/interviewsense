from pydantic import(
    BaseModel, 
    Field, 
    computed_field,
)
from session.models import Recommendation

class TurnEvaluationResponse(BaseModel):
    turn_number: int
    score: int = Field(ge=1, le=5)
    feedback: str
    tags: list[str]

class EvaluationResponse(BaseModel):
    turns: list[TurnEvaluationResponse]
    overall_score: float = Field(ge=1.0, le=5.0)
    strengths: list[str]
    improvements: list[str]
    recommendation: Recommendation