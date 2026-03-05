from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from logger import get_logger
from session.models import (
    Session, 
    Summary, 
    TurnEvaluation, 
    Recommendation,
)
from agents.schemas import (
    TurnEvaluationResponse, 
    EvaluationResponse,
)

logger = get_logger(__name__)

EVALUATOR_PROMPT = """You are an expert technical interview evaluator.

You are evaluating a candidate interviewing for a {seniority} {role} position.

Expectations for this level: {seniority_context}

Domain: {domain}
Key topics: {topics}

Strictness: {strictness}
- high: Apply senior-level standards rigorously. Vague or incomplete answers 
  must score low regardless of partial correctness.
- medium: Apply reasonable standards. Allow some imprecision if fundamentals are sound.
- low: Be encouraging. Focus on problem-solving approach over correctness.

Valid evaluation tags: {tags}
Only use tags from the above list. Assign multiple tags per turn where appropriate.

Interview transcript:
{transcript}

Evaluate each turn strictly against the expectations for a {seniority} {role}.
A strong answer from a mid-level candidate is not necessarily strong at senior level."""

class EvaluatorAgent:
    def __init__(self, client: genai.Client, config: dict):
        self.client = client
        self.model = config["model"]["default"]
        self.config = config
        logger.info("EvaluatorAgent initialized | model={}", self.model)
    
    def _format_transcript(self, session: Session) -> str:
        lines = []
        for turn in session.turns:
            lines.append(f"[Turn {turn.turn_number}]")
            lines.append(f"Interviewer: {turn.interviewer_response}")
            lines.append(f"Candidate: {turn.candidate_input}")
            lines.append("")
        return "\n".join(lines)

    def _format_tags(self) -> str:
        tags = self.config["evaluation"]["tags"]
        all_tags = tags["positive"] + tags["negative"]
        return ", ".join(all_tags)
    
    def _build_prompt(self, session: Session) -> str:
        role = session.role
        domain = session.domain
        return EVALUATOR_PROMPT.format(
            role=role["name"],
            seniority=role["seniority"],
            seniority_context=role["seniority_context"],
            domain=domain["name"],
            topics=", ".join(domain["topics"]),
            strictness=role["strictness"],
            tags=self._format_tags(),
            transcript=self._format_transcript(session)
        )
    
    async def evaluate(self, session: Session) -> Session:
        if not session.turns:
            raise ValueError("Cannot evaluate session with no turns")

        logger.info(
            "Evaluating session | session_id={} turns={}",
            session.session_id, len(session.turns)
        )

        try:
            prompt = self._build_prompt(session)
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=EvaluationResponse
                )
            )

            result = EvaluationResponse.model_validate_json(response.text)
            logger.info("Evaluation response received and validated")

            # attach per-turn evaluations
            turn_map = {t.turn_number: t for t in session.turns}
            for turn_eval in result.turns:
                if turn_eval.turn_number in turn_map:
                    turn_map[turn_eval.turn_number].evaluation = TurnEvaluation(
                        score=turn_eval.score,
                        feedback=turn_eval.feedback,
                        tags=turn_eval.tags
                    )

            session.summary = Summary(
                overall_score=result.overall_score,
                strengths=result.strengths,
                improvements=result.improvements,
                recommendation=Recommendation(result.recommendation)
            )

            logger.info(
                "Evaluation complete | overall_score={} recommendation={}",
                result.overall_score, result.recommendation
            )
            return session

        except Exception as e:
            logger.error("Evaluation failed | error={}", e)
            raise