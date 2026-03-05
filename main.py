import asyncio
from google import genai
from logger import setup_logger, get_logger

from config import (
    config,
    get_domain,
    get_persona,
    get_role,
)
from agents.interviewer import InterviewerAgent
from agents.evaluator import EvaluatorAgent
from session.models import Session, SessionStatus

setup_logger()
log = get_logger(__name__)

def prompt_user_selection(cfg: dict) -> tuple[dict, dict, dict]:
    roles = cfg["interview"]["roles"]
    domains = cfg["interview"]["domains"]
    personas = cfg["interview"]["personas"]

    print("\n=== InterviewSense ===\n")

    log.info(f"Available roles: {', '.join(roles.keys())}")
    role_key = input("Select role: ").strip().lower()
    role = get_role(cfg, role_key)

    log.info(f"\nAvailable domains: {', '.join(domains.keys())}")
    domain_key = input("Select domain: ").strip().lower()
    domain = get_domain(cfg, domain_key)

    log.info(f"\nAvailable personas: {', '.join(personas.keys())}")
    persona_key = input("Select persona: ").strip().lower()
    persona = get_persona(cfg, persona_key)

    return role, domain, persona

def display_summary(session: Session):
    summary = session.summary
    if not summary:
        print("\nNo summary available.")
        return

    print("\n" + "=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)
    print(f"Session ID   : {session.human_readable_id}")
    print(f"Duration     : {session.duration_seconds:.0f} seconds")
    print(f"Overall Score: {summary.overall_score:.1f} / 5.0")
    print(f"Recommendation: {summary.recommendation.value.upper()}")

    print("\nStrengths:")
    for s in summary.strengths:
        print(f"  + {s}")

    print("\nAreas for Improvement:")
    for i in summary.improvements:
        print(f"  - {i}")

    print("\nTurn Breakdown:")
    for turn in session.turns:
        if turn.evaluation:
            print(f"  Turn {turn.turn_number}: {turn.evaluation.score}/5 — {turn.evaluation.feedback}")
            print(f"    Tags: {', '.join(turn.evaluation.tags)}")

    print("=" * 50)

async def run_interview(
        agent: InterviewerAgent, 
        session: Session,
) -> SessionStatus:
    opening = await agent.start_session()
    log.info(f"\nInterviewer: {opening}\n")

    while True:
        candidate_input = input("You: ").strip()

        if not candidate_input:
            return
        
        if candidate_input.lower() in {"exit", "quit", "bye"}:
            print("\nInterviewer: Thank you for your time. We'll be in touch!")
            log.info("Interview session ended by candidate")
            return SessionStatus.TERMINATED_EARLY

        response, is_end = await agent.next_turn(candidate_input)
        session.add_turn(
            candidate_input=candidate_input,
            interviewer_response=response
        )
        log.info(f"\nInterviewer: {response}\n")

        if is_end:
            log.info("Interview session completed")
            return SessionStatus.COMPLETED

async def main():
    log.info("InterviewSense starting up")

    role, domain, persona = prompt_user_selection(config)
    log.info(
        "Session config | role={} domain={} persona={}",
        role, domain, persona
    )

    client = genai.Client(
        project=config["gcp"]["project_id"],
        location=config["gcp"]["location"],
        vertexai=True
    )

    session = Session(role=role, domain=domain, persona=persona)
    log.info("Session created | session_id={}", session.session_id)

    interviewer = InterviewerAgent(
        client=client,
        config=config,
        role=role,
        domain=domain,
        persona=persona,
    )

    status = await run_interview(interviewer, session)
    session.close(status=status)
    log.info(
        "Session closed | status={} turns={}",
        status, len(session.turns)
    )

    if session.turns:
        print("\nEvaluating your performance. Please wait...\n")
        evaluator = EvaluatorAgent(client=client, config=config)
        session = await evaluator.evaluate(session)
        display_summary(session)
    else:
        print("\nNo turns recorded. Skipping evaluation.")

if __name__ == "__main__":
    asyncio.run(main())