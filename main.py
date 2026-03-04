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

async def run_interview(agent: InterviewerAgent):
    opening = await agent.start_session()
    log.info(f"\nInterviewer: {opening}\n")

    while True:
        candidate_input = input("You: ").strip()

        if not candidate_input:
            return
        
        if candidate_input.lower() in {"exit", "quit", "bye"}:
            print("\nInterviewer: Thank you for your time. We'll be in touch!")
            log.info("Interview session ended by candidate")
            break

        response = await agent.next_turn(candidate_input)
        log.info(f"\nInterviewer: {response.text}\n")

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

    agent = InterviewerAgent(
        client=client,
        config=config,
        role=role,
        domain=domain,
        persona=persona,
    )

    await run_interview(agent)

if __name__ == "__main__":
    asyncio.run(main())