from google import genai
from google.genai import types
from logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a seasoned technical interviewer with 15+ years of experience hiring 
for {seniority} {role} positions specializing in {domain}.

Your interviewing style:
- Ask one focused question at a time
- Listen carefully to answers and ask relevant follow-ups based on what the candidate says
- Probe shallow or vague answers: if the candidate is hand-wavy, ask them to be specific
- Nudge without giving away answers if the candidate is stuck: offer a hint or reframe the question
- Call out buzzword-heavy answers: ask them to explain concepts in plain terms
- Control the pace: move on when an answer is sufficient
- Structure the session naturally: start with a warm-up, escalate difficulty gradually
- Be direct but encouraging — this should feel like a professional conversation, not an interrogation

Important rules:
- Never give away answers
- Never ask multiple questions at once
- If the candidate goes off-topic, redirect them politely but firmly
- Keep your responses concise — you are the interviewer, not the candidate"""

class InterviewerAgent:
    def __init__(self, client: genai.Client, config: dict, role: str, seniority: str, domain: str):
        self.client = client
        self.model = config["model"]["default"]
        self.role = role
        self.seniority = seniority
        self.domain = domain
        self.chat = None
        logger.info(
            "InterviewerAgent initialized | role={} seniority={} domain={} model={}",
            role, seniority, domain, self.model
        )
    
    async def start_session(self) -> str:
        """Initialize the chat session and get the opening question."""

        system_prompt = SYSTEM_PROMPT.format(
            role=self.role,
            seniority=self.seniority,
            domain=self.domain
        )
        self.chat = self.client.aio.chats.create(
            model=self.model,
            conig=types.GenerateContentConfig(
                system_instruction=system_prompt,
            )
        )
        logger.info("Chat session started")
        opening = await self.chat.send_message(
            "Begin the interview with a brief professional greeting and your first warm-up question."
        )
        return opening.text.strip()

    async def next_turn(self, candidate_input: str) -> str:
        """Send candidate response and get interviewer's next message."""
        if not self.chat:
            raise RuntimeError("Session not started. Call start_session() first.")

        logger.info("Candidate input received | length={} chars", len(candidate_input))
        try:
            response = await self.chat.send_message(candidate_input)
            reply = response.text.strip()
            logger.info("Interviewer response generated")
            return reply
        except Exception as e:
            logger.error("Failed to get interviewer response | error={}", e)
            raise

    
