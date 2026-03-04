from google import genai
from google.genai import types
from logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are {name}, a seasoned technical interviewer with 15+ years of experience hiring 
for {seniority} {role} positions. You specialize in {domain}, with particular focus on {topics}.

Your persona:
- Tone: {tone}
- You acknowledge good answers briefly ("Good.", "Fair enough.") before probing deeper
- You never say "Great question!", "Absolutely!", or use hollow affirmations
- You occasionally reference real-world scenarios: "At a previous company we faced this exact problem..."

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
    def __init__(
            self, 
            client: genai.Client, 
            config: dict, 
            role: dict,
            domain: dict,
            persona: dict,
        ):
        self.client = client
        self.model = config["model"]["default"]
        self.role = role
        self.domain = domain
        self.persona = persona
        self.chat = None
        logger.info(
                "InterviewerAgent initialized | persona={} role={} domain={} model={}",
                persona["name"], role["name"], domain["name"], self.model
            )
    
    async def start_session(self) -> str:
        """Initialize the chat session and get the opening question."""

        system_prompt = SYSTEM_PROMPT.format(
            name=self.persona["name"],
            tone=self.persona["tone"],
            role=self.role["name"],
            seniority=self.role["seniority"],
            domain=self.domain["name"],
            topics=", ".join(self.domain["topics"])
        )
        self.chat = self.client.aio.chats.create(
            model=self.model,
            config=types.GenerateContentConfig(
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

    
