from google import genai
from google.genai import types
from logger import get_logger

logger = get_logger(__name__)

END_SIGNAL = "[END_INTERVIEW]"
END_SIGNAL_CONDUCT = "[END_INTERVIEW_CONDUCT]"

SYSTEM_PROMPT = """You are {name}, a seasoned technical interviewer {experience} of experience hiring 
for {seniority} {role} positions. A {seniority} {role} is expected to have {seniority_context}.
You specialize in {domain}, with particular focus on {topics}.

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

Strictness level: {strictness}

Guidelines by strictness:
- high: You do not simplify questions or adjust difficulty downward. If a senior candidate 
  cannot answer a fundamental question, note it and move on — do not rephrase or offer easier 
  alternatives. Expect precise, experience-backed answers. Vague answers are not acceptable 
  at this level.
- medium: You may rephrase a question once if the candidate appears genuinely confused. 
  Allow some imprecision but expect sound fundamentals.
- low: You are patient and encouraging. Rephrase freely, offer hints generously, 
  focus on problem-solving approach over correctness.

Important rules:
- Never give away answers
- Never ask multiple questions at once
- If the candidate goes off-topic, redirect them politely but firmly
- Keep your responses concise — you are the interviewer, not the candidate
- If the candidate appears unserious — giving flippant, consistently one-word, irrelevant, or 
  disrespectful responses — politely end the interview early. Say something like: 
  "I appreciate you coming in today. We'll have HR follow up with you shortly." 
  Do not explain why you are ending it early.
- When the candidate responds with brief acknowledgments like "yes", "sure", "okay", "uh huh", or similar filler words, 
  do not treat these as a complete answer. Wait for the candidate to provide a substantive response before proceeding. 
  These are natural conversational fillers, not answers to your questions.
- When you have covered sufficient ground or the candidate has asked their closing questions,
  end the interview naturally and append exactly: [END_INTERVIEW]
- If the candidate appears unserious or unprofessional, end politely and append: [END_INTERVIEW_CONDUCT]"""

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
        self.turn_count = 0
        self.max_turns = config["interview"]["max_turns"]
        self.min_turns = config["interview"]["min_turns"]
        logger.info(
            "InterviewerAgent initialized | persona={} role={} domain={} model={} max_turns={}",
            persona["name"], role["name"], domain["name"], self.model, self.max_turns
        )
    
    async def start_session(self) -> str:
        """Initialize the chat session and get the opening question."""

        system_prompt = SYSTEM_PROMPT.format(
            name=self.persona["name"],
            tone=self.persona["tone"],
            experience=self.persona["experience"],
            role=self.role["name"],
            seniority=self.role["seniority"],
            seniority_context=self.role["seniority_context"],
            strictness=self.role["strictness"],
            domain=self.domain["name"],
            topics=", ".join(self.domain["topics"])
        )
        self.chat = self.client.aio.chats.create(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            )
        )
        self.turn_count = 1 # opening exchange counts as turn 1
        logger.info("Chat session started | persona={}", self.persona["name"])
        opening = await self.chat.send_message(
            "Begin the interview with a brief professional greeting and your first warm-up question."
        )
        return opening.text.strip()

    async def next_turn(self, candidate_input: str) -> tuple[str, bool]:
        """Send candidate response and get interviewer's next message."""
        if not self.chat:
            raise RuntimeError("Session not started. Call start_session() first.")
        
        if self.turn_count >= self.max_turns:
                logger.info("Max turns reached, ending session")
                return "We've covered a lot of ground today. Thanks for your time — we'll be in touch.", True

        try:
            # increment after successful exchange
            self.turn_count += 1
            logger.info(
                "Turn {} of {} | input_length={} chars",
                self.turn_count, self.max_turns, len(candidate_input)
            )
            
            response = await self.chat.send_message(candidate_input)
            reply = response.text.strip()

            # conduct-based end — always respected
            if END_SIGNAL_CONDUCT in reply:
                reply = reply.replace(END_SIGNAL_CONDUCT, "").strip()
                logger.info("Interviewer ended session due to candidate conduct")
                return reply, True
            
            is_end = END_SIGNAL in reply
            reply = reply.replace(END_SIGNAL, "").strip()

            # Don't allow early end signal before min_turns
            if is_end and self.turn_count < self.min_turns:
                is_end = False
                logger.warning(
                    "End signal ignored — min_turns not reached | turn={} min={}",
                    self.turn_count, self.min_turns
                )

            if is_end:
                logger.info("Interviewer signaled end of session")

            return reply, is_end
        
        except Exception as e:
            logger.error("Failed to get interviewer response | error={}", e)
            raise