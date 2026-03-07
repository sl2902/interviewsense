import asyncio
import numpy as np
import sounddevice as sd
import samplerate
from google.genai import types

from agents.interviewer import SYSTEM_PROMPT
from logger import get_logger
from session.models import Session, SessionStatus

logger = get_logger(__name__)


class LiveInterviewerAgent:
    def __init__(self, client, config, role, domain, persona):
        self.client = client
        self.config = config
        self.model = config["model"]["live"]
        self.role = role
        self.domain = domain
        self.persona = persona

        self.session = None
        self.loop = None
        self.interview_complete = False
        self.session_status = SessionStatus.COMPLETED

        self.candidate_buffer = []
        self.interviewer_buffer = []

        self.audio_in_queue = asyncio.Queue()
        self.audio_out_queue = asyncio.Queue()

        self.input_cfg  = config["audio"]["input"]
        self.output_cfg = config["audio"]["output"]
        self.system_cfg = config["audio"]["system"]["input"]

        # Hardware rate (mic/speaker) vs Gemini rates
        self.hw_rate         = self.system_cfg["sample_rate"]
        self.gemini_in_rate  = self.input_cfg["sample_rate"]
        self.gemini_out_rate = self.output_cfg["sample_rate"]

        self.resampler_in = samplerate.Resampler("sinc_fastest", channels=self.input_cfg["channels"])

        # Tracks whether audio is currently playing (used for drain + echo gate)
        self._playing = False

        # Set by receiver when Gemini sends interruption — speaker checks this
        self._interrupted = False

        # Mic gating: don't send mic audio until greeting finishes
        self._mic_ready = False
        self._got_greeting_audio = False

        # When True, the next turn_complete triggers graceful shutdown
        self._wrapping_up = False

        # Graceful shutdown: let final audio finish before exiting
        self._draining = False

        # Session resumption handle (survives reconnects)
        self._session_handle = None

        # Echo gate toggle: set to False when using headphones for barge-in
        self._echo_gate = config.get("audio", {}).get("echo_gate", True)

        logger.info(
            "LiveInterviewerAgent initialized | persona={} role={} domain={} voice={} echo_gate={}",
            persona["name"], role["name"], domain["name"], persona["voice"], self._echo_gate
        )

    def _build_system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            name=self.persona["name"],
            tone=self.persona["tone"],
            experience=self.persona["experience"],
            role=self.role["name"],
            seniority=self.role["seniority"],
            seniority_context=self.role["seniority_context"],
            domain=self.domain["name"],
            topics=", ".join(self.domain["topics"]),
            strictness=self.role["strictness"]
        )
        return base

    def _build_live_config(self) -> types.LiveConnectConfig:
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=self._build_system_prompt(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.persona["voice"]
                    )
                )
            ),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            session_resumption=types.SessionResumptionConfig(
                handle=self._session_handle
            ),
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=10000,
                sliding_window=types.SlidingWindow(target_tokens=512),
            ),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=False,
                    start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                    prefix_padding_ms=20,
                    silence_duration_ms=500,
                )
            ),
        )

    def _commit_turn(self, interview_session: Session) -> str:
        """Commit a turn if it has content.
        
        Returns:
            "continue"  — interview continues
            "end"       — natural end or conduct end, no wrap-up needed
            "wrap_up"   — max turns hit, send closing prompt
        """
        candidate_input      = " ".join([t.strip() for t in self.candidate_buffer if t]).strip()
        interviewer_response = " ".join([t.strip() for t in self.interviewer_buffer if t]).strip()

        is_conduct_end = "[END_INTERVIEW_CONDUCT]" in interviewer_response
        is_natural_end = "[END_INTERVIEW]" in interviewer_response

        clean_response = (
            interviewer_response
            .replace("[END_INTERVIEW_CONDUCT]", "")
            .replace("[END_INTERVIEW]", "")
            .strip()
        )

        has_content = bool(candidate_input) or bool(clean_response)

        if has_content or is_conduct_end or is_natural_end:
            if has_content:
                interview_session.add_turn(
                    candidate_input=candidate_input,
                    interviewer_response=clean_response,
                )
                logger.info(
                    "Turn committed | turn={} candidate_len={} response_len={}",
                    len(interview_session.turns), len(candidate_input), len(clean_response)
                )

        # Always clear buffers
        self.candidate_buffer.clear()
        self.interviewer_buffer.clear()

        if is_conduct_end:
            self.session_status = SessionStatus.TERMINATED_EARLY
            return "end"
        if is_natural_end:
            return "end"

        # Check max turns; let the candidate complete the last question before graceful exit
        max_turns = self.config["interview"]["max_turns"]
        if has_content and len(interview_session.turns) > max_turns:
            logger.info("Max turns reached | turns={}", len(interview_session.turns))
            return "wrap_up"

        return "continue"

    def _input_callback(self, indata, frames, time, status):
        """Hardware float32 48kHz -> int16 16kHz -> queue."""
        if status:
            logger.warning("Mic Status: {}", status)

        if not self.loop:
            return

        try:
            audio_float = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            ratio = self.gemini_in_rate / self.hw_rate
            resampled = self.resampler_in.process(audio_float.astype(np.float32), ratio)
            chunk = (resampled * 32767).astype(np.int16).tobytes()
            self.loop.call_soon_threadsafe(self.audio_in_queue.put_nowait, chunk)
        except Exception as e:
            logger.error("Input callback error: {}", e)

    async def _sender_loop(self):
        """Stream resampled mic audio to Gemini.

        Echo gate (configurable) mutes mic during playback to prevent
        false interruptions from speaker echo. Disable echo_gate in
        config when using headphones for barge-in support.
        """
        while not self.interview_complete:
            try:
                chunk = await asyncio.wait_for(self.audio_in_queue.get(), timeout=0.01)
                if not self._mic_ready:
                    continue
                if self._echo_gate and self._playing:
                    continue
                await self.session.send_realtime_input(
                    audio=types.Blob(
                        data=chunk,
                        mime_type=f"audio/pcm;rate={self.gemini_in_rate}"
                    )
                )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Sender error: {}", e)
                break

    async def _heartbeat_loop(self):
        """Send silent chunks every 5s to keep the connection alive."""
        silent_chunk = b"\x00" * 320
        while not self.interview_complete:
            try:
                await self.session.send_realtime_input(
                    audio=types.Blob(
                        data=silent_chunk,
                        mime_type=f"audio/pcm;rate={self.gemini_in_rate}"
                    )
                )
                await asyncio.sleep(5)
            except Exception as e:
                logger.error("Heartbeat error: {}", e)
                break

    async def _speaker_task(self):
        """Play Gemini 24kHz audio with barge-in support.

        Plays audio in small chunks (~100ms) so interruption events
        from the receiver can stop playback mid-sentence.
        """
        CHUNK_SAMPLES = self.gemini_out_rate // 10  # 100ms chunks

        stream = sd.OutputStream(
            samplerate=self.gemini_out_rate,
            channels=1,
            dtype="int16",
        )
        stream.start()

        try:
            while not self.interview_complete or self._draining:
                try:
                    audio_data = await asyncio.wait_for(
                        self.audio_out_queue.get(), timeout=0.1
                    )
                    self._playing = True
                    self._interrupted = False

                    offset = 0
                    while offset < len(audio_data):
                        if self._interrupted:
                            logger.info("Speaker interrupted — stopping playback.")
                            break
                        end = min(offset + CHUNK_SAMPLES, len(audio_data))
                        chunk = audio_data[offset:end]
                        await asyncio.to_thread(
                            stream.write, chunk.reshape(-1, 1)
                        )
                        offset = end

                    self._playing = False
                except asyncio.TimeoutError:
                    self._playing = False
                    if self._draining and self.audio_out_queue.empty():
                        break
                    continue
                except Exception as e:
                    self._playing = False
                    logger.error("Speaker error: {}", e)
        finally:
            stream.stop()
            stream.close()

    async def _receiver_loop(self, interview_session: Session):
        """
        Receive responses using __anext__ with timeout.
        Returns normally on stream close so the outer loop can reconnect.
        """
        while not self.interview_complete:
            try:
                response = await asyncio.wait_for(
                    self.session.receive().__anext__(), timeout=0.01
                )

                # Capture session resumption handle
                if (
                    hasattr(response, "session_resumption_update")
                    and response.session_resumption_update
                ):
                    update = response.session_resumption_update
                    if update.new_handle:
                        self._session_handle = update.new_handle
                        logger.info("Session handle saved for resumption.")

                if not response.server_content:
                    if hasattr(response, 'go_away') and response.go_away is not None:
                        logger.warning(
                            "GoAway received — connection ending soon. time_left={}",
                            response.go_away.time_left
                        )
                    continue

                content = response.server_content

                if content.interrupted:
                    logger.info("Interruption detected.")
                    self._interrupted = True
                    self._playing = False
                    while not self.audio_out_queue.empty():
                        self.audio_out_queue.get_nowait()
                    continue

                if content.input_transcription and content.input_transcription.text:
                    self.candidate_buffer.append(str(content.input_transcription.text))

                if content.output_transcription and content.output_transcription.text:
                    self.interviewer_buffer.append(str(content.output_transcription.text))

                if content.model_turn:
                    for part in content.model_turn.parts:
                        if part.inline_data:
                            chunk = np.frombuffer(part.inline_data.data, dtype=np.int16)
                            await self.audio_out_queue.put(chunk)
                            if not self._mic_ready:
                                self._got_greeting_audio = True

                if content.turn_complete:

                    candidate_text = "".join(self.candidate_buffer).strip()
                    if candidate_text:
                        print(f"\nYou: {candidate_text}")

                    interviewer_text = "".join(self.interviewer_buffer).strip()
                    if interviewer_text:
                        display_text = (
                            interviewer_text
                            .replace("[END_INTERVIEW_CONDUCT]", "")
                            .replace("[END_INTERVIEW]", "")
                            .strip()
                        )
                        if display_text:
                            print(f"\nInterviewer: {display_text}\n")

                    # If wrapping up, this is the closing statement — drain and exit
                    if self._wrapping_up:
                        self._commit_turn(interview_session)
                        self._draining = True
                        logger.info("Closing statement received — draining final audio...")
                        while not self.audio_out_queue.empty() or self._playing:
                            await asyncio.sleep(0.1)
                        self._draining = False
                        self.interview_complete = True
                        logger.info("Final audio drained. Interview complete.")
                        break

                    result = self._commit_turn(interview_session)

                    if result == "end":
                        # Natural end or conduct end — already has closing,
                        # just drain and exit
                        self._draining = True
                        logger.info("Interview ended — draining final audio...")
                        while not self.audio_out_queue.empty() or self._playing:
                            await asyncio.sleep(0.1)
                        self._draining = False
                        self.interview_complete = True
                        logger.info("Final audio drained. Interview complete.")
                        break

                    elif result == "wrap_up":
                        # Max turns — ask for closing statement, stay in loop
                        self._wrapping_up = True
                        logger.info("Sending wrap-up prompt...")
                        try:
                            await self.session.send_client_content(
                                turns=[types.Content(
                                    role="user",
                                    parts=[types.Part(
                                        text="Wrap up the interview now with a brief closing statement."
                                    )]
                                )],
                                turn_complete=True,
                            )
                        except Exception as e:
                            logger.error("Failed to send wrap-up: {}", e)
                            self._draining = True
                            while not self.audio_out_queue.empty() or self._playing:
                                await asyncio.sleep(0.1)
                            self._draining = False
                            self.interview_complete = True
                            break
                        continue  # Stay in loop to get closing response

                    # result == "continue"
                    logger.info("Turn complete. Keeping session alive...")

                    # Enable mic after greeting finishes
                    if not self._mic_ready:
                        if not interviewer_text and not self._got_greeting_audio:
                            logger.warning("Empty greeting — will reconnect and retry.")
                            self._session_handle = None
                            break
                        self._mic_ready = True
                        logger.info("Mic enabled — greeting complete.")

            except asyncio.TimeoutError:
                continue
            except StopAsyncIteration:
                logger.warning("Server closed the stream.")
                break
            except Exception as e:
                logger.error("Receiver error: {}: {}", type(e).__name__, e)
                break

    async def run(self, interview_session: Session) -> SessionStatus:
        logger.info("Opening Live Session...")
        self.loop = asyncio.get_running_loop()

        mic_stream = sd.InputStream(
            samplerate=self.hw_rate,
            channels=1,
            dtype="float32",
            blocksize=self.input_cfg["chunk_size"],
            callback=self._input_callback,
        )

        speaker_task = asyncio.create_task(self._speaker_task())

        with mic_stream:
            while not self.interview_complete:
                try:
                    logger.info(
                        "Connecting to Gemini (Resumption: {}, Handle: {})...",
                        self._session_handle is not None,
                        self._session_handle,
                    )

                    async with self.client.aio.live.connect(
                        model=self.model, config=self._build_live_config()
                    ) as session:
                        self.session = session
                        logger.info("Live connection established.")

                        if self._session_handle is not None:
                            self._mic_ready = True

                        if self._session_handle is None:
                            await session.send_client_content(
                                turns=[types.Content(
                                    role="user",
                                    parts=[types.Part(text="Start the interview.")]
                                )],
                                turn_complete=True,
                            )
                            logger.info("Initial prompt sent.")

                        sender_task    = asyncio.create_task(self._sender_loop())
                        heartbeat_task = asyncio.create_task(self._heartbeat_loop())

                        await self._receiver_loop(interview_session)

                        sender_task.cancel()
                        heartbeat_task.cancel()

                except Exception as e:
                    logger.error("Session error: {}: {}", type(e).__name__, e)
                    if self.interview_complete:
                        break
                    await asyncio.sleep(2)

        speaker_task.cancel()

        # Flush any remaining buffers
        interviewer_text = "".join(self.interviewer_buffer).strip()
        candidate_text   = "".join(self.candidate_buffer).strip()
        if interviewer_text or candidate_text:
            clean = (
                interviewer_text
                .replace("[END_INTERVIEW_CONDUCT]", "")
                .replace("[END_INTERVIEW]", "")
                .strip()
            )
            interview_session.add_turn(
                candidate_input=candidate_text,
                interviewer_response=clean,
            )
            logger.info("Final turn flushed | candidate_len={}", len(candidate_text))

        logger.info("Session cycle completed.")
        return self.session_status