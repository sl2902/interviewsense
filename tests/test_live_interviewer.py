import pytest
from session.models import Session, SessionStatus

class TestBuildConfig:
    def test_build_system_prompt_contains_persona(self, make_agent):
        live_agent = make_agent()

        prompt = live_agent._build_system_prompt()
        assert live_agent.persona["name"] in prompt
        assert live_agent.role["seniority"] in prompt
        assert live_agent.domain["name"] in prompt

    def test_build_live_config_sets_correct_voice(self, make_agent):
        live_agent = make_agent()

        config = live_agent._build_live_config()
        assert config.speech_config.voice_config.prebuilt_voice_config.voice_name == "Charon"

# normal turn commit

class TestNormalTurns:
    def test_basic_turn_committed(self, make_agent, make_session):
        agent = make_agent()
        session = make_session()

        agent.candidate_buffer = ["I have", " 5 years experience"]
        agent.interviewer_buffer = ["Tell me", " more about that."]

        result = agent._commit_turn(session)

        assert result == "continue"
        assert len(session.turns) == 1
        session.add_turn.assert_called_once_with(
            candidate_input="I have 5 years experience",
            interviewer_response="Tell me more about that.",
        )
    
    def test_buffers_cleared_after_commit(self, make_agent, make_session):
        agent = make_agent()
        session = make_session()

        agent.candidate_buffer = ["hello"]
        agent.interviewer_buffer = ["world"]

        agent._commit_turn(session)

        assert agent.candidate_buffer == []
        assert agent.interviewer_buffer == []
    
    def test_only_candidate_input(self, make_agent, make_session):
        agent = make_agent()
        session = make_session()

        agent.candidate_buffer = ["yes sure"]

        result = agent._commit_turn(session)

        assert result == "continue"
        assert len(session.turns) == 1
        session.add_turn.assert_called_once_with(
            candidate_input="yes sure",
            interviewer_response="",
        )
    
    def test_only_interviewer_response(self, make_agent, make_session):
        agent = make_agent()
        session = make_session()

        agent.interviewer_buffer = ["Good morning. Let's begin."]

        result = agent._commit_turn(session)

        assert result == "continue"
        assert len(session.turns) == 1
        session.add_turn.assert_called_once_with(
            candidate_input="",
            interviewer_response="Good morning. Let's begin.",
        )

# empty turns (interruption artifacts)

class TestEmptyTurns:
    def test_empty_buffers_no_commit(self, make_agent, make_session):
        agent = make_agent()
        session = make_session()

        result = agent._commit_turn(session)

        assert result == "continue"
        assert len(session.turns) == 0
        session.add_turn.assert_not_called()
    
    def test_whitespace_only_no_commit(self, make_agent, make_session):
        agent = make_agent()
        session = make_session()

        agent.candidate_buffer = ["  ", " "]
        agent.interviewer_buffer = ["", "  "]

        result = agent._commit_turn(session)

        assert result == "continue"
        assert len(session.turns) == 0
        session.add_turn.assert_not_called()

    def test_none_values_in_buffer_no_commit(self, make_agent, make_session):
        agent = make_agent()
        session = make_session()

        agent.candidate_buffer = [None, None]
        agent.interviewer_buffer = [None]

        result = agent._commit_turn(session)

        assert result == "continue"
        assert len(session.turns) == 0
        session.add_turn.assert_not_called()

    def test_buffers_cleared_even_when_empty(self, make_agent, make_session):
        agent = make_agent()
        session = make_session()

        agent._commit_turn(session)

        assert agent.candidate_buffer == []
        assert agent.interviewer_buffer == []

# end markers

class TestEndMarkers:
    def test_natural_end_with_content(self, make_agent, make_session):
        agent = make_agent()
        session = make_session()

        agent.candidate_buffer = ["no questions"]
        agent.interviewer_buffer = ["Thank you for coming in. [END_INTERVIEW]"]

        result = agent._commit_turn(session)

        assert result == "end"
        assert len(session.turns) == 1
        session.add_turn.assert_called_once_with(
            candidate_input="no questions",
            interviewer_response="Thank you for coming in.",
        )

    def test_natural_end_marker_only(self, make_agent, make_session):
        """[END_INTERVIEW] with no other content — no turn committed but still ends."""
        agent = make_agent()
        session = make_session()

        agent.interviewer_buffer = ["[END_INTERVIEW]"]

        result = agent._commit_turn(session)

        assert result == "end"
        assert len(session.turns) == 0
        session.add_turn.assert_not_called()

    def test_conduct_end_with_content(self, make_agent, make_session):
        agent = make_agent()
        session = make_session()

        agent.candidate_buffer = ["I don't know anything"]
        agent.interviewer_buffer = ["This interview is over. [END_INTERVIEW_CONDUCT]"]

        result = agent._commit_turn(session)

        assert result == "end"
        assert agent.session_status == SessionStatus.TERMINATED_EARLY
        assert len(session.turns) == 1
        session.add_turn.assert_called_once_with(
            candidate_input="I don't know anything",
            interviewer_response="This interview is over.",
        )

    def test_conduct_end_sets_status(self, make_agent, make_session):
        agent = make_agent()
        session = make_session()

        agent.interviewer_buffer = ["[END_INTERVIEW_CONDUCT]"]

        agent._commit_turn(session)

        assert agent.session_status == SessionStatus.TERMINATED_EARLY

    def test_conduct_end_takes_priority_over_natural_end(self, make_agent, make_session):
        agent = make_agent()
        session = make_session()

        agent.interviewer_buffer = ["Done. [END_INTERVIEW_CONDUCT] [END_INTERVIEW]"]

        result = agent._commit_turn(session)

        assert result == "end"
        assert agent.session_status == SessionStatus.TERMINATED_EARLY

    def test_end_marker_cleaned_from_response(self, make_agent, make_session):
        agent = make_agent()
        session = make_session()

        agent.candidate_buffer = ["bye"]
        agent.interviewer_buffer = ["HR will follow up. [END_INTERVIEW]"]

        agent._commit_turn(session)

        session.add_turn.assert_called_once_with(
            candidate_input="bye",
            interviewer_response="HR will follow up.",
        )

    def test_multiple_markers_all_cleaned(self, make_agent, make_session):
        agent = make_agent()
        session = make_session()

        agent.candidate_buffer = ["thanks"]
        agent.interviewer_buffer = ["Goodbye [END_INTERVIEW] [END_INTERVIEW_CONDUCT]"]

        agent._commit_turn(session)

        session.add_turn.assert_called_once_with(
            candidate_input="thanks",
            interviewer_response="Goodbye",
        )
# max turns

class TestMaxTurns:
    def test_wrap_up_at_max_turns(self, make_agent, make_session):
        agent = make_agent(max_turns=5)
        session = make_session(existing_turns=5)

        agent.candidate_buffer = ["my answer"]
        agent.interviewer_buffer = ["next question"]

        result = agent._commit_turn(session)

        assert result == "wrap_up"
        assert len(session.turns) == 6

    def test_wrap_up_over_max_turns(self, make_agent, make_session):
        agent = make_agent(max_turns=3)
        session = make_session(existing_turns=5)

        agent.candidate_buffer = ["another answer"]
        agent.interviewer_buffer = ["another question"]

        result = agent._commit_turn(session)

        assert result == "wrap_up"

    def test_no_wrap_up_below_max_turns(self, make_agent, make_session):
        agent = make_agent(max_turns=10)
        session = make_session(existing_turns=3)

        agent.candidate_buffer = ["answer"]
        agent.interviewer_buffer = ["question"]

        result = agent._commit_turn(session)

        assert result == "continue"

    def test_no_wrap_up_on_empty_turn_at_max(self, make_agent, make_session):
        """Empty turns don't count — shouldn't trigger wrap_up."""
        agent = make_agent(max_turns=5)
        session = make_session(existing_turns=5)

        result = agent._commit_turn(session)

        assert result == "continue"

    def test_natural_end_takes_priority_over_max_turns(self, make_agent, make_session):
        agent = make_agent(max_turns=5)
        session = make_session(existing_turns=4)

        agent.candidate_buffer = ["goodbye"]
        agent.interviewer_buffer = ["Thanks! [END_INTERVIEW]"]

        result = agent._commit_turn(session)

        assert result == "end"

# sequential calls (state isolation)

class TestSequentialCalls:
    def test_consecutive_turns_independent(self, make_agent, make_session):
        agent = make_agent()
        session = make_session()

        agent.candidate_buffer = ["first answer"]
        agent.interviewer_buffer = ["first question"]
        result1 = agent._commit_turn(session)

        agent.candidate_buffer = ["second answer"]
        agent.interviewer_buffer = ["second question"]
        result2 = agent._commit_turn(session)

        assert result1 == "continue"
        assert result2 == "continue"
        assert len(session.turns) == 2

    def test_empty_turn_between_real_turns(self, make_agent, make_session):
        agent = make_agent()
        session = make_session()

        agent.candidate_buffer = ["answer"]
        agent.interviewer_buffer = ["question"]
        agent._commit_turn(session)

        agent._commit_turn(session)  # empty

        agent.candidate_buffer = ["another answer"]
        agent.interviewer_buffer = ["another question"]
        agent._commit_turn(session)

        assert len(session.turns) == 2

    def test_end_after_multiple_turns(self, make_agent, make_session):
        agent = make_agent()
        session = make_session()

        for i in range(5):
            agent.candidate_buffer = [f"answer {i}"]
            agent.interviewer_buffer = [f"question {i}"]
            agent._commit_turn(session)

        agent.candidate_buffer = ["final"]
        agent.interviewer_buffer = ["Goodbye. [END_INTERVIEW]"]
        result = agent._commit_turn(session)

        assert result == "end"
        assert len(session.turns) == 6
