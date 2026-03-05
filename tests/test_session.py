import pytest
from datetime import datetime
from session.models import (
    Session,
    SessionStatus,
    Recommendation,
    Turn,
    TurnEvaluation,
    Summary
)

# Session initialiation

def test_session_has_unique_id(mock_role, mock_domain, mock_persona):
    session1 = Session(role=mock_role, domain=mock_domain, persona=mock_persona)
    session2 = Session(role=mock_role, domain=mock_domain, persona=mock_persona)

    assert session1 != session2

def test_session_initial_status_is_in_progress(session):
    assert session.status == SessionStatus.IN_PROGRESS


def test_session_initial_turns_is_empty(session):
    assert session.turns == []


def test_session_initial_summary_is_none(session):
    assert session.summary is None


def test_session_initial_ended_at_is_none(session):
    assert session.ended_at is None

# add_turn

def test_add_turn_returns_turn_object(session):
    turn = session.add_turn("my answer", "interviewer response")
    assert isinstance(turn, Turn)

def test_add_turn_increments_turn_number(session):
    turn1 = session.add_turn("answer 1", "response 1")
    turn2 = session.add_turn("answer 2", "response 2")
    assert turn1.turn_number == 1
    assert turn2.turn_number == 2

def test_add_turn_stores_correct_inputs(session):
    turn = session.add_turn("my answer", "interviewer response")
    assert turn.candidate_input == "my answer"
    assert turn.interviewer_response == "interviewer response"

def test_add_turn_appends_to_session(session):
    session.add_turn("answer 1", "response 1")
    session.add_turn("answer 2", "response 2")
    assert len(session.turns) == 2

def test_turn_evaluation_can_be_attached(session):
    turn = session.add_turn("my answer", "interviewer response")
    turn.evaluation = TurnEvaluation(
        score=4,
        feedback="Good answer with clear examples.",
        tags=["good_depth", "clear"]
    )
    assert session.turns[0].evaluation.score == 4

# close

def test_close_sets_ended_at(session):
    session.close()
    assert session.ended_at is not None
    assert isinstance(session.ended_at, datetime)


def test_close_defaults_to_completed(session):
    session.close()
    assert session.status == SessionStatus.COMPLETED


def test_close_accepts_terminated_early(session):
    session.close(status=SessionStatus.TERMINATED_EARLY)
    assert session.status == SessionStatus.TERMINATED_EARLY


def test_close_rejects_invalid_status(session):
    with pytest.raises(ValueError):
        session.close(status="completed")

# duration seconds

def test_duration_seconds_is_none_before_close(session):
    assert session.duration_seconds is None


def test_duration_seconds_is_positive_after_close(session):
    session.close()
    assert session.duration_seconds >= 0

# human_readable_id

def test_human_readable_id_contains_role_slug(session):
    assert "data_engineer" in session.human_readable_id


def test_human_readable_id_contains_session_id_prefix(session):
    assert session.session_id[:8] in session.human_readable_id

# summary and recommendation

def test_summary_recommendation_enum(session):
    session.summary = Summary(
        overall_score=3.5,
        strengths=["clear communication"],
        improvements=["depth on distributed systems"],
        recommendation=Recommendation.HIRE
    )
    assert session.summary.recommendation == Recommendation.HIRE
    assert session.summary.recommendation.value == "hire"