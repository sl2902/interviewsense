import pytest
from session.models import (
    Session,
    SessionStatus,
    Recommendation,
    TurnEvaluation,
    Summary
)

# pytestmark = pytest.mark.asyncio

# initialization

def test_evaluator_initializes_with_correct_model(evaluator, mock_config):
    assert evaluator.model == mock_config["model"]["default"]


# _format_transcript

def test_format_transcript_includes_all_turns(evaluator, session_with_turns):
    transcript = evaluator._format_transcript(session_with_turns)
    assert "[Turn 1]" in transcript
    assert "[Turn 2]" in transcript


def test_format_transcript_includes_both_perspectives(evaluator, session_with_turns):
    transcript = evaluator._format_transcript(session_with_turns)
    assert "Interviewer:" in transcript
    assert "Candidate:" in transcript


# _format_tags

def test_format_tags_includes_positive_and_negative(evaluator, mock_config):
    tags = evaluator._format_tags()
    for tag in mock_config["evaluation"]["tags"]["positive"]:
        assert tag in tags
    for tag in mock_config["evaluation"]["tags"]["negative"]:
        assert tag in tags


# evaluate

async def test_evaluate_raises_on_empty_session(evaluator, session):
    with pytest.raises(ValueError, match="no turns"):
        await evaluator.evaluate(session)


async def test_evaluate_attaches_turn_evaluations(
    evaluator, session_with_turns, mock_client, mock_evaluation_response, mocker
):
    mock_response = mocker.MagicMock()
    mock_response.text = mock_evaluation_response.model_dump_json()
    mock_client.aio.models.generate_content = mocker.AsyncMock(
        return_value=mock_response
    )

    result = await evaluator.evaluate(session_with_turns)

    assert result.turns[0].evaluation is not None
    assert result.turns[0].evaluation.score == 3
    assert result.turns[1].evaluation is not None
    assert result.turns[1].evaluation.score == 4


async def test_evaluate_attaches_summary(
    evaluator, session_with_turns, mock_client, mock_evaluation_response, mocker
):
    mock_response = mocker.MagicMock()
    mock_response.text = mock_evaluation_response.model_dump_json()
    mock_client.aio.models.generate_content = mocker.AsyncMock(
        return_value=mock_response
    )

    result = await evaluator.evaluate(session_with_turns)

    assert result.summary is not None
    assert result.summary.overall_score == 3.5
    assert result.summary.recommendation == Recommendation.HIRE


async def test_evaluate_raises_on_api_error(
    evaluator, session_with_turns, mock_client, mocker
):
    mock_client.aio.models.generate_content = mocker.AsyncMock(
        side_effect=Exception("API error")
    )

    with pytest.raises(Exception, match="API error"):
        await evaluator.evaluate(session_with_turns)