import pytest

# pytestmark = pytest.mark.asyncio

async def test_start_session_returns_opening_question(agent, mock_client, mocker):
    mock_response = mocker.MagicMock()
    mock_response.text = "  Welcome! Can you walk me through your data pipeline experience?  "
    agent.chat = mock_client.aio.chats.create.return_value
    agent.chat.send_message = mocker.AsyncMock(return_value=mock_response)

    result = await agent.start_session()

    assert isinstance(result, str)
    assert len(result) > 0
    assert result == mock_response.text.strip()

async def test_next_turn_returns_interviewer_response(agent, mock_client, mocker):
    mock_response = mocker.MagicMock()
    mock_response.text = "How did you handle schema evolution in that pipeline?"
    agent.chat = mock_client.aio.chats.create.return_value
    agent.chat.send_message = mocker.AsyncMock(return_value=mock_response)

    response, is_end = await agent.next_turn("I built an ETL pipeline using Spark and Airflow.")

    assert isinstance(response, str)
    assert len(response) > 0
    assert is_end is False

    agent.chat.send_message.assert_called_once_with(
        "I built an ETL pipeline using Spark and Airflow."
    )

async def test_next_turn_detects_end_signal(agent, mock_client, mocker):
    mock_response = mocker.MagicMock()
    mock_response.text = "Thanks for your time. We'll be in touch. [END_INTERVIEW]"
    agent.chat = mock_client.aio.chats.create.return_value
    agent.chat.send_message = mocker.AsyncMock(return_value=mock_response)
    agent.turn_count = agent.min_turns

    response, is_end = await agent.next_turn("No questions from my side.")
    assert is_end is True
    assert "[END_INTERVIEW]" not in response

async def test_next_turn_respects_conduct_end_before_min_turns(agent, mock_client, mocker):
    mock_response = mocker.MagicMock()
    mock_response.text = "We'll have HR follow up with you shortly. [END_INTERVIEW_CONDUCT]"
    agent.chat = mock_client.aio.chats.create.return_value
    agent.chat.send_message = mocker.AsyncMock(return_value=mock_response)
    agent.turn_count = 0  # well before min_turns

    response, is_end = await agent.next_turn("ugh whatever")

    assert is_end is True
    assert "[END_INTERVIEW_CONDUCT]" not in response

async def test_next_turn_ignores_end_signal_before_min_turns(agent, mock_client, mocker):
    mock_response = mocker.MagicMock()
    mock_response.text = "Thanks for your time. [END_INTERVIEW]"
    agent.chat = mock_client.aio.chats.create.return_value
    agent.chat.send_message = mocker.AsyncMock(return_value=mock_response)
    agent.turn_count = agent.min_turns - 2 # 1 less than min_turns

    response, is_end = await agent.next_turn("No questions from my side.")

    assert is_end is False

async def test_next_turn_ends_at_max_turns(agent, mock_client, mocker):
    agent.chat = mock_client.aio.chats.create.return_value
    agent.turn_count = agent.max_turns

    response, is_end = await agent.next_turn("some input")

    assert is_end is True
    assert isinstance(response, str)

async def test_next_turn_raises_if_session_not_started(agent):
    with pytest.raises(RuntimeError, match="Session not started"):
        await agent.next_turn("some input")


async def test_next_turn_raises_on_api_error(agent, mock_client, mocker):
    agent.chat= mock_client.aio.chats.create.return_value
    agent.chat.send_message = mocker.AsyncMock(side_effect=Exception("API error"))

    with pytest.raises(Exception, match="API error"):
        await agent.next_turn("some input")

async def test_start_session_returns_opening_question(agent, mock_client, mocker):
    mock_response = mocker.MagicMock()
    mock_response.text = "Begin the interview with a brief professional greeting and your first warm-up question."
    agent.chat = mock_client.aio.chats.create.return_value
    agent.chat.send_message = mocker.AsyncMock(return_value=mock_response)
    result = await agent.start_session()

    assert isinstance(result, str)
    assert len(result) > 0
    assert result == mock_response.text.strip()
    assert agent.turn_count == 1 