import pytest

pytestmark = pytest.mark.asyncio

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

    result = await agent.next_turn("I built an ETL pipeline using Spark and Airflow.")

    assert isinstance(result, str)
    assert len(result) > 0
    agent.chat.send_message.assert_called_once_with(
        "I built an ETL pipeline using Spark and Airflow."
    )

async def test_next_turn_raises_if_session_not_started(agent):
    with pytest.raises(RuntimeError, match="Session not started"):
        await agent.next_turn("some input")


async def test_next_turn_raises_on_api_error(agent, mock_client, mocker):
    agent.chat= mock_client.aio.chats.create.return_value
    agent.chat.send_message = mocker.AsyncMock(side_effect=Exception("API error"))

    with pytest.raises(Exception, match="API error"):
        await agent.next_turn("some input")