import pytest

@pytest.fixture
def mock_config():
    return {
        "model": {"default": "gemini-2.0-flash-001"}
    }

@pytest.fixture
def mock_client(mocker):
    client = mocker.MagicMock()
    client.aio.chats.create.return_value = mocker.AsyncMock()
    return client

@pytest.fixture
def agent(mock_client, mock_config):
    from agents.interviewer import InterviewerAgent
    return InterviewerAgent(
        client=mock_client,
        config=mock_config,
        role="Data Engineer",
        seniority="Senior",
        domain="Data Engineering",
    )