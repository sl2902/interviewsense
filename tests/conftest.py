import pytest

@pytest.fixture
def mock_config():
    return {
        "model": {"default": "gemini-2.0-flash-001"},
        "interview": {
            "personas": {
                "alex": {"name": "Alex", "tone": "calm, direct, and professional"}
            },
            "roles": {
                "data_engineer": {"name": "Data Engineer", "seniority": "Senior"}
            },
            "domains": {
                "data_engineering": {
                    "name": "Data Engineering",
                    "topics": ["ETL pipelines", "data modeling"]
                }
            }
        }
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
        role=mock_config["interview"]["roles"]["data_engineer"],
        domain=mock_config["interview"]["domains"]["data_engineering"],
        persona=mock_config["interview"]["personas"]["alex"],
    )