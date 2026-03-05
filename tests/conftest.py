import pytest
from session.models import (
    Session,
)
from agents.evaluator import (
    EvaluatorAgent,
)
from agents.schemas import (
    EvaluationResponse, 
    TurnEvaluationResponse,
)

@pytest.fixture
def mock_config():
    return {
        "model": {"default": "gemini-2.0-flash-001"},
        "interview": {
            "max_turns": 20,
            "min_turns": 8,
            "personas": {
                "alex": {
                    "name": "Alex", 
                    "tone": "calm, direct, and professional", 
                    "experience": "15+ years"
                },
            },
            "roles": {
                "data_engineer": {
                    "name": "Data Engineer", 
                    "seniority": "Senior", 
                    "seniority_context": "5+ years experience, expected to lead pipeline design and mentor juniors",
                    "strictness": "high"
                },
            },
            "domains": {
                "data_engineering": {
                    "name": "Data Engineering",
                    "topics": ["ETL pipelines", "data modeling"],
                },
            },
        },
        "evaluation": {
            "tags": {
                "positive": ["good_depth", "clear_communication", "strong_example"],
                "negative": ["vague", "lacks_depth", "buzzword_heavy"]
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

@pytest.fixture
def mock_role(mock_config):
    return mock_config["interview"]["roles"]["data_engineer"]


@pytest.fixture
def mock_domain(mock_config):
    return mock_config["interview"]["domains"]["data_engineering"]


@pytest.fixture
def mock_persona(mock_config):
    return mock_config["interview"]["personas"]["alex"]


@pytest.fixture
def session(mock_role, mock_domain, mock_persona):
    return Session(role=mock_role, domain=mock_domain, persona=mock_persona)


@pytest.fixture
def evaluator(mock_client, mock_config):
    return EvaluatorAgent(client=mock_client, config=mock_config)


@pytest.fixture
def session_with_turns(session):
    session.add_turn(
        candidate_input="I built ETL pipelines using Spark and Airflow.",
        interviewer_response="Can you elaborate on the architecture?"
    )
    session.add_turn(
        candidate_input="We used a medallion architecture with bronze, silver and gold layers.",
        interviewer_response="How did you handle schema evolution?"
    )
    return session


@pytest.fixture
def mock_evaluation_response():
    return EvaluationResponse(
        turns=[
            TurnEvaluationResponse(
                turn_number=1,
                score=3,
                feedback="Decent answer but lacks depth on implementation details.",
                tags=["vague", "lacks_depth"]
            ),
            TurnEvaluationResponse(
                turn_number=2,
                score=4,
                feedback="Good architectural knowledge with concrete example.",
                tags=["good_depth", "strong_example"]
            )
        ],
        overall_score=3.5,
        strengths=["good architectural knowledge", "clear communication"],
        improvements=["provide more implementation details", "discuss trade-offs"],
        recommendation="hire"
    )