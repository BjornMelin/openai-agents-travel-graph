"""
Unit tests for the parallel execution functionality.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from travel_planner.agents.base import BaseAgent
from travel_planner.data.models import TravelPlan
from travel_planner.orchestration.parallel import (
    ParallelResult,
    ParallelTask,
    combine_parallel_branch_results,
    execute_in_parallel,
    merge_parallel_results,
)
from travel_planner.orchestration.states.planning_state import TravelPlanningState


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock(spec=BaseAgent)
    agent.name = "MockAgent"
    agent.process = AsyncMock()
    agent.process.return_value = {"result": "test_result"}
    return agent


@pytest.fixture
def travel_state():
    """Create a basic travel planning state for testing."""
    from travel_planner.data.models import TravelQuery

    # Create a planning state with all required fields
    state = TravelPlanningState(
        query=TravelQuery(raw_query="Plan a trip to Paris"), plan=TravelPlan()
    )

    return state


@pytest.mark.asyncio
async def test_execute_in_parallel(mock_agent):
    """Test parallel execution of multiple agent tasks."""
    # Create a list of tasks with two mock agents
    agent1 = mock_agent
    agent1.name = "Agent1"
    agent1.process.return_value = {"data": "result1"}

    agent2 = MagicMock(spec=BaseAgent)
    agent2.name = "Agent2"
    agent2.process = AsyncMock()
    agent2.process.return_value = {"data": "result2"}

    tasks = [(agent1, {"param1": "value1"}), (agent2, {"param2": "value2"})]

    # Create a mock state
    state = MagicMock()

    # Execute tasks in parallel
    results = await execute_in_parallel(tasks, state)

    # Verify results
    assert "Agent1" in results
    assert "Agent2" in results
    assert results["Agent1"]["result"]["data"] == "result1"
    assert results["Agent2"]["result"]["data"] == "result2"
    assert results["Agent1"]["error"] is None
    assert results["Agent2"]["error"] is None

    # Verify agent.process was called with the correct parameters
    agent1.process.assert_called_once_with(param1="value1", context=state)
    agent2.process.assert_called_once_with(param2="value2", context=state)


@pytest.mark.asyncio
async def test_execute_in_parallel_with_error(mock_agent):
    """Test parallel execution with an agent that raises an exception."""
    # Create a list of tasks with two mock agents
    agent1 = mock_agent
    agent1.name = "Agent1"
    agent1.process.return_value = {"data": "result1"}

    agent2 = MagicMock(spec=BaseAgent)
    agent2.name = "Agent2"
    agent2.process = AsyncMock()
    agent2.process.side_effect = Exception("Test error")

    tasks = [(agent1, {"param1": "value1"}), (agent2, {"param2": "value2"})]

    # Create a mock state
    state = MagicMock()

    # Execute tasks in parallel
    results = await execute_in_parallel(tasks, state)

    # Verify results
    assert "Agent1" in results
    assert "Agent2" in results
    assert results["Agent1"]["result"]["data"] == "result1"
    assert results["Agent1"]["error"] is None
    assert results["Agent2"]["result"] is None
    assert results["Agent2"]["error"] == "Test error"


def test_merge_parallel_results(travel_state):
    """Test merging results from parallel execution into the state."""
    # Create sample results
    results = {
        "FlightSearchAgent": {
            "result": {"flights": [{"airline": "Air France", "price": 500}]},
            "error": None,
        },
        "AccommodationAgent": {
            "result": {"accommodations": [{"name": "Hotel Paris", "price": 200}]},
            "error": None,
        },
        "TransportationAgent": {
            "result": {
                "transportation": {"subway": {"price": 20}, "taxi": {"price": 50}}
            },
            "error": None,
        },
        "ActivityPlanningAgent": {
            "result": {"activities": {"day1": ["Visit Eiffel Tower", "Louvre Museum"]}},
            "error": None,
        },
    }

    # Merge results
    updated_state = merge_parallel_results(travel_state, results)

    # Verify merged results
    assert updated_state.plan.flights == [{"airline": "Air France", "price": 500}]
    assert updated_state.plan.accommodation == [{"name": "Hotel Paris", "price": 200}]
    assert updated_state.plan.transportation == {
        "subway": {"price": 20},
        "taxi": {"price": 50},
    }
    assert updated_state.plan.activities == {
        "day1": ["Visit Eiffel Tower", "Louvre Museum"]
    }
    assert not updated_state.plan.alerts


def test_merge_parallel_results_with_errors(travel_state):
    """Test merging results with errors from parallel execution."""
    # Create sample results with errors
    results = {
        "FlightSearchAgent": {
            "result": {"flights": [{"airline": "Air France", "price": 500}]},
            "error": None,
        },
        "AccommodationAgent": {"result": None, "error": "API rate limit exceeded"},
    }

    # Merge results
    updated_state = merge_parallel_results(travel_state, results)

    # Verify merged results
    assert updated_state.plan.flights == [{"airline": "Air France", "price": 500}]
    assert len(updated_state.plan.alerts) == 1
    assert "AccommodationAgent" in updated_state.plan.alerts[0]
    assert "API rate limit exceeded" in updated_state.plan.alerts[0]


def test_combine_parallel_branch_results(travel_state):
    """Test combining results from LangGraph parallel branch execution."""
    # Create sample branch results
    branch_results = {
        "result": True,
        "flight_search": ParallelResult(
            task_type=ParallelTask.FLIGHT_SEARCH,
            result={"flight_options": [{"airline": "Air France", "price": 500}]},
            completed=True,
        ),
        "accommodation": ParallelResult(
            task_type=ParallelTask.ACCOMMODATION,
            result={"accommodations": [{"name": "Hotel Paris", "price": 200}]},
            completed=True,
        ),
        "transportation": ParallelResult(
            task_type=ParallelTask.TRANSPORTATION,
            result={"transportation_options": {"subway": {"price": 20}}},
            completed=True,
        ),
        "activities": ParallelResult(
            task_type=ParallelTask.ACTIVITIES,
            result={"daily_itineraries": {"day1": ["Visit Eiffel Tower"]}},
            completed=True,
        ),
        "budget": ParallelResult(
            task_type=ParallelTask.BUDGET,
            result={
                "report": {
                    "total": 720,
                    "breakdown": {"flights": 500, "hotel": 200, "transportation": 20},
                }
            },
            completed=True,
        ),
    }

    # Combine branch results
    updated_state = combine_parallel_branch_results(travel_state, branch_results)

    # Verify combined results
    assert updated_state.plan.flights == [{"airline": "Air France", "price": 500}]
    assert updated_state.plan.accommodation == [{"name": "Hotel Paris", "price": 200}]
    assert updated_state.plan.transportation == {"subway": {"price": 20}}
    assert updated_state.plan.activities == {"day1": ["Visit Eiffel Tower"]}
    assert updated_state.plan.budget == {
        "total": 720,
        "breakdown": {"flights": 500, "hotel": 200, "transportation": 20},
    }
    assert not updated_state.plan.alerts


def test_combine_parallel_branch_results_with_errors(travel_state):
    """Test combining results with errors from parallel branch execution."""
    # Create sample branch results with errors
    branch_results = {
        "result": True,
        "flight_search": ParallelResult(
            task_type=ParallelTask.FLIGHT_SEARCH,
            result={"flight_options": [{"airline": "Air France", "price": 500}]},
            completed=True,
        ),
        "accommodation": ParallelResult(
            task_type=ParallelTask.ACCOMMODATION,
            result={},
            error="API error",
            completed=False,
        ),
    }

    # Combine branch results
    updated_state = combine_parallel_branch_results(travel_state, branch_results)

    # Verify combined results
    assert updated_state.plan.flights == [{"airline": "Air France", "price": 500}]
    assert len(updated_state.plan.alerts) == 1
    assert "accommodation" in updated_state.plan.alerts[0].lower()
    assert "API error" in updated_state.plan.alerts[0]
