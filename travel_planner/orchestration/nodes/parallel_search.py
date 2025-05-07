"""
Parallel search node implementation for the travel planning workflow.

This module defines functions for setting up and combining results from
parallel searches for flights, accommodations, and transportation.
"""

from langgraph.graph.branches.parallel import ParallelBranch

from travel_planner.orchestration.nodes.accommodation_search import accommodation_task
from travel_planner.orchestration.nodes.flight_search import flight_search_task
from travel_planner.orchestration.nodes.transportation_planning import (
    transportation_task,
)
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def create_parallel_search_branch() -> ParallelBranch:
    """
    Create a parallel branch for searching flights, accommodation, and transportation.

    Returns:
        ParallelBranch for parallel search operations
    """
    branch = ParallelBranch("parallel_search")
    branch.add_node("flight_search", flight_search_task)
    branch.add_node("accommodation_search", accommodation_task)
    branch.add_node("transportation_planning", transportation_task)

    return branch


def combine_search_results(state: TravelPlanningState) -> TravelPlanningState:
    """
    Combine the results from the parallel search branch.

    Args:
        state: Current travel planning state

    Returns:
        Updated state with combined results from parallel search
    """
    logger.info("Combining results from parallel search")

    # Update the stage to indicate completion of parallel search
    state.update_stage(WorkflowStage.PARALLEL_SEARCH_COMPLETED)

    # Log combined results
    has_plan = state.plan is not None
    # Get counts of each type of result
    flights = state.plan.flights if has_plan else None
    flight_count = len(flights) if flights else 0

    accom = state.plan.accommodation if has_plan else None
    accom_count = len(accom) if accom else 0

    transport = state.plan.transportation if has_plan else None
    transport_count = len(transport) if transport else 0

    state.conversation_history.append(
        {
            "role": "system",
            "content": (
                f"Completed parallel search: {flight_count} flights, "
                f"{accom_count} accommodations, "
                f"{transport_count} transportation options"
            ),
        }
    )

    return state
