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
    flight_count = len(state.plan.flights) if state.plan and state.plan.flights else 0
    accom_count = len(state.plan.accommodation) if state.plan and state.plan.accommodation else 0
    transport_count = len(state.plan.transportation) if state.plan and state.plan.transportation else 0
    
    state.conversation_history.append({
        "role": "system",
        "content": f"Completed parallel search: {flight_count} flights, {accom_count} accommodations, {transport_count} transportation options"
    })
    
    return state