"""
Graph builder for the travel planner workflow.

This module defines the functions for building the workflow state graph
using LangGraph. It connects the various nodes and defines the edges
and conditions for workflow transitions.
"""

from langgraph.graph import StateGraph
from langgraph.graph.branches.human import human_in_the_loop

from travel_planner.orchestration.nodes.activity_planning import activity_planning
from travel_planner.orchestration.nodes.budget_management import budget_management
from travel_planner.orchestration.nodes.destination_research import destination_research
from travel_planner.orchestration.nodes.final_plan import generate_final_plan
from travel_planner.orchestration.nodes.parallel_search import (
    combine_search_results,
    create_parallel_search_branch,
)
from travel_planner.orchestration.nodes.query_analysis import query_analysis
from travel_planner.orchestration.routing.conditions import (
    continue_after_intervention,
    error_recoverable,
    has_error,
    needs_human_intervention,
    query_research_needed,
    recover_to_stage,
)
from travel_planner.orchestration.routing.error_recovery import (
    handle_error,
    handle_interruption,
)
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def create_planning_graph() -> StateGraph:
    """
    Create an optimized state graph for travel planning with parallel execution.

    This graph implements the travel planning workflow with parallel branches
    for independent tasks, error handling, and support for human-in-the-loop
    interruptions.

    Returns:
        StateGraph instance that orchestrates the travel planning workflow
    """
    logger.info("Creating planning graph")

    # Create a new state graph
    workflow = StateGraph(TravelPlanningState)

    # Define the critical path nodes in the graph
    workflow.add_node("analyze_query", query_analysis)
    workflow.add_node("research_destination", destination_research)
    workflow.add_node("parallel_search", create_parallel_search_branch())
    workflow.add_node("combine_search_results", combine_search_results)
    workflow.add_node("plan_activities", activity_planning)
    workflow.add_node("manage_budget", budget_management)
    workflow.add_node("generate_final_plan", generate_final_plan)

    # Define error and interruption handling nodes
    workflow.add_node("handle_error", handle_error)
    workflow.add_node("handle_interruption", handle_interruption)

    # Define the edges in the graph (optimized flow)

    # Start with query analysis
    workflow.add_edge("START", "analyze_query")

    # Conditional: Need destination research or can move directly to search?
    workflow.add_conditional_edges(
        "analyze_query",
        query_research_needed,
        {
            "research_destination": "research_destination",
            "flight_search": "parallel_search",
        },
    )

    # After destination research, move to parallel search
    workflow.add_edge("research_destination", "parallel_search")

    # After parallel search, combine results
    workflow.add_edge("parallel_search", "combine_search_results")

    # After combining search results, move to activity planning
    workflow.add_edge("combine_search_results", "plan_activities")

    # After activity planning, move to budget management
    workflow.add_edge("plan_activities", "manage_budget")

    # After budget management, generate the final plan
    workflow.add_edge("manage_budget", "generate_final_plan")

    # After generating the final plan, end the workflow
    workflow.add_edge("generate_final_plan", "END")

    # Error handling edges - detect and handle errors at any stage
    for node in [
        "analyze_query",
        "research_destination",
        "parallel_search",
        "combine_search_results",
        "plan_activities",
        "manage_budget",
        "generate_final_plan",
    ]:
        workflow.add_conditional_edges(
            node,
            has_error,
            {
                "true": "handle_error",
                "false": node,  # Continue current node if no error
            },
        )

    # After error handling, either exit or route back into workflow
    workflow.add_conditional_edges(
        "handle_error",
        error_recoverable,
        {
            "true": recover_to_stage,  # Route to appropriate recovery point
            "false": "END",  # End if unrecoverable
        },
    )

    # Set up human-in-the-loop capability
    human_branch = human_in_the_loop()
    workflow.add_branch("human_in_the_loop", human_branch)

    # Add human intervention capabilities to critical nodes
    for node in [
        "analyze_query",
        "research_destination",
        "parallel_search",
        "plan_activities",
        "manage_budget",
        "generate_final_plan",
    ]:
        workflow.add_conditional_edges(
            node,
            needs_human_intervention,
            {
                "true": "human_in_the_loop",
                "false": node,  # Continue with node if no intervention needed
            },
        )

    # After human intervention, route back to appropriate point in workflow
    workflow.add_edge("human_in_the_loop", continue_after_intervention)

    # Compile the graph with optimizations
    logger.info("Planning graph created and compiled")
    return workflow.compile()
