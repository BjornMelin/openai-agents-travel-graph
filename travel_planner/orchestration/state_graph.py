"""
State graph implementation for the travel planner system.

This module provides backward compatibility with the original state_graph.py
implementation, re-exporting the refactored components from their new locations.
"""

# Re-export the dependency injection system
from travel_planner.orchestration.core.agent_registry import (
    AgentRegistry,
    get_agent,
    register_agent,
    register_default_agents,
)

# Re-export the graph builder
from travel_planner.orchestration.core.graph_builder import create_planning_graph

# Re-export the node implementations
from travel_planner.orchestration.nodes.accommodation_search import accommodation_search
from travel_planner.orchestration.nodes.activity_planning import activity_planning
from travel_planner.orchestration.nodes.budget_management import budget_management
from travel_planner.orchestration.nodes.destination_research import destination_research
from travel_planner.orchestration.nodes.final_plan import generate_final_plan
from travel_planner.orchestration.nodes.flight_search import flight_search
from travel_planner.orchestration.nodes.parallel_search import (
    combine_search_results,
    create_parallel_search_branch,
)
from travel_planner.orchestration.nodes.query_analysis import query_analysis
from travel_planner.orchestration.nodes.transportation_planning import (
    transportation_planning,
)

# Re-export the parallel execution components
from travel_planner.orchestration.parallel import (
    ParallelResult,
    ParallelTask,
    combine_parallel_branch_results,
    execute_in_parallel,
    merge_parallel_results,
    parallel_search_tasks,
)

# Re-export the routing functions
from travel_planner.orchestration.routing.conditions import (
    continue_after_intervention,
    error_recoverable,
    has_error,
    needs_human_intervention,
    plan_complete,
    query_research_needed,
    recover_to_stage,
)
from travel_planner.orchestration.routing.error_recovery import (
    handle_error,
    handle_interruption,
)

# Re-export the checkpoint functionality
from travel_planner.orchestration.serialization.checkpoint import (
    load_state_checkpoint,
    save_state_checkpoint,
)
from travel_planner.orchestration.serialization.incremental import (
    load_incremental_checkpoint,
    save_incremental_checkpoint,
)
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage

# This allows for backward compatibility, so existing code will continue to work
# while we transition to the new modular structure.

__all__ = [
    "AgentRegistry",
    "ParallelResult",
    "ParallelTask",
    "TravelPlanningState",
    "WorkflowStage",
    "accommodation_search",
    "activity_planning",
    "budget_management",
    "combine_parallel_branch_results",
    "combine_search_results",
    "continue_after_intervention",
    "create_parallel_search_branch",
    "create_planning_graph",
    "destination_research",
    "error_recoverable",
    "execute_in_parallel",
    "flight_search",
    "generate_final_plan",
    "get_agent",
    "handle_error",
    "handle_interruption",
    "has_error",
    "load_incremental_checkpoint",
    "load_state_checkpoint",
    "merge_parallel_results",
    "needs_human_intervention",
    "parallel_search_tasks",
    "plan_complete",
    "query_analysis",
    "query_research_needed",
    "recover_to_stage",
    "register_agent",
    "register_default_agents",
    "save_incremental_checkpoint",
    "save_state_checkpoint",
    "transportation_planning",
]
