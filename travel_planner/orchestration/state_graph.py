"""
State graph implementation for the travel planner system.

This module provides backward compatibility with the original state_graph.py
implementation, re-exporting the refactored components from their new locations.
"""

# Re-export the state models
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage

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
from travel_planner.orchestration.nodes.transportation_planning import transportation_planning

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

# Re-export the parallel execution components
from travel_planner.orchestration.parallel import (
    ParallelResult,
    ParallelTask,
    combine_parallel_branch_results,
    execute_in_parallel,
    merge_parallel_results,
    parallel_search_tasks,
)

# Re-export the checkpoint functionality
from travel_planner.orchestration.serialization.checkpoint import (
    save_state_checkpoint,
    load_state_checkpoint,
)
from travel_planner.orchestration.serialization.incremental import (
    save_incremental_checkpoint,
    load_incremental_checkpoint,
)

# Re-export the dependency injection system
from travel_planner.orchestration.core.agent_registry import (
    AgentRegistry,
    get_agent,
    register_agent,
    register_default_agents,
)

# This allows for backward compatibility, so existing code will continue to work
# while we transition to the new modular structure.

__all__ = [
    # States
    "TravelPlanningState",
    "WorkflowStage",
    
    # Graph Builder
    "create_planning_graph",
    
    # Nodes
    "accommodation_search",
    "activity_planning",
    "budget_management",
    "destination_research",
    "flight_search",
    "generate_final_plan",
    "query_analysis",
    "transportation_planning",
    "combine_search_results",
    "create_parallel_search_branch",
    
    # Routing
    "continue_after_intervention",
    "error_recoverable",
    "has_error",
    "needs_human_intervention",
    "plan_complete",
    "query_research_needed",
    "recover_to_stage",
    "handle_error",
    "handle_interruption",
    
    # Parallel Execution
    "ParallelResult",
    "ParallelTask",
    "combine_parallel_branch_results",
    "execute_in_parallel",
    "merge_parallel_results",
    "parallel_search_tasks",
    
    # Checkpointing
    "save_state_checkpoint",
    "load_state_checkpoint",
    "save_incremental_checkpoint",
    "load_incremental_checkpoint",
    
    # Dependency Injection
    "AgentRegistry",
    "get_agent",
    "register_agent",
    "register_default_agents",
]