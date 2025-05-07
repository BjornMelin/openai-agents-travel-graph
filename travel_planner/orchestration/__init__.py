"""
Orchestration package for the travel planner system.

This package implements the workflow orchestration for the travel planner, connecting
specialized agents through a LangGraph state graph with support for parallel execution,
error recovery, and human-in-the-loop interruptions.
"""

from travel_planner.orchestration.core import (
    AgentRegistry,
    create_planning_graph,
    get_agent,
    register_agent,
    register_default_agents,
)
from travel_planner.orchestration.parallel import (
    ParallelResult,
    ParallelTask,
    execute_in_parallel,
    parallel_search_tasks,
)
from travel_planner.orchestration.serialization import (
    load_incremental_checkpoint,
    load_state_checkpoint,
    save_incremental_checkpoint,
    save_state_checkpoint,
)
from travel_planner.orchestration.states import TravelPlanningState, WorkflowStage

__all__ = [
    "AgentRegistry",
    "ParallelResult",
    # Parallel execution
    "ParallelTask",
    # States
    "TravelPlanningState",
    "WorkflowStage",
    # Core
    "create_planning_graph",
    "execute_in_parallel",
    "get_agent",
    "load_incremental_checkpoint",
    "load_state_checkpoint",
    "parallel_search_tasks",
    "register_agent",
    "register_default_agents",
    "save_incremental_checkpoint",
    # Serialization
    "save_state_checkpoint",
]
