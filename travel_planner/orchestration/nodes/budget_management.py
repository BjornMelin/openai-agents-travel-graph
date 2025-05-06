"""
Budget management node implementation for the travel planning workflow.

This module defines functions for managing the travel budget using the budget
management agent, both for individual execution and as part of parallel processing.
"""

from travel_planner.agents.budget_management import BudgetManagementAgent
from travel_planner.orchestration.nodes.base_node import (
    execute_agent_task,
)
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def budget_management(state: TravelPlanningState) -> TravelPlanningState:
    """
    Manage and optimize the budget for the trip.

    Args:
        state: Current travel planning state

    Returns:
        Updated travel planning state with budget information
    """

    def result_formatter(result):
        budget_report = result.get("report", {})
        total_budget = budget_report.get("total_budget", "Unknown")
        return f"Budget plan created with total: {total_budget}"

    def result_processor(state, result):
        if state.plan and "report" in result:
            state.plan.budget = result["report"]

    return execute_agent_task(
        state=state,
        agent=BudgetManagementAgent(),
        task_name="budget_management",
        complete_stage=WorkflowStage.BUDGET_MANAGED,
        result_formatter=result_formatter,
        result_processor=result_processor,
    )


def budget_task(state: TravelPlanningState) -> dict[str, any]:
    """
    Execute budget management task (usually runs after other tasks are complete).

    Args:
        state: Current travel planning state

    Returns:
        Dictionary with task results
    """
    from travel_planner.orchestration.parallel import ParallelResult, ParallelTask

    try:
        agent = BudgetManagementAgent()
        result = agent.invoke(state)

        return {
            "result": ParallelResult(
                task_type=ParallelTask.BUDGET, result=result, completed=True
            )
        }
    except Exception as e:
        logger.error(f"Error in budget task: {e!s}")
        return {
            "result": ParallelResult(
                task_type=ParallelTask.BUDGET, result={}, error=str(e), completed=False
            )
        }
