"""
Base node implementation for the travel planning workflow.

This module defines base functionality for workflow nodes, including
common execution patterns and error handling to reduce code duplication.
"""

from collections.abc import Callable
from typing import Any

from travel_planner.agents.base import BaseAgent
from travel_planner.data.models import TravelPlan
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def execute_agent_task(
    state: TravelPlanningState,
    agent: BaseAgent,
    task_name: str,
    complete_stage: WorkflowStage,
    result_formatter: Callable[[dict[str, Any]], str],
    result_processor: Callable[[TravelPlanningState, dict[str, Any]], None] | None = None,
) -> TravelPlanningState:
    """
    Generic function to execute an agent task and update state.

    Args:
        state: Current travel planning state
        agent: Agent to execute
        task_name: Name of the task (for logging and tracking)
        complete_stage: Workflow stage to set upon completion
        result_formatter: Function to format result message for conversation history
        result_processor: Optional function to process results and update state

    Returns:
        Updated travel planning state
    """
    logger.info(f"Executing {task_name} with {agent.__class__.__name__}")

    try:
        # Execute the agent
        result = agent.invoke(state)

        # Initialize plan if needed
        if state.plan is None:
            state.plan = TravelPlan()

        # Update workflow stage
        state.update_stage(complete_stage)

        # Format message and add to conversation history
        message = result_formatter(result)
        state.conversation_history.append({"role": "system", "content": message})

        # Process results if a processor is provided
        if result_processor:
            result_processor(state, result)

        # Add task result
        state.add_task_result(task_name, result)

        logger.info(f"Completed {task_name} successfully")
        return state

    except Exception as e:
        logger.error(f"Error in {task_name}: {e!s}")
        state.mark_error(f"Error during {task_name}: {e!s}")

        # Check if we should retry
        if state.should_retry(task_name):
            logger.info(
                f"Will retry {task_name} (attempt {state.retry_count.get(task_name, 0)})"
            )

        return state


def create_node_function(
    agent_class: type[BaseAgent],
    task_name: str,
    complete_stage: WorkflowStage,
    result_field: str,
    plan_field: str,
    message_template: str,
) -> Callable[[TravelPlanningState], TravelPlanningState]:
    """
    Factory function to create node execution functions with common implementation.

    Args:
        agent_class: The agent class to instantiate
        task_name: Name of the task
        complete_stage: Stage to set on completion
        result_field: Field in the result dictionary to extract
        plan_field: Field in the plan to update with results
        message_template: Template for the conversation message

    Returns:
        Node execution function
    """

    def result_formatter(result: dict[str, Any]) -> str:
        """Format the result for conversation history."""
        data = result.get(result_field, [])
        count = len(data) if isinstance(data, list) else 1 if data else 0
        return message_template.format(count=count)

    def result_processor(state: TravelPlanningState, result: dict[str, Any]) -> None:
        """Process results and update the plan."""
        if state.plan and result_field in result:
            setattr(state.plan, plan_field, result[result_field])

    def node_function(state: TravelPlanningState) -> TravelPlanningState:
        """The actual node execution function."""
        agent = agent_class()
        return execute_agent_task(
            state=state,
            agent=agent,
            task_name=task_name,
            complete_stage=complete_stage,
            result_formatter=result_formatter,
            result_processor=result_processor,
        )

    # Set function metadata
    node_function.__name__ = task_name
    node_function.__doc__ = f"Execute {task_name} using {agent_class.__name__}."

    return node_function
