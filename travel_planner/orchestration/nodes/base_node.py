"""
Base node implementation for the travel planning workflow.

This module defines base functionality for workflow nodes, including
common execution patterns and error handling to reduce code duplication.
"""

from collections.abc import Callable
from typing import Any

from travel_planner.data.models import AgentTaskParams, NodeFunctionParams, TravelPlan
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def execute_agent_task(params: AgentTaskParams) -> TravelPlanningState:
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
    logger.info(f"Executing {params.task_name} with {params.agent.__class__.__name__}")

    try:
        # Execute the agent
        result = params.agent.invoke(params.state)

        # Initialize plan if needed
        if params.state.plan is None:
            params.state.plan = TravelPlan()

        # Update workflow stage
        params.state.update_stage(params.complete_stage)

        # Format message and add to conversation history
        message = params.result_formatter(result)
        params.state.conversation_history.append({"role": "system", "content": message})

        # Process results if a processor is provided
        if params.result_processor:
            params.result_processor(params.state, result)

        # Add task result
        params.state.add_task_result(params.task_name, result)

        logger.info(f"Completed {params.task_name} successfully")
        return params.state

    except Exception as e:
        logger.error(f"Error in {params.task_name}: {e!s}")
        params.state.mark_error(f"Error during {params.task_name}: {e!s}")

        # Check if we should retry
        if params.state.should_retry(params.task_name):
            logger.info(
                f"Will retry {params.task_name} (attempt {params.state.retry_count.get(params.task_name, 0)})"
            )

        return params.state


def create_node_function(
    params: NodeFunctionParams,
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
        data = result.get(params.result_field, [])
        count = len(data) if isinstance(data, list) else 1 if data else 0
        return params.message_template.format(count=count)

    def result_processor(state: TravelPlanningState, result: dict[str, Any]) -> None:
        """Process results and update the plan."""
        if state.plan and params.result_field in result:
            setattr(state.plan, params.plan_field, result[params.result_field])

    def node_function(state: TravelPlanningState) -> TravelPlanningState:
        """The actual node execution function."""
        agent = params.agent_class()

        agent_task_params = AgentTaskParams(
            state=state,
            agent=agent,
            task_name=params.task_name,
            complete_stage=params.complete_stage,
            result_formatter=result_formatter,
            result_processor=result_processor,
        )

        return execute_agent_task(agent_task_params)

    # Set function metadata
    node_function.__name__ = params.task_name
    node_function.__doc__ = (
        f"Execute {params.task_name} using {params.agent_class.__name__}."
    )

    return node_function
