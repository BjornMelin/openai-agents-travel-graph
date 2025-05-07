"""
Orchestrator agent for the travel planner system.

This module implements the orchestrator agent that coordinates the activities
of all specialized agents, manages the travel planning workflow, and ensures
proper handoffs between different components of the system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from travel_planner.agents.base import AgentConfig, AgentContext, BaseAgent
from travel_planner.utils import (
    AgentExecutionError,
    AgentLogger,
    handle_errors,
    safe_serialize,
)


class PlanningStage(str, Enum):
    """Stages of the travel planning process."""

    INITIAL = "initial"
    DESTINATION_RESEARCH = "destination_research"
    FLIGHT_SEARCH = "flight_search"
    ACCOMMODATION_SEARCH = "accommodation_search"
    TRANSPORTATION_PLANNING = "transportation_planning"
    ACTIVITY_PLANNING = "activity_planning"
    BUDGET_MANAGEMENT = "budget_management"
    FINAL_ITINERARY = "final_itinerary"


@dataclass
class TravelRequirements:
    """User's travel requirements."""

    destination: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    budget: float | None = None
    currency: str = "USD"
    num_travelers: int = 1
    accommodation_preferences: list[str] = field(default_factory=list)
    transportation_preferences: list[str] = field(default_factory=list)
    activity_preferences: list[str] = field(default_factory=list)
    dietary_restrictions: list[str] = field(default_factory=list)
    accessibility_needs: list[str] = field(default_factory=list)
    additional_notes: str | None = None


@dataclass
class OrchestratorContext(AgentContext):
    """Context for the orchestrator agent."""

    session_id: str
    planning_stage: PlanningStage = PlanningStage.INITIAL
    travel_requirements: TravelRequirements = field(default_factory=TravelRequirements)
    destination_details: dict[str, Any] = field(default_factory=dict)
    flight_options: list[dict[str, Any]] = field(default_factory=list)
    accommodation_options: list[dict[str, Any]] = field(default_factory=list)
    transportation_options: list[dict[str, Any]] = field(default_factory=list)
    activity_options: list[dict[str, Any]] = field(default_factory=list)
    budget_allocation: dict[str, float] = field(default_factory=dict)
    selected_options: dict[str, Any] = field(default_factory=dict)
    user_feedback: dict[str, Any] = field(default_factory=dict)
    final_itinerary: dict[str, Any] = field(default_factory=dict)
    conversation_history: list[dict[str, Any]] = field(default_factory=list)


class OrchestratorAgent(BaseAgent[OrchestratorContext]):
    """
    Orchestrator agent that coordinates the overall travel planning process.

    This agent is responsible for:
    1. Managing the travel planning workflow
    2. Coordinating communication between specialized agents
    3. Maintaining the master context of the planning session
    4. Ensuring all user requirements are met
    5. Handling exceptions and fallbacks from other agents
    """

    def __init__(self, config: AgentConfig | None = None):
        """
        Initialize the orchestrator agent.

        Args:
            config: Configuration for the agent (optional)
        """
        default_config = AgentConfig(
            name="Travel Orchestrator",
            instructions=(
                "You are an AI travel planning orchestrator. Your job is to guide the overall "
                "travel planning process by coordinating specialized agents for destination research, "
                "flight search, accommodation booking, transportation arrangements, activity planning, "
                "and budget management. Maintain a coherent plan that satisfies all user requirements "
                "while optimizing for budget, convenience, and user preferences."
            ),
        )
        super().__init__(config or default_config, OrchestratorContext)
        self.logger = AgentLogger(self.name)

    async def run(
        self,
        input_data: str | list[dict[str, Any]],
        context: OrchestratorContext | None = None,
    ) -> dict[str, Any]:
        """
        Run the orchestrator agent with the provided input and context.

        Args:
            input_data: User input or conversation history
            context: Optional orchestrator context

        Returns:
            Updated orchestrator context and response
        """
        self.logger.info(
            f"Running orchestrator agent with input: {input_data if isinstance(input_data, str) else '...'}"
        )

        if context is None:
            # Create a new context with a generated session ID
            from travel_planner.utils.helpers import generate_session_id

            context = OrchestratorContext(session_id=generate_session_id())

        # Add user input to conversation history
        if isinstance(input_data, str):
            context.conversation_history.append({"role": "user", "content": input_data})

        # Process the input based on the current planning stage
        try:
            response = await self.process(input_data, context)

            # Add agent response to conversation history
            if isinstance(response, dict) and "content" in response:
                context.conversation_history.append(
                    {"role": "assistant", "content": response["content"]}
                )

            return {
                "context": context,
                "response": response,
            }
        except Exception as e:
            error_msg = f"Error in orchestrator agent: {e!s}"
            self.logger.error(error_msg)
            raise AgentExecutionError(error_msg, self.name, original_error=e) from e

    @handle_errors(error_cls=AgentExecutionError)
    async def process(
        self, input_data: str | list[dict[str, Any]], context: OrchestratorContext
    ) -> dict[str, Any]:
        """
        Process the input based on the current planning stage.

        Args:
            input_data: User input or conversation history
            context: Orchestrator context

        Returns:
            Agent response
        """
        self.logger.info(f"Processing input in stage: {context.planning_stage}")

        # Prepare messages for the model
        messages = self._prepare_messages(input_data)

        # Extract or update requirements from the user input
        if context.planning_stage == PlanningStage.INITIAL:
            updated_requirements = await self._extract_requirements(
                input_data, context.travel_requirements
            )
            context.travel_requirements = updated_requirements
            context.planning_stage = PlanningStage.DESTINATION_RESEARCH

        # Determine the next planning stage based on the current stage and context
        messages.append(
            {
                "role": "system",
                "content": (
                    f"Current planning stage: {context.planning_stage}. "
                    f"Travel requirements: {safe_serialize(context.travel_requirements)}. "
                    f"Use the context information to determine the next steps in the planning process."
                ),
            }
        )

        # Call the OpenAI API to get the orchestrator's response
        response = await self._call_model(messages)

        # Update the planning stage based on the response
        await self._update_planning_stage(context, response)

        return response

    async def _extract_requirements(
        self,
        input_data: str | list[dict[str, Any]],
        current_requirements: TravelRequirements,
    ) -> TravelRequirements:
        """
        Extract travel requirements from user input.

        Args:
            input_data: User input or conversation history
            current_requirements: Current travel requirements

        Returns:
            Updated travel requirements
        """
        self.logger.info("Extracting travel requirements")

        # Prepare a specific prompt for requirement extraction
        extraction_prompt = (
            "Extract the travel requirements from the user's input. Include destination, dates, "
            "budget, number of travelers, and any preferences or restrictions. If information is "
            "missing, keep the current values. Format the output as a structured JSON object."
        )

        user_input = (
            input_data
            if isinstance(input_data, str)
            else self._get_latest_user_input(input_data)
        )

        messages = [
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": user_input},
        ]

        # Add current requirements as context if they exist
        if current_requirements and any(vars(current_requirements).values()):
            messages.append(
                {
                    "role": "system",
                    "content": f"Current requirements: {safe_serialize(current_requirements)}",
                }
            )

        response = await self._call_model(messages)

        # Try to extract structured data from the response
        try:
            # Implementation depends on the actual API response format
            # This is a simplified approach
            response.get("content", "")

            # If the model didn't return valid JSON, we'll use the existing requirements
            return current_requirements

        except Exception as e:
            self.logger.error(f"Error extracting requirements: {e!s}")
            return current_requirements

    async def _update_planning_stage(
        self, context: OrchestratorContext, response: dict[str, Any]
    ) -> None:
        """
        Update the planning stage based on the agent's response.

        Args:
            context: Orchestrator context
            response: Agent response
        """
        # Simple stage progression logic - in a real implementation, would be more sophisticated
        current_stage = context.planning_stage

        # Map of stages and conditions to move to the next stage
        stage_progression = {
            PlanningStage.INITIAL: PlanningStage.DESTINATION_RESEARCH,
            PlanningStage.DESTINATION_RESEARCH: PlanningStage.FLIGHT_SEARCH,
            PlanningStage.FLIGHT_SEARCH: PlanningStage.ACCOMMODATION_SEARCH,
            PlanningStage.ACCOMMODATION_SEARCH: PlanningStage.TRANSPORTATION_PLANNING,
            PlanningStage.TRANSPORTATION_PLANNING: PlanningStage.ACTIVITY_PLANNING,
            PlanningStage.ACTIVITY_PLANNING: PlanningStage.BUDGET_MANAGEMENT,
            PlanningStage.BUDGET_MANAGEMENT: PlanningStage.FINAL_ITINERARY,
        }

        # For now, simply progress to the next stage based on the map
        if current_stage in stage_progression:
            context.planning_stage = stage_progression[current_stage]
            self.logger.info(
                f"Updating planning stage from {current_stage} to {context.planning_stage}"
            )

    def _get_latest_user_input(self, messages: list[dict[str, Any]]) -> str:
        """
        Extract the latest user input from a list of messages.

        Args:
            messages: List of message dictionaries

        Returns:
            Latest user input text
        """
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    async def _call_model(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Call the OpenAI API with the given messages.

        Args:
            messages: List of message dictionaries

        Returns:
            Model response
        """
        self.logger.info(f"Calling model with {len(messages)} messages")

        # Log inputs for debugging (sensitive data would be handled appropriately in production)
        self.logger.log_llm_input(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
        )

        try:
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            # Log the response
            self.logger.log_llm_output(model=self.config.model, response=response)

            # Extract the content from the response
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                return {"content": content}

            return {"content": "No response generated."}

        except Exception as e:
            self.logger.error(f"Error calling model: {e!s}")
            raise
