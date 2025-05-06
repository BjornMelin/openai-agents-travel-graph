"""
Workflow orchestration for the travel planner system.

This module implements the high-level workflow orchestration for the travel planner,
integrating the LangGraph state graph with agent interactions and event handling.
"""

import traceback
from datetime import datetime
from typing import Any

from langgraph.errors import GraphError, InterruptibleError, NodeError, ValidationError
from langgraph.graph import END

from travel_planner.agents.accommodation import AccommodationAgent
from travel_planner.agents.activity_planning import ActivityPlanningAgent
from travel_planner.agents.budget_management import BudgetManagementAgent
from travel_planner.agents.destination_research import DestinationResearchAgent
from travel_planner.agents.flight_search import FlightSearchAgent
from travel_planner.agents.orchestrator import OrchestratorAgent
from travel_planner.agents.transportation import TransportationAgent
from travel_planner.data.models import TravelPlan, TravelQuery, UserPreferences
from travel_planner.orchestration.state_graph import (
    TravelPlanningState,
    create_planning_graph,
)
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


class TravelWorkflow:
    """
    Coordinates the entire travel planning workflow.
    
    This class is responsible for:
    1. Initializing all the agents
    2. Creating and managing the state graph
    3. Processing user queries and generating travel plans
    4. Handling interruptions and state updates
    """
    
    def __init__(self):
        """Initialize the travel planning workflow."""
        self.orchestrator = OrchestratorAgent()
        self.destination_agent = DestinationResearchAgent()
        self.flight_agent = FlightSearchAgent()
        self.accommodation_agent = AccommodationAgent()
        self.transportation_agent = TransportationAgent()
        self.activity_agent = ActivityPlanningAgent()
        self.budget_agent = BudgetManagementAgent()
        
        # Initialize the state graph
        self.graph = create_planning_graph()
        
    async def process_query(self, query: str, preferences: UserPreferences | None = None) -> TravelPlan:
        """
        Process a travel query and generate a complete travel plan.
        
        Args:
            query: User's travel query
            preferences: Optional user preferences
            
        Returns:
            Complete travel plan
        """
        logger.info(f"Processing travel query: {query}")
        
        # Create initial state
        initial_state = TravelPlanningState(
            query=TravelQuery(raw_query=query),
            preferences=preferences or UserPreferences(),
            conversation_history=[{"role": "user", "content": query}]
        )
        
        # Execute the graph with the initial state
        try:
            # Execute the full state graph workflow
            final_state = await self._execute_graph(initial_state)
            return final_state.plan
        except ValidationError as e:
            # Handle validation errors (e.g., invalid state format)
            logger.error(f"Validation error in workflow: {e!s}")
            initial_state.error = f"Validation error: {e!s}"
            initial_state.conversation_history.append({
                "role": "system",
                "content": f"Error: The travel query couldn't be processed due to validation issues. {e!s}"
            })
            # Create a minimal plan with error information
            return self._create_error_plan(e, "validation_error")
        except NodeError as e:
            # Handle specific node execution errors
            logger.error(f"Node error in workflow: {e!s}")
            node_name = getattr(e, "node_name", "unknown")
            logger.error(f"Error occurred in node: {node_name}")
            # Create a partial plan with what we have so far
            initial_state.error = f"Error in {node_name} stage: {e!s}"
            return self._create_error_plan(e, f"node_error_{node_name}")
        except InterruptibleError as e:
            # Handle interruptions (could be user-triggered or system-triggered)
            logger.info(f"Workflow interrupted: {e!s}")
            # Return partial results or resume later
            return self._handle_interruption(initial_state, e)
        except GraphError as e:
            # Handle graph structure or execution errors
            logger.error(f"Graph error in workflow: {e!s}")
            initial_state.error = f"Workflow error: {e!s}"
            return self._create_error_plan(e, "graph_error")
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Unexpected error in travel planning workflow: {e!s}")
            logger.error(traceback.format_exc())
            initial_state.error = f"Unexpected error: {e!s}"
            return self._create_error_plan(e, "unexpected_error")
    
    async def _execute_graph(self, initial_state: TravelPlanningState) -> TravelPlanningState:
        """
        Execute the state graph with the given initial state.
        
        This method uses LangGraph's arun() method to execute the workflow graph
        asynchronously, passing through all the defined nodes in the proper sequence
        based on the conditional edges and transitions defined in the state graph.
        
        Args:
            initial_state: Starting state for the workflow
            
        Returns:
            Final state after workflow completion
        """
        logger.info("Starting workflow graph execution")
        
        # Create event stream from graph execution
        # The arun() method returns an async generator that yields events
        # as the graph progresses through its nodes
        event_stream = await anext(self.graph.arun(initial_state))
        
        # The event stream contains information about the execution, including
        # the final state when the END node is reached
        final_state = None
        async for event in event_stream:
            # Log each node transition and update
            logger.debug(f"Graph event: {event}")
            
            # Check for the final state when END node is reached
            if event['type'] == 'node' and event['node'] == END:
                final_state = event['state']
                break
            
            # Could add additional processing here for monitoring or
            # capturing intermediate states if needed
        
        if final_state is None:
            raise RuntimeError("Workflow did not reach END state")
        
        logger.info("Workflow execution completed successfully")
        return final_state
        
    def _create_error_plan(self, error: Exception, error_type: str) -> TravelPlan:
        """
        Create a minimal travel plan with error information.
        
        Args:
            error: The exception that occurred
            error_type: Type of error for categorization
            
        Returns:
            A minimal travel plan with error information
        """
        error_plan = TravelPlan()
        error_plan.metadata = {
            "error": str(error),
            "error_type": error_type,
            "timestamp": datetime.now().isoformat(),
            "status": "failed"
        }
        error_plan.alerts = [f"Error: {error!s}"]
        return error_plan
    
    def _handle_interruption(self, state: TravelPlanningState, interrupt_error: InterruptibleError) -> TravelPlan:
        """
        Handle workflow interruption by creating a partial travel plan.
        
        Args:
            state: Current workflow state at interruption
            interrupt_error: The interruption error
            
        Returns:
            A partial travel plan with available information
        """
        # Create a plan with whatever information we have so far
        partial_plan = TravelPlan()
        
        # Copy any existing information from the state
        if state.plan:
            # Copy all available attributes from the existing plan
            for attr_name in state.plan.__annotations__:
                if hasattr(state.plan, attr_name):
                    value = getattr(state.plan, attr_name)
                    if value is not None:
                        setattr(partial_plan, attr_name, value)
        
        # Add interruption metadata
        partial_plan.metadata = {
            "interrupted": True,
            "interruption_reason": str(interrupt_error),
            "timestamp": datetime.now().isoformat(),
            "current_stage": state.current_stage,
            "resumable": True,
            "state_id": id(state)  # For referencing this state later
        }
        
        # Add an alert about the interruption
        partial_plan.alerts = [f"Note: This plan is incomplete due to an interruption: {interrupt_error!s}"]
        
        # Store the interrupted state for possible resumption
        self._store_interrupted_state(state)
        
        return partial_plan
    
    def _store_interrupted_state(self, state: TravelPlanningState) -> None:
        """
        Store an interrupted state for later resumption.
        
        In a real implementation, this might persist the state to a database.
        
        Args:
            state: The state to store
        """
        # In a full implementation, this would store the state in a database
        # or other persistent storage for later retrieval
        state_id = id(state)
        logger.info(f"Storing interrupted state with ID: {state_id}")
        # For now, we'll just log it - in a real implementation, we'd store it
        
    async def handle_interrupt(self, current_state: TravelPlanningState, update: dict[str, Any]) -> TravelPlanningState:
        """
        Handle workflow interruptions and state updates.
        
        Args:
            current_state: Current state of the workflow
            update: Update to apply to the state
            
        Returns:
            Updated state
        """
        logger.info(f"Handling interruption at stage: {current_state.current_stage}")
        
        # Add a special marker in the conversation history
        current_state.conversation_history.append({
            "role": "system",
            "content": f"Workflow interrupted at stage: {current_state.current_stage}"
        })
        
        # Apply the update to the current state
        for key, value in update.items():
            if hasattr(current_state, key):
                setattr(current_state, key, value)
                logger.debug(f"Updated {key} in interrupted state")
        
        # Check if we need to cancel or just pause
        if update.get("cancel", False):
            logger.info("Interruption requested workflow cancellation")
            current_state.error = "Workflow cancelled by user request"
        else:
            logger.info("Interruption is a pause, workflow can be resumed")
            
        # Store the state for possible resumption
        self._store_interrupted_state(current_state)
            
        return current_state
        
    async def resume_workflow(self, state: TravelPlanningState) -> TravelPlanningState:
        """
        Resume an interrupted workflow from the current state.
        
        Args:
            state: Current state to resume from
            
        Returns:
            Final state after workflow completion
        """
        logger.info(f"Resuming workflow from stage: {state.current_stage}")
        
        # Add a note about resumption to conversation history
        state.conversation_history.append({
            "role": "system",
            "content": f"Resuming workflow from stage: {state.current_stage}"
        })
        
        try:
            # Determine where to resume from based on current stage
            if state.current_stage == "query_analyzed":
                # Skip query analysis and start from next stage
                logger.info("Resuming after query analysis")
                # In a full implementation, we'd modify the graph entry point
            
            # Execute the graph with the current state
            resumed_state = await self._execute_graph(state)
            logger.info("Successfully resumed and completed workflow")
            return resumed_state
            
        except Exception as e:
            logger.error(f"Error resuming workflow: {e!s}")
            logger.error(traceback.format_exc())
            state.error = f"Resume error: {e!s}"
            return state