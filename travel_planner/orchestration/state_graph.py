"""
State graph implementation for the travel planner system.

This module implements a LangGraph state graph that orchestrates the flow of
the travel planning process, connecting various specialized agents together
into a coherent workflow with support for parallel execution, interruptions,
and error recovery.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from langgraph.graph import StateGraph
from langgraph.graph.branches.parallel import ParallelBranch
from langgraph.graph.branches.human import human_in_the_loop, HumanHandler
from pydantic import BaseModel, Field

from travel_planner.agents.accommodation import AccommodationAgent
from travel_planner.agents.activity_planning import ActivityPlanningAgent
from travel_planner.agents.budget_management import BudgetManagementAgent
from travel_planner.agents.destination_research import DestinationResearchAgent
from travel_planner.agents.flight_search import FlightSearchAgent
from travel_planner.agents.orchestrator import OrchestratorAgent
from travel_planner.agents.transportation import TransportationAgent
from travel_planner.data.models import TravelPlan, TravelQuery, UserPreferences


class WorkflowStage(str, Enum):
    """Enum representing stages in the travel planning workflow."""
    START = "start"
    QUERY_ANALYZED = "query_analyzed"
    DESTINATION_RESEARCHED = "destination_researched"
    FLIGHTS_SEARCHED = "flights_searched"
    ACCOMMODATION_SEARCHED = "accommodation_searched"
    TRANSPORTATION_PLANNED = "transportation_planned"
    ACTIVITIES_PLANNED = "activities_planned"
    BUDGET_MANAGED = "budget_managed"
    COMPLETE = "complete"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    PARALLEL_SEARCH_COMPLETED = "parallel_search_completed"


class TravelPlanningState(BaseModel):
    """
    State representation for the travel planning workflow.
    
    This state is passed between nodes in the LangGraph state graph and
    maintains the full context of the planning process. It includes
    progress tracking, interruption handling, and checkpointing capabilities.
    """
    # Core travel data
    query: TravelQuery | None = None
    preferences: UserPreferences | None = None
    plan: TravelPlan | None = None
    
    # Conversation and history tracking
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    
    # Workflow state management
    current_stage: WorkflowStage = WorkflowStage.START
    previous_stage: WorkflowStage | None = None
    start_time: datetime | None = None
    last_update_time: datetime | None = None
    stage_times: dict[str, datetime] = Field(default_factory=dict)
    
    # Error handling and recovery
    error: str | None = None
    error_count: int = 0
    retry_count: dict[str, int] = Field(default_factory=dict)
    
    # Interruption handling
    interrupted: bool = False
    interruption_reason: str | None = None
    checkpoint_id: str | None = None
    
    # Progress tracking
    progress: float = 0.0  # 0.0 to 1.0
    stage_progress: dict[str, float] = Field(default_factory=dict)
    
    # Parallel execution tracking
    parallel_tasks: list[str] = Field(default_factory=list)
    completed_tasks: list[str] = Field(default_factory=list)
    task_results: dict[str, dict[str, Any]] = Field(default_factory=dict)
    
    # Human feedback and guidance
    human_feedback: list[dict[str, Any]] = Field(default_factory=list)
    guidance_requested: bool = False
    
    def __init__(self, **data):
        """Initialize the travel planning state with timing information."""
        super().__init__(**data)
        current_time = datetime.now()
        if self.start_time is None:
            self.start_time = current_time
        self.last_update_time = current_time
        self.stage_times[str(self.current_stage)] = current_time
    
    def update_stage(self, new_stage: WorkflowStage) -> None:
        """
        Update the current stage and related timing information.
        
        Args:
            new_stage: The new workflow stage
        """
        self.previous_stage = self.current_stage
        self.current_stage = new_stage
        current_time = datetime.now()
        self.last_update_time = current_time
        self.stage_times[str(new_stage)] = current_time
        
        # Update progress based on stage
        stage_weights = {
            WorkflowStage.START: 0.0,
            WorkflowStage.QUERY_ANALYZED: 0.1,
            WorkflowStage.DESTINATION_RESEARCHED: 0.2,
            WorkflowStage.FLIGHTS_SEARCHED: 0.4,
            WorkflowStage.ACCOMMODATION_SEARCHED: 0.5,
            WorkflowStage.TRANSPORTATION_PLANNED: 0.6,
            WorkflowStage.ACTIVITIES_PLANNED: 0.8,
            WorkflowStage.BUDGET_MANAGED: 0.9,
            WorkflowStage.COMPLETE: 1.0,
            WorkflowStage.PARALLEL_SEARCH_COMPLETED: 0.6,  # Equivalent to completing several stages
        }
        self.progress = stage_weights.get(new_stage, self.progress)
    
    def mark_interrupted(self, reason: str) -> None:
        """
        Mark the state as interrupted with a specific reason.
        
        Args:
            reason: The reason for the interruption
        """
        self.interrupted = True
        self.interruption_reason = reason
        previous_stage = self.current_stage
        self.update_stage(WorkflowStage.INTERRUPTED)
        self.previous_stage = previous_stage  # Preserve the actual stage we were interrupted at
        self.checkpoint_id = f"checkpoint_{uuid.uuid4().hex}"
        
    def mark_error(self, error_message: str) -> None:
        """
        Mark the state with an error.
        
        Args:
            error_message: Description of the error
        """
        self.error = error_message
        self.error_count += 1
        previous_stage = self.current_stage
        self.update_stage(WorkflowStage.ERROR)
        self.previous_stage = previous_stage  # Preserve the stage where the error occurred
        
    def add_human_feedback(self, feedback: dict[str, Any]) -> None:
        """
        Add human feedback to the state.
        
        Args:
            feedback: Dictionary with feedback information
        """
        feedback["timestamp"] = datetime.now().isoformat()
        self.human_feedback.append(feedback)
        
        # Add feedback to conversation history for context
        self.conversation_history.append({
            "role": "human",
            "content": feedback.get("content", ""),
            "feedback": True
        })
        
    def add_task_result(self, task_name: str, result: dict[str, Any], 
                        error: str | None = None) -> None:
        """
        Add a task result to the state.
        
        Args:
            task_name: Name of the completed task
            result: Result data from the task
            error: Optional error information
        """
        self.task_results[task_name] = {
            "result": result,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        if task_name not in self.completed_tasks:
            self.completed_tasks.append(task_name)
    
    def should_retry(self, stage: str, max_retries: int = 3) -> bool:
        """
        Determine if a failed stage should be retried.
        
        Args:
            stage: The stage that failed
            max_retries: Maximum number of retries allowed
            
        Returns:
            True if a retry should be attempted, False otherwise
        """
        current_retries = self.retry_count.get(stage, 0)
        if current_retries < max_retries:
            self.retry_count[stage] = current_retries + 1
            return True
        return False
    
    def create_checkpoint(self) -> dict[str, Any]:
        """
        Create a serializable checkpoint of the current state.
        
        Returns:
            Dictionary with serialized state data
        """
        checkpoint_data = self.model_dump()
        
        # Generate a unique checkpoint ID if not already set
        if not self.checkpoint_id:
            self.checkpoint_id = f"checkpoint_{uuid.uuid4().hex}"
            
        checkpoint_data["checkpoint_id"] = self.checkpoint_id
        checkpoint_data["checkpoint_time"] = datetime.now().isoformat()
        
        # Convert datetime objects to ISO format strings for serialization
        if self.start_time:
            checkpoint_data["start_time"] = self.start_time.isoformat()
        
        if self.last_update_time:
            checkpoint_data["last_update_time"] = self.last_update_time.isoformat()
        
        # Convert stage_times datetime objects to strings
        if self.stage_times:
            checkpoint_data["stage_times"] = {
                k: v.isoformat() if v else None 
                for k, v in self.stage_times.items()
            }
        
        return checkpoint_data
        
    def from_checkpoint(self, checkpoint_data: dict[str, Any]) -> None:
        """
        Load state from a checkpoint.
        
        Args:
            checkpoint_data: Checkpoint data to load
        """
        # Handle special fields first
        
        # Convert stage strings to enums
        current_stage_str = checkpoint_data.get("current_stage", "start")
        checkpoint_data["current_stage"] = WorkflowStage(current_stage_str) if current_stage_str else WorkflowStage.START
        
        previous_stage_str = checkpoint_data.get("previous_stage")
        if previous_stage_str:
            checkpoint_data["previous_stage"] = WorkflowStage(previous_stage_str)
        
        # Convert ISO timestamp strings back to datetime objects
        start_time_str = checkpoint_data.get("start_time")
        if start_time_str:
            self.start_time = datetime.fromisoformat(start_time_str)
        
        last_update_time_str = checkpoint_data.get("last_update_time")
        if last_update_time_str:
            self.last_update_time = datetime.fromisoformat(last_update_time_str)
        
        # Convert stage_times strings back to datetime objects
        stage_times_data = checkpoint_data.get("stage_times", {})
        for stage, time_str in stage_times_data.items():
            if time_str:
                self.stage_times[stage] = datetime.fromisoformat(time_str)
                
        # Now update all the basic attributes
        for key, value in checkpoint_data.items():
            if key not in ["start_time", "last_update_time", "stage_times"] and hasattr(self, key):
                setattr(self, key, value)
    

def create_parallel_search_branch() -> ParallelBranch:
    """
    Create a parallel branch for searching flights, accommodation, and transportation.
    
    Returns:
        ParallelBranch for parallel search operations
    """
    branch = ParallelBranch("parallel_search")
    branch.add_node("flight_search", flight_search)
    branch.add_node("accommodation_search", accommodation_search)
    branch.add_node("transportation_planning", transportation_planning)
    
    return branch


def combine_search_results(state: TravelPlanningState) -> TravelPlanningState:
    """
    Combine the results from the parallel search branch.
    
    Args:
        state: Current travel planning state
        
    Returns:
        Updated state with combined results from parallel search
    """
    # In a real implementation, you might need to resolve conflicts or
    # ensure data consistency across parallel results
    
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


def create_planning_graph() -> StateGraph:
    """
    Create an optimized state graph for travel planning with parallel execution.
    
    This graph implements the travel planning workflow with parallel branches
    for independent tasks, error handling, and support for human-in-the-loop
    interruptions.
    
    Returns:
        StateGraph instance that orchestrates the travel planning workflow
    """
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
            "flight_search": "parallel_search"
        }
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
    for node in ["analyze_query", "research_destination", "parallel_search", 
                "combine_search_results", "plan_activities", "manage_budget", 
                "generate_final_plan"]:
        workflow.add_conditional_edges(
            node,
            has_error,
            {
                "true": "handle_error",
                "false": node  # Continue current node if no error
            }
        )
    
    # After error handling, either exit or route back into workflow
    workflow.add_conditional_edges(
        "handle_error",
        error_recoverable,
        {
            "true": recover_to_stage,       # Route to appropriate recovery point
            "false": "END"                 # End if unrecoverable
        }
    )
    
    # Set up human-in-the-loop capability
    human_branch = human_in_the_loop()
    workflow.add_branch("human_in_the_loop", human_branch)
    
    # Add human intervention capabilities to critical nodes
    for node in ["analyze_query", "research_destination", "parallel_search", 
                "plan_activities", "manage_budget", "generate_final_plan"]:
        workflow.add_conditional_edges(
            node,
            needs_human_intervention,
            {
                "true": "human_in_the_loop",
                "false": node  # Continue with node if no intervention needed
            }
        )
    
    # After human intervention, route back to appropriate point in workflow
    workflow.add_edge("human_in_the_loop", continue_after_intervention)
    
    # Compile the graph with optimizations
    return workflow.compile()


def query_research_needed(state: TravelPlanningState) -> str:
    """
    Determine if destination research is needed based on query analysis.
    
    Args:
        state: Current travel planning state
        
    Returns:
        Next stage in the workflow
    """
    # This is a router function that helps determine the next stage
    # based on the current state
    if not state.query or not state.query.destination:
        return "research_destination"
    return "flight_search"


def query_analysis(state: TravelPlanningState) -> TravelPlanningState:
    """
    Analyze the user query to understand requirements and preferences.
    
    Args:
        state: Current travel planning state
        
    Returns:
        Updated travel planning state
    """
    orchestrator = OrchestratorAgent()
    result = orchestrator.invoke(state)
    
    # Update state with query analysis results
    state.query = result.get("query", state.query)
    state.preferences = result.get("preferences", state.preferences)
    state.update_stage(WorkflowStage.QUERY_ANALYZED)
    
    # Add the result to conversation history for context
    state.conversation_history.append({
        "role": "system",
        "content": f"Query analyzed: {state.query.destination if state.query and state.query.destination else 'Destination research needed'}"
    })
    
    return state


def destination_research(state: TravelPlanningState) -> TravelPlanningState:
    """
    Research destination information based on user query.
    
    Args:
        state: Current travel planning state
        
    Returns:
        Updated travel planning state with destination information
    """
    try:
        research_agent = DestinationResearchAgent()
        result = research_agent.invoke(state)
        
        # Update state with destination research results
        if state.plan is None:
            state.plan = TravelPlan()
        
        destination_details = result.get("destination_details", {})
        state.plan.destination = destination_details
        state.update_stage(WorkflowStage.DESTINATION_RESEARCHED)
        
        # Add destination information to conversation history
        destination_name = destination_details.get("name", "Unknown destination")
        state.conversation_history.append({
            "role": "system",
            "content": f"Destination researched: {destination_name}"
        })
        
        # Track the task result
        state.add_task_result("destination_research", result)
        
        return state
    except Exception as e:
        # Handle errors during destination research
        state.mark_error(f"Error during destination research: {str(e)}")
        if state.should_retry("destination_research"):
            # Retry logic would go here in a full implementation
            pass
        return state


def flight_search(state: TravelPlanningState) -> TravelPlanningState:
    """
    Search for flight options based on travel requirements.
    
    Args:
        state: Current travel planning state
        
    Returns:
        Updated travel planning state with flight options
    """
    try:
        flight_agent = FlightSearchAgent()
        result = flight_agent.invoke(state)
        
        # Update state with flight search results
        if state.plan is None:
            state.plan = TravelPlan()
        
        flight_options = result.get("flight_options", [])
        state.plan.flights = flight_options
        state.update_stage(WorkflowStage.FLIGHTS_SEARCHED)
        
        # Add flight info to conversation history
        num_options = len(flight_options)
        state.conversation_history.append({
            "role": "system",
            "content": f"Found {num_options} flight options"
        })
        
        # Track the task result
        state.add_task_result("flight_search", result)
        
        return state
    except Exception as e:
        # Handle errors during flight search
        state.mark_error(f"Error during flight search: {str(e)}")
        if state.should_retry("flight_search"):
            # Retry logic would go here in a full implementation
            pass
        return state


def accommodation_search(state: TravelPlanningState) -> TravelPlanningState:
    """
    Search for accommodation options based on travel requirements.
    
    Args:
        state: Current travel planning state
        
    Returns:
        Updated travel planning state with accommodation options
    """
    try:
        accommodation_agent = AccommodationAgent()
        result = accommodation_agent.invoke(state)
        
        # Update state with accommodation search results
        if state.plan is None:
            state.plan = TravelPlan()
        
        accommodations = result.get("accommodations", [])
        state.plan.accommodation = accommodations
        state.update_stage(WorkflowStage.ACCOMMODATION_SEARCHED)
        
        # Add accommodation info to conversation history
        num_options = len(accommodations)
        state.conversation_history.append({
            "role": "system",
            "content": f"Found {num_options} accommodation options"
        })
        
        # Track the task result
        state.add_task_result("accommodation_search", result)
        
        return state
    except Exception as e:
        # Handle errors during accommodation search
        state.mark_error(f"Error during accommodation search: {str(e)}")
        if state.should_retry("accommodation_search"):
            # Retry logic would go here in a full implementation
            pass
        return state


def transportation_planning(state: TravelPlanningState) -> TravelPlanningState:
    """
    Plan local transportation for the trip.
    
    Args:
        state: Current travel planning state
        
    Returns:
        Updated travel planning state with transportation plans
    """
    try:
        transportation_agent = TransportationAgent()
        result = transportation_agent.invoke(state)
        
        # Update state with transportation planning results
        if state.plan is None:
            state.plan = TravelPlan()
        
        transportation_options = result.get("transportation_options", {})
        state.plan.transportation = transportation_options
        state.update_stage(WorkflowStage.TRANSPORTATION_PLANNED)
        
        # Add transportation info to conversation history
        state.conversation_history.append({
            "role": "system",
            "content": f"Local transportation planned with {len(transportation_options)} options"
        })
        
        # Track the task result
        state.add_task_result("transportation_planning", result)
        
        return state
    except Exception as e:
        # Handle errors during transportation planning
        state.mark_error(f"Error during transportation planning: {str(e)}")
        if state.should_retry("transportation_planning"):
            # Retry logic would go here in a full implementation
            pass
        return state


def activity_planning(state: TravelPlanningState) -> TravelPlanningState:
    """
    Plan activities and create daily itineraries.
    
    Args:
        state: Current travel planning state
        
    Returns:
        Updated travel planning state with activities and itineraries
    """
    try:
        activity_agent = ActivityPlanningAgent()
        result = activity_agent.invoke(state)
        
        # Update state with activity planning results
        if state.plan is None:
            state.plan = TravelPlan()
        
        daily_itineraries = result.get("daily_itineraries", {})
        state.plan.activities = daily_itineraries
        state.update_stage(WorkflowStage.ACTIVITIES_PLANNED)
        
        # Add activities info to conversation history
        num_days = len(daily_itineraries)
        state.conversation_history.append({
            "role": "system",
            "content": f"Activities planned for {num_days} days"
        })
        
        # Track the task result
        state.add_task_result("activity_planning", result)
        
        return state
    except Exception as e:
        # Handle errors during activity planning
        state.mark_error(f"Error during activity planning: {str(e)}")
        if state.should_retry("activity_planning"):
            # Retry logic would go here in a full implementation
            pass
        return state


def budget_management(state: TravelPlanningState) -> TravelPlanningState:
    """
    Manage and optimize the budget for the trip.
    
    Args:
        state: Current travel planning state
        
    Returns:
        Updated travel planning state with budget information
    """
    try:
        budget_agent = BudgetManagementAgent()
        result = budget_agent.invoke(state)
        
        # Update state with budget management results
        if state.plan is None:
            state.plan = TravelPlan()
        
        budget_report = result.get("report", {})
        state.plan.budget = budget_report
        state.update_stage(WorkflowStage.BUDGET_MANAGED)
        
        # Add budget info to conversation history
        total_budget = budget_report.get("total_budget", "Unknown")
        state.conversation_history.append({
            "role": "system",
            "content": f"Budget plan created with total: {total_budget}"
        })
        
        # Track the task result
        state.add_task_result("budget_management", result)
        
        return state
    except Exception as e:
        # Handle errors during budget management
        state.mark_error(f"Error during budget management: {str(e)}")
        if state.should_retry("budget_management"):
            # Retry logic would go here in a full implementation
            pass
        return state


def generate_final_plan(state: TravelPlanningState) -> TravelPlanningState:
    """
    Generate the final travel plan with all components.
    
    Args:
        state: Current travel planning state
        
    Returns:
        Updated travel planning state with complete plan
    """
    from travel_planner.orchestration.checkpoint import save_state_checkpoint
    
    try:
        orchestrator = OrchestratorAgent()
        result = orchestrator.invoke(state)
        
        # Update state with final plan
        state.plan = result.get("final_plan", state.plan)
        state.update_stage(WorkflowStage.COMPLETE)
        
        # Create and save a checkpoint of the final state
        checkpoint_id = save_state_checkpoint(state)
        state.checkpoint_id = checkpoint_id
        
        # Add completion info to conversation history
        state.conversation_history.append({
            "role": "system",
            "content": f"Travel planning completed successfully. Final plan saved as checkpoint {checkpoint_id}"
        })
        
        # Record task result
        state.add_task_result("generate_final_plan", result)
        
        return state
    except Exception as e:
        # Handle errors during final plan generation
        state.mark_error(f"Error during final plan generation: {str(e)}")
        if state.should_retry("generate_final_plan"):
            # Retry logic would go here in a full implementation
            pass
        return state


def has_error(state: TravelPlanningState) -> str:
    """
    Check if the state has an error.
    
    Args:
        state: Current travel planning state
        
    Returns:
        "true" if the state has an error, "false" otherwise
    """
    if state.error or state.current_stage == WorkflowStage.ERROR:
        return "true"
    return "false"


def error_recoverable(state: TravelPlanningState) -> str:
    """
    Determine if the error in the state is recoverable.
    
    Args:
        state: Current travel planning state
        
    Returns:
        "true" if the error is recoverable, "false" otherwise
    """
    # In a real implementation, we'd have more complex logic here
    # For now, we'll consider an error recoverable if we haven't
    # exceeded the retry limit for the stage
    
    # Get the stage that had the error
    error_stage = str(state.previous_stage) if state.previous_stage else "unknown"
    
    # Check if we can retry this stage
    if state.should_retry(error_stage):
        return "true"
    
    # If error count is too high, not recoverable
    if state.error_count > 3:
        return "false"
        
    return "true"


def recover_to_stage(state: TravelPlanningState) -> str:
    """
    Determine which stage to recover to after an error.
    
    Args:
        state: Current travel planning state
        
    Returns:
        Name of the stage to recover to
    """
    # In a real implementation, this would have more complex logic
    # For now, we'll attempt to go back to the stage that had the error
    
    # Get the stage that had the error
    error_stage = str(state.previous_stage) if state.previous_stage else "analyze_query"
    
    # Clear the error state
    state.error = None
    state.update_stage(state.previous_stage if state.previous_stage else WorkflowStage.START)
    
    # Add a note to the conversation history
    state.conversation_history.append({
        "role": "system", 
        "content": f"Recovering from error, retrying {error_stage}"
    })
    
    return error_stage


def handle_error(state: TravelPlanningState) -> TravelPlanningState:
    """
    Handle an error in the workflow.
    
    Args:
        state: Current travel planning state
        
    Returns:
        Updated state with error handling
    """
    from travel_planner.orchestration.checkpoint import save_state_checkpoint
    
    # Get the error message
    error_message = state.error or "Unknown error"
    
    # Add error to the conversation history
    state.conversation_history.append({
        "role": "system",
        "content": f"Error occurred: {error_message}"
    })
    
    # Create checkpoint for potential recovery
    checkpoint_id = save_state_checkpoint(state)
    
    # Update checkpoint ID in state
    state.checkpoint_id = checkpoint_id
    
    # In a real implementation, we might send notifications or
    # take other actions to handle the error
    
    return state


def needs_human_intervention(state: TravelPlanningState) -> str:
    """
    Determine if human intervention is needed.
    
    Args:
        state: Current travel planning state
        
    Returns:
        "true" if human intervention is needed, "false" otherwise
    """
    # Check if the state has requested guidance
    if state.guidance_requested:
        return "true"
    
    # Check if we have too many errors (might need human help)
    if state.error_count >= 2:
        return "true"
    
    # Check if we have an interrupted state
    if state.interrupted:
        return "true"
    
    return "false"


def handle_interruption(state: TravelPlanningState) -> TravelPlanningState:
    """
    Handle a workflow interruption.
    
    Args:
        state: Current travel planning state
        
    Returns:
        Updated state with interruption handling
    """
    from travel_planner.orchestration.checkpoint import save_state_checkpoint
    
    # Mark the state as interrupted if not already
    if not state.interrupted:
        state.mark_interrupted("User requested interruption")
    
    # Create a checkpoint and persist it
    checkpoint_id = save_state_checkpoint(state)
    
    # Update checkpoint ID in state
    state.checkpoint_id = checkpoint_id
    
    # Add note to conversation history
    state.conversation_history.append({
        "role": "system",
        "content": f"Workflow interrupted: {state.interruption_reason}. Checkpoint ID: {checkpoint_id}"
    })
    
    return state


def continue_after_intervention(state: TravelPlanningState) -> str:
    """
    Determine where to continue after human intervention.
    
    Args:
        state: Current travel planning state
        
    Returns:
        Name of the node to continue at
    """
    # Clear intervention flags
    state.guidance_requested = False
    
    # If we were interrupted, stay interrupted until explicitly resumed
    if state.interrupted:
        return "END"
    
    # Get the stage we should return to (previous stage or start)
    return_stage = str(state.previous_stage) if state.previous_stage else "analyze_query"
    
    # Add note to conversation history
    state.conversation_history.append({
        "role": "system",
        "content": f"Continuing workflow at {return_stage} after human intervention"
    })
    
    return return_stage


def plan_complete(state: TravelPlanningState) -> bool:
    """
    Check if the travel plan is complete.
    
    Args:
        state: Current travel planning state
        
    Returns:
        True if the plan is complete, False otherwise
    """
    # Check if the workflow has been marked as complete
    if state.current_stage == WorkflowStage.COMPLETE:
        return True
        
    # Check if all required components of the travel plan are present
    if not state.plan:
        return False
    
    required_fields = [
        state.plan.destination,
        state.plan.flights,
        state.plan.accommodation,
        state.plan.activities,
        state.plan.transportation,
        state.plan.budget
    ]
    
    # Check if all required fields are present
    fields_complete = all(required_fields)
    
    # If all fields are complete but state isn't marked complete, update it
    if fields_complete and state.current_stage != WorkflowStage.COMPLETE:
        state.update_stage(WorkflowStage.COMPLETE)
        
    return fields_complete