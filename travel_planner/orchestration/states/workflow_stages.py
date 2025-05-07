"""
Workflow stage definitions for the travel planner system.

This module defines the various stages of the travel planning workflow
as enum values, which are used throughout the system for state tracking
and workflow progression.
"""

from enum import Enum


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


# Create string constants for the workflow stages
# This allows importing just the strings without the enum
# to avoid circular imports
START = WorkflowStage.START.value
QUERY_ANALYZED = WorkflowStage.QUERY_ANALYZED.value
DESTINATION_RESEARCHED = WorkflowStage.DESTINATION_RESEARCHED.value
FLIGHTS_SEARCHED = WorkflowStage.FLIGHTS_SEARCHED.value
ACCOMMODATION_SEARCHED = WorkflowStage.ACCOMMODATION_SEARCHED.value
TRANSPORTATION_PLANNED = WorkflowStage.TRANSPORTATION_PLANNED.value
ACTIVITIES_PLANNED = WorkflowStage.ACTIVITIES_PLANNED.value
BUDGET_MANAGED = WorkflowStage.BUDGET_MANAGED.value
COMPLETE = WorkflowStage.COMPLETE.value
ERROR = WorkflowStage.ERROR.value
INTERRUPTED = WorkflowStage.INTERRUPTED.value
PARALLEL_SEARCH_COMPLETED = WorkflowStage.PARALLEL_SEARCH_COMPLETED.value
