"""
Main entry point for the Travel Planner application.

This module serves as the application's entry point, initializing the necessary
components and providing a CLI interface for interacting with the travel planning system.
"""

import argparse
import asyncio
import json
import os
import sys
import traceback
from datetime import date, datetime

from travel_planner.agents.accommodation import AccommodationAgent
from travel_planner.agents.activity_planning import ActivityPlanningAgent
from travel_planner.agents.budget_management import BudgetManagementAgent
from travel_planner.agents.destination_research import DestinationResearchAgent
from travel_planner.agents.flight_search import FlightSearchAgent
from travel_planner.agents.research_tools import DestinationResearchTools
from travel_planner.agents.transportation import TransportationAgent
from travel_planner.config import initialize_config
from travel_planner.data.models import TravelPlan, TravelQuery
from travel_planner.data.supabase import SupabaseClient
from travel_planner.orchestration.state_graph import TravelPlanningState
from travel_planner.orchestration.workflow import TravelWorkflow
from travel_planner.utils.logging import get_logger, setup_logging

# Initialize logger
logger = get_logger(__name__)


def setup_argparse() -> argparse.ArgumentParser:
    """
    Set up the argument parser for the CLI.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="AI Travel Planning System powered by OpenAI Agents SDK and LangGraph"
    )

    # System configuration arguments
    system_group = parser.add_argument_group("System Configuration")
    system_group.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    system_group.add_argument(
        "--log-file",
        type=str,
        help="Path to write log file (optional)",
    )
    system_group.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file",
    )
    system_group.add_argument(
        "--headless",
        action="store_true",
        help="Run browser automation in headless mode",
    )
    system_group.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching for browser automation and API calls",
    )

    # Query mode arguments
    query_group = parser.add_argument_group("Query Mode")
    query_group.add_argument(
        "--query",
        type=str,
        help="Initial travel query to start planning",
    )
    query_group.add_argument(
        "--origin",
        type=str,
        help="Origin location (e.g., city or airport code)",
    )
    query_group.add_argument(
        "--destination",
        type=str,
        help="Destination location",
    )
    query_group.add_argument(
        "--departure-date",
        type=str,
        help="Departure date (YYYY-MM-DD)",
    )
    query_group.add_argument(
        "--return-date",
        type=str,
        help="Return date (YYYY-MM-DD)",
    )
    query_group.add_argument(
        "--travelers",
        type=int,
        default=1,
        help="Number of travelers",
    )
    query_group.add_argument(
        "--budget",
        type=str,
        help="Budget range (e.g., '1000-2000')",
    )
    query_group.add_argument(
        "--preferences-file",
        type=str,
        help="Path to JSON file with detailed user preferences",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--save-to",
        type=str,
        help="Save the travel plan to specified file path",
    )
    output_group.add_argument(
        "--format",
        type=str,
        choices=["json", "text", "html", "pdf"],
        default="json",
        help="Output format for saved travel plans",
    )
    output_group.add_argument(
        "--save-to-db",
        action="store_true",
        help="Save the travel plan to the Supabase database",
    )

    return parser


async def run_interactive_mode(args: argparse.Namespace) -> None:
    """
    Run the travel planner in interactive mode, allowing users to have a conversation.

    Args:
        args: Command-line arguments
    """
    logger.info("Starting interactive travel planning session")

    # Initialize the travel workflow
    workflow = create_travel_workflow(args)

    print("\n=== AI Travel Planning System ===")
    print("Welcome! Describe your travel plans and preferences.")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'help' for available commands.\n")

    state = TravelPlanningState()
    travel_plan = None

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Check for exit command
        if user_input.lower() in ["exit", "quit", "q", "bye"]:
            print("\nThank you for using the Travel Planner. Goodbye!")
            break

        # Check for help command
        if user_input.lower() == "help":
            display_help()
            continue

        # Check for save command
        if user_input.lower().startswith("save"):
            if travel_plan:
                path = user_input[5:].strip() if len(user_input) > 5 else None
                await save_travel_plan(travel_plan, path, args.format)
                print("\nTravel Planner: Travel plan saved successfully.")
            else:
                print("\nTravel Planner: No travel plan available to save.")
            continue

        try:
            # Parse the user input as a travel query
            query = TravelQuery(raw_query=user_input)

            # Update the state with the query
            state.query = query

            # Execute the workflow
            print(
                "\nTravel Planner: Processing your request. This may take a moment..."
            )
            state = await workflow.execute(state)

            # Store the travel plan for later saving
            if state and state.travel_plan:
                travel_plan = state.travel_plan

            # Display the results
            if state and state.travel_plan:
                display_travel_plan(state.travel_plan)
            elif state and state.error:
                print(f"\nTravel Planner: I encountered an issue: {state.error}")
            else:
                print(
                    "\nTravel Planner: I couldn't complete your travel planning request."
                )

        except Exception as e:
            logger.error(
                f"Error in interactive session: {e!s}\n{traceback.format_exc()}"
            )
            print(f"\nTravel Planner: I'm sorry, I encountered an error: {e!s}")


def display_help() -> None:
    """
    Display available commands for interactive mode.
    """
    print("\nAvailable commands:")
    print("  help                 - Display this help message")
    print("  save [path]         - Save the current travel plan to a file")
    print("  exit, quit, q, bye  - Exit the application")
    print("\nTravel query examples:")
    print("  I want to visit Tokyo for a week in October")
    print("  Plan a budget trip from New York to London from June 10-17 for 2 people")
    print("  Find family-friendly activities in Paris for a 3-day weekend")


def display_travel_plan(plan: TravelPlan) -> None:
    """
    Display a travel plan in a readable format.

    Args:
        plan: The travel plan to display
    """
    print(f"\n=== Travel Plan to {plan.destination} ===\n")

    # Display flight information
    if plan.flights and len(plan.flights) > 0:
        print("Flights:")
        for i, flight in enumerate(plan.flights):
            print(
                f"  {i+1}. {flight.airline}: {flight.departure_location} to {flight.arrival_location}"
            )
            print(
                f"     {flight.departure_time.strftime('%Y-%m-%d %H:%M')} - {flight.arrival_time.strftime('%Y-%m-%d %H:%M')}"
            )
            print(f"     Price: ${flight.price:.2f}")
        print()

    # Display accommodation information
    if plan.accommodations and len(plan.accommodations) > 0:
        print("Accommodations:")
        for i, acc in enumerate(plan.accommodations):
            print(f"  {i+1}. {acc.name} ({acc.type})")
            print(
                f"     {acc.check_in_date.strftime('%Y-%m-%d')} to {acc.check_out_date.strftime('%Y-%m-%d')}"
            )
            print(
                f"     Price: ${acc.price_per_night:.2f} per night (Total: ${acc.total_price:.2f})"
            )
        print()

    # Display daily itinerary
    if plan.daily_itinerary and len(plan.daily_itinerary) > 0:
        print("Daily Itinerary:")
        for i, day in enumerate(plan.daily_itinerary):
            print(f"  Day {i+1} ({day.date.strftime('%Y-%m-%d')}):")
            for j, activity in enumerate(day.activities):
                print(
                    f"     {j+1}. {activity.name} ({activity.time_start.strftime('%H:%M')} - {activity.time_end.strftime('%H:%M')})"
                )
                print(f"        {activity.description}")
                if activity.price > 0:
                    print(f"        Price: ${activity.price:.2f}")
            print()

    # Display budget summary
    if plan.budget_summary:
        print("Budget Summary:")
        print(f"  Total Estimated Cost: ${plan.budget_summary.total_cost:.2f}")

        if len(plan.budget_summary.breakdown) > 0:
            print("  Breakdown:")
            for category, amount in plan.budget_summary.breakdown.items():
                print(f"     {category}: ${amount:.2f}")
        print()


async def save_travel_plan(
    plan: TravelPlan, file_path: str | None = None, format_type: str = "json"
) -> None:
    """
    Save a travel plan to a file.

    Args:
        plan: Travel plan to save
        file_path: Path to save the file (optional)
        format_type: Format type (json, text, html, pdf)
    """
    if not file_path:
        # Generate a default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = plan.destination.replace(" ", "_")
        file_path = f"travel_plan_{destination}_{timestamp}.{format_type}"

    # Create the directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Save the plan in the specified format
    if format_type == "json":
        with open(file_path, "w") as f:
            json.dump(plan.model_dump(mode="json"), f, indent=2)
    elif format_type == "text":
        with open(file_path, "w") as f:
            f.write(f"Travel Plan to {plan.destination}\n\n")
            # Write formatted text representation (implementation details omitted for brevity)
    elif format_type == "html":
        # Generate HTML representation (implementation details omitted for brevity)
        html_content = f"<html><head><title>Travel Plan to {plan.destination}</title></head><body>...</body></html>"
        with open(file_path, "w") as f:
            f.write(html_content)
    elif format_type == "pdf":
        # PDF generation would require additional libraries
        logger.error("PDF export not implemented yet")
        raise NotImplementedError("PDF export not implemented yet")

    logger.info(f"Travel plan saved to {file_path}")


async def run_query_mode(args: argparse.Namespace) -> TravelPlanningState:
    """
    Run the travel planner with a single query and return the results.

    Args:
        args: Command-line arguments

    Returns:
        Final state after travel planning process
    """
    query_text = args.query or ""
    logger.info(f"Starting travel planning for query: {query_text}")

    # Build a travel query from the command-line arguments
    query = build_travel_query_from_args(args)

    # Create initial state
    state = TravelPlanningState(query=query)

    # Initialize the travel workflow
    workflow = create_travel_workflow(args)

    # Execute the workflow
    logger.info("Executing travel planning workflow")
    final_state = await workflow.execute(state)

    # Save the results if requested
    if args.save_to and final_state and final_state.travel_plan:
        await save_travel_plan(final_state.travel_plan, args.save_to, args.format)
        logger.info(f"Travel plan saved to {args.save_to}")

    # Save to database if requested
    if args.save_to_db and final_state and final_state.travel_plan:
        await save_to_database(final_state.travel_plan)
        logger.info("Travel plan saved to database")

    return final_state


def build_travel_query_from_args(args: argparse.Namespace) -> TravelQuery:
    """
    Build a TravelQuery from command-line arguments.

    Args:
        args: Command-line arguments

    Returns:
        Constructed TravelQuery object
    """
    # Start with the raw query
    query_params = {"raw_query": args.query or ""}

    # Add optional arguments if provided
    if args.origin:
        query_params["origin"] = args.origin
    if args.destination:
        query_params["destination"] = args.destination
    if args.departure_date:
        try:
            query_params["departure_date"] = date.fromisoformat(args.departure_date)
        except ValueError:
            logger.warning(
                f"Invalid departure date format: {args.departure_date}. Expected YYYY-MM-DD."
            )
    if args.return_date:
        try:
            query_params["return_date"] = date.fromisoformat(args.return_date)
        except ValueError:
            logger.warning(
                f"Invalid return date format: {args.return_date}. Expected YYYY-MM-DD."
            )
    if args.travelers:
        query_params["travelers"] = args.travelers
    if args.budget:
        try:
            # Parse budget range like "1000-2000"
            min_val, max_val = map(float, args.budget.split("-"))
            query_params["budget_range"] = {"min": min_val, "max": max_val}
        except ValueError:
            logger.warning(
                f"Invalid budget format: {args.budget}. Expected format like '1000-2000'."
            )

    # Load preferences from file if provided
    if args.preferences_file and os.path.exists(args.preferences_file):
        try:
            with open(args.preferences_file) as f:
                preferences = json.load(f)
                query_params["requirements"] = preferences
        except Exception as e:
            logger.warning(f"Error loading preferences file: {e!s}")

    return TravelQuery(**query_params)


async def save_to_database(travel_plan: TravelPlan) -> None:
    """
    Save a travel plan to the Supabase database.

    Args:
        travel_plan: Travel plan to save
    """
    try:
        # Initialize Supabase client
        supabase_client = SupabaseClient()

        # Save the travel plan
        travel_plan_data = travel_plan.model_dump(mode="json")
        result = (
            await supabase_client.from_("travel_plans")
            .insert(travel_plan_data)
            .execute()
        )

        logger.info(
            f"Travel plan saved to database with ID: {result.data[0]['id'] if result.data else 'unknown'}"
        )
    except Exception as e:
        logger.error(f"Error saving to database: {e!s}")
        raise


def create_travel_workflow(args: argparse.Namespace) -> TravelWorkflow:
    """
    Create a TravelWorkflow with all necessary agents and components.

    Args:
        args: Command-line arguments

    Returns:
        Configured TravelWorkflow
    """
    # Initialize research tools
    research_tools = DestinationResearchTools()

    # Set up browser automation options
    browser_options = {
        "headless": args.headless,
        "use_cache": not args.no_cache,
    }

    # Initialize agents
    destination_agent = DestinationResearchAgent(research_tools=research_tools)
    flight_agent = FlightSearchAgent(browser_options=browser_options)
    accommodation_agent = AccommodationAgent(browser_options=browser_options)
    transportation_agent = TransportationAgent(browser_options=browser_options)
    activity_agent = ActivityPlanningAgent(research_tools=research_tools)
    budget_agent = BudgetManagementAgent()

    # Create the workflow
    workflow = TravelWorkflow(
        destination_research_agent=destination_agent,
        flight_search_agent=flight_agent,
        accommodation_agent=accommodation_agent,
        transportation_agent=transportation_agent,
        activity_planning_agent=activity_agent,
        budget_management_agent=budget_agent,
    )

    return workflow


def main() -> int:
    """
    Main entry point function.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = setup_argparse()
    args = parser.parse_args()

    # Initialize configuration
    system_config = initialize_config(custom_config_path=args.config)

    # Setup logging
    setup_logging(
        log_level=args.log_level or system_config.system.log_level,
        log_file=args.log_file,
    )

    logger.info("Starting AI Travel Planning System")

    try:
        # Determine if we should run in query mode (any query-related argument provided)
        query_mode = any(
            [
                args.query,
                args.origin,
                args.destination,
                args.departure_date,
                args.return_date,
                args.budget,
                args.preferences_file,
            ]
        )

        if query_mode:
            # Run in query mode
            final_state = asyncio.run(run_query_mode(args))

            # Display the results
            if final_state and final_state.travel_plan:
                display_travel_plan(final_state.travel_plan)
            elif final_state and final_state.error:
                print(f"Error: {final_state.error}")
                return 1
            else:
                print("No travel plan was generated.")
                return 1
        else:
            # Run in interactive mode
            asyncio.run(run_interactive_mode(args))

        return 0
    except KeyboardInterrupt:
        logger.info("Travel planning session interrupted by user")
        print("\nTravel planning session interrupted. Goodbye!")
        return 0
    except Exception as e:
        logger.error(f"Error in main function: {e!s}\n{traceback.format_exc()}")
        print(f"Error: {e!s}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
