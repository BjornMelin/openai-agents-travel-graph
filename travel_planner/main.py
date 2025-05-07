"""
Main entry point for the Travel Planner application.

This module serves as the application's entry point, initializing the necessary
components and providing a CLI interface for interacting with the travel planning
system.
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
from travel_planner.config import TravelPlannerConfig, initialize_config
from travel_planner.data.models import TravelPlan, TravelQuery
from travel_planner.data.setup import initialize_database
from travel_planner.data.supabase import SupabaseClient
from travel_planner.orchestration.state_graph import TravelPlanningState
from travel_planner.orchestration.workflow import TravelWorkflow
from travel_planner.utils.logging import get_logger, setup_logging
from travel_planner.utils.rate_limiting import initialize_rate_limiting

# Initialize logger
logger = get_logger(__name__)

# Constants
SAVE_COMMAND_LENGTH = 5  # Length of the "save " command prefix


def setup_argparse() -> argparse.ArgumentParser:
    """
    Set up the argument parser for the CLI.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="AI Travel Planning System powered by OpenAI Agents SDK"
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
    system_group.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize Supabase database tables if they don't exist",
    )
    system_group.add_argument(
        "--disable-rate-limits",
        action="store_true",
        help="Disable API rate limiting (use with caution)",
    )
    system_group.add_argument(
        "--rate-limit-config",
        type=str,
        help="Path to custom rate limit configuration file",
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
                path = (
                    user_input[SAVE_COMMAND_LENGTH:].strip()
                    if len(user_input) > SAVE_COMMAND_LENGTH
                    else None
                )
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
                    "\nTravel Planner: I couldn't complete your travel planning"
                    " request."
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
                f"  {i + 1}. {flight.airline}: {flight.departure_location} to "
                f"{flight.arrival_location}"
            )
            print(
                f"     {flight.departure_time.strftime('%Y-%m-%d %H:%M')} - "
                f"{flight.arrival_time.strftime('%Y-%m-%d %H:%M')}"
            )
            print(f"     Price: ${flight.price:.2f}")
        print()

    # Display accommodation information
    if plan.accommodations and len(plan.accommodations) > 0:
        print("Accommodations:")
        for i, acc in enumerate(plan.accommodations):
            print(f"  {i + 1}. {acc.name} ({acc.type})")
            print(
                f"     {acc.check_in_date.strftime('%Y-%m-%d')} to "
                f"{acc.check_out_date.strftime('%Y-%m-%d')}"
            )
            print(
                f"     Price: ${acc.price_per_night:.2f} per night "
                f"(Total: ${acc.total_price:.2f})"
            )
        print()

    # Display daily itinerary
    if plan.daily_itinerary and len(plan.daily_itinerary) > 0:
        print("Daily Itinerary:")
        for i, day in enumerate(plan.daily_itinerary):
            print(f"  Day {i + 1} ({day.date.strftime('%Y-%m-%d')}):")
            for j, activity in enumerate(day.activities):
                print(
                    f"     {j + 1}. {activity.name} "
                    f"({activity.time_start.strftime('%H:%M')} - "
                    f"{activity.time_end.strftime('%H:%M')})"
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
    file_path = _prepare_file_path(plan, file_path, format_type)

    # Save the plan in the specified format
    if format_type == "json":
        _save_as_json(plan, file_path)
    elif format_type == "text":
        _save_as_text(plan, file_path)
    elif format_type == "html":
        _save_as_html(plan, file_path)
    elif format_type == "pdf":
        _handle_pdf_save(file_path)

    logger.info(f"Travel plan saved to {file_path}")


def _prepare_file_path(
    plan: TravelPlan, file_path: str | None, format_type: str
) -> str:
    """Prepare the file path for saving the travel plan."""
    if not file_path:
        # Generate a default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = plan.destination.replace(" ", "_")
        file_path = f"travel_plan_{destination}_{timestamp}.{format_type}"

    # Create the directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    return file_path


def _save_as_json(plan: TravelPlan, file_path: str) -> None:
    """Save the travel plan as JSON."""
    with open(file_path, "w") as f:
        json.dump(plan.model_dump(mode="json"), f, indent=2)


def _handle_pdf_save(file_path: str) -> None:
    """Handle PDF save request."""
    # PDF generation would require additional libraries
    logger.error("PDF export not implemented yet")
    raise NotImplementedError("PDF export not implemented yet")


def _save_as_text(plan: TravelPlan, file_path: str) -> None:
    """Save the travel plan as plain text."""
    with open(file_path, "w") as f:
        # Write header and overview
        _write_text_header(f, plan)
        _write_text_overview(f, plan)

        # Write main sections
        _write_text_flights(f, plan)
        _write_text_accommodations(f, plan)
        _write_text_activities(f, plan)
        _write_text_budget(f, plan)
        _write_text_recommendations(f, plan)
        _write_text_alerts(f, plan)


def _write_text_header(f: TextIO, plan: TravelPlan) -> None:
    """Write the header section to the text file."""
    destination_name = (
        plan.destination.get("name", "Unknown")
        if isinstance(plan.destination, dict)
        else plan.destination
    )
    f.write(f"TRAVEL PLAN TO {destination_name.upper()}\n")
    f.write("=" * 50 + "\n\n")


def _write_text_overview(f: TextIO, plan: TravelPlan) -> None:
    """Write the overview section to the text file."""
    if not plan.overview:
        return

    f.write("OVERVIEW\n")
    f.write("-" * 8 + "\n")
    f.write(f"{plan.overview}\n\n")


def _write_text_flights(f: TextIO, plan: TravelPlan) -> None:
    """Write flight information to the text file."""
    if not plan.flights or len(plan.flights) == 0:
        return

    f.write("FLIGHTS\n")
    f.write("-" * 7 + "\n")

    for i, flight in enumerate(plan.flights):
        f.write(f"{i + 1}. {flight.airline}: {flight.flight_number}\n")
        f.write(f"   From: {flight.departure_airport} - To: {flight.arrival_airport}\n")
        f.write(f"   Departure: {flight.departure_time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"   Arrival: {flight.arrival_time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"   Class: {flight.travel_class.value}\n")
        f.write(f"   Price: {flight.currency} {flight.price:.2f}\n")

        _write_text_flight_layovers(f, flight)

        f.write(f"   Duration: {flight.duration_minutes} minutes\n")
        if flight.booking_link:
            f.write(f"   Booking: {flight.booking_link}\n")
        f.write("\n")


def _write_text_flight_layovers(f: TextIO, flight: Flight) -> None:
    """Write flight layover information to the text file."""
    if not flight.layovers or len(flight.layovers) == 0:
        return

    f.write(f"   Layovers: {len(flight.layovers)}\n")
    for j, layover in enumerate(flight.layovers):
        f.write(
            f"      {j + 1}. {layover.get('airport', 'Unknown')} - "
            f"Duration: {layover.get('duration_minutes', 0)} min\n"
        )


def _write_text_accommodations(f: TextIO, plan: TravelPlan) -> None:
    """Write accommodation information to the text file."""
    if not plan.accommodation or len(plan.accommodation) == 0:
        return

    f.write("ACCOMMODATIONS\n")
    f.write("-" * 14 + "\n")

    for i, acc in enumerate(plan.accommodation):
        f.write(f"{i + 1}. {acc.name} ({acc.type.value})\n")
        f.write(f"   Address: {acc.address}\n")
        if acc.rating:
            f.write(f"   Rating: {acc.rating}/5\n")
        f.write(f"   Check-in: {acc.check_in_time} - Check-out: {acc.check_out_time}\n")
        f.write(f"   Price per night: {acc.currency} {acc.price_per_night:.2f}\n")
        f.write(f"   Total price: {acc.currency} {acc.total_price:.2f}\n")

        if acc.amenities and len(acc.amenities) > 0:
            f.write(f"   Amenities: {', '.join(acc.amenities)}\n")
        if acc.booking_link:
            f.write(f"   Booking: {acc.booking_link}\n")
        f.write("\n")


def _write_text_activities(f: TextIO, plan: TravelPlan) -> None:
    """Write activities information to the text file."""
    if not plan.activities or len(plan.activities) == 0:
        return

    f.write("DAILY ITINERARY\n")
    f.write("-" * 15 + "\n")

    for _day_key, day in plan.activities.items():
        f.write(f"Day {day.day_number} - {day.date.strftime('%Y-%m-%d')}\n")

        # Write weather if available
        _write_text_day_weather(f, day)

        # Write activities
        _write_text_day_activities(f, day)

        # Write transportation
        _write_text_day_transportation(f, day)

        # Write notes
        if day.notes:
            f.write(f"   Notes: {day.notes}\n")

        f.write("\n")


def _write_text_day_weather(f: TextIO, day: DayItinerary) -> None:
    """Write day weather information to the text file."""
    if not day.weather_forecast:
        return

    weather = day.weather_forecast
    f.write(
        f"   Weather: {weather.get('description', 'N/A')}, "
        f"{weather.get('temperature', 'N/A')}°C\n"
    )


def _write_text_day_activities(f: TextIO, day: DayItinerary) -> None:
    """Write day activities information to the text file."""
    if not day.activities or len(day.activities) == 0:
        return

    for i, activity in enumerate(day.activities):
        duration_hours = activity.duration_minutes // 60
        duration_mins = activity.duration_minutes % 60
        duration = (
            f"{duration_hours}h {duration_mins}m"
            if duration_hours > 0
            else f"{duration_mins}m"
        )

        f.write(f"   {i + 1}. {activity.name} ({activity.type.value})\n")
        f.write(f"      {activity.description}\n")
        f.write(f"      Location: {activity.location}\n")
        f.write(f"      Duration: {duration}\n")

        if activity.cost:
            f.write(f"      Cost: {activity.currency} {activity.cost:.2f}\n")

        if activity.booking_required:
            booking_info = (
                f" - {activity.booking_link}" if activity.booking_link else ""
            )
            f.write(f"      Booking required{booking_info}\n")


def _write_text_day_transportation(f: TextIO, day: DayItinerary) -> None:
    """Write day transportation information to the text file."""
    if not day.transportation or len(day.transportation) == 0:
        return

    f.write("   Transportation:\n")
    for i, transport in enumerate(day.transportation):
        f.write(f"      {i + 1}. {transport.type.value}: {transport.description}\n")
        if transport.cost:
            f.write(f"         Cost: {transport.currency} {transport.cost:.2f}\n")


def _write_text_budget(f: TextIO, plan: TravelPlan) -> None:
    """Write budget information to the text file."""
    if not plan.budget:
        return

    f.write("BUDGET SUMMARY\n")
    f.write("-" * 14 + "\n")
    f.write(f"Total budget: {plan.budget.currency} {plan.budget.total_budget:.2f}\n")
    f.write(f"Spent: {plan.budget.currency} {plan.budget.spent:.2f}\n")
    f.write(f"Remaining: {plan.budget.currency} {plan.budget.remaining:.2f}\n")

    # Write budget breakdown
    _write_text_budget_breakdown(f, plan)

    # Write saving recommendations
    _write_text_saving_recommendations(f, plan)

    f.write("\n")


def _write_text_budget_breakdown(f: TextIO, plan: TravelPlan) -> None:
    """Write budget breakdown to the text file."""
    if not plan.budget or not plan.budget.breakdown or len(plan.budget.breakdown) == 0:
        return

    f.write("Breakdown:\n")
    for category, amount in plan.budget.breakdown.items():
        f.write(f"   {category}: {plan.budget.currency} {amount:.2f}\n")


def _write_text_saving_recommendations(f: TextIO, plan: TravelPlan) -> None:
    """Write saving recommendations to the text file."""
    if not plan.budget or not plan.budget.saving_recommendations:
        return

    if len(plan.budget.saving_recommendations) == 0:
        return

    f.write("Saving Recommendations:\n")
    for i, rec in enumerate(plan.budget.saving_recommendations):
        f.write(f"   {i + 1}. {rec}\n")


def _write_text_recommendations(f: TextIO, plan: TravelPlan) -> None:
    """Write recommendations to the text file."""
    if not plan.recommendations or len(plan.recommendations) == 0:
        return

    f.write("RECOMMENDATIONS\n")
    f.write("-" * 16 + "\n")
    for i, rec in enumerate(plan.recommendations):
        f.write(f"{i + 1}. {rec}\n")
    f.write("\n")


def _write_text_alerts(f: TextIO, plan: TravelPlan) -> None:
    """Write alerts to the text file."""
    if not plan.alerts or len(plan.alerts) == 0:
        return

    f.write("IMPORTANT ALERTS\n")
    f.write("-" * 16 + "\n")
    for i, alert in enumerate(plan.alerts):
        f.write(f"{i + 1}. {alert}\n")


def _save_as_html(plan: TravelPlan, file_path: str) -> None:
    """Save the travel plan as HTML."""
    destination_name = _get_destination_name(plan)

    # Generate HTML content
    html_parts = [
        _generate_html_header(destination_name),
        _generate_html_overview(plan),
        _generate_html_flights(plan),
        _generate_html_accommodations(plan),
        _generate_html_activities(plan),
        _generate_html_budget(plan),
        _generate_html_recommendations(plan),
        _generate_html_alerts(plan),
        _generate_html_footer(),
    ]

    html_content = "".join(html_parts)

    with open(file_path, "w") as f:
        f.write(html_content)


def _get_destination_name(plan: TravelPlan) -> str:
    """Get the destination name from the travel plan."""
    return (
        plan.destination.get("name", "Unknown")
        if isinstance(plan.destination, dict)
        else plan.destination
    )


def _generate_html_header(destination_name: str) -> str:
    """Generate HTML header section."""
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Travel Plan to {destination_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        .card {{
            background: #f9f9f9;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .card h3 {{
            margin-top: 0;
            color: #3498db;
        }}
        .flight, .accommodation, .activity, .transportation {{
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px dashed #ddd;
        }}
        .flight:last-child, .accommodation:last-child, 
        .activity:last-child, .transportation:last-child {{
            border-bottom: none;
        }}
        .day {{
            margin-bottom: 30px;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 5px;
        }}
        .day-header {{
            background: #3498db;
            color: white;
            padding: 10px;
            margin: -15px -15px 15px -15px;
            border-radius: 5px 5px 0 0;
        }}
        .price {{
            font-weight: bold;
            color: #e74c3c;
        }}
        .alert {{
            background-color: #f8d7da;
            color: #721c24;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .recommendation {{
            background-color: #d4edda;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        table, th, td {{
            border: 1px solid #ddd;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
    </style>
</head>
<body>
    <h1>Travel Plan to {destination_name}</h1>
"""


def _generate_html_overview(plan: TravelPlan) -> str:
    """Generate HTML overview section."""
    if not plan.overview:
        return ""

    return f"""
    <div class="card">
        <h2>Overview</h2>
        <p>{plan.overview}</p>
    </div>
"""


def _generate_html_flights(plan: TravelPlan) -> str:
    """Generate HTML flights section."""
    if not plan.flights or len(plan.flights) == 0:
        return ""

    html = """
    <h2>Flights</h2>
"""

    for flight in plan.flights:
        html += _generate_html_flight_card(flight)

    return html


def _generate_html_flight_card(flight: Flight) -> str:
    """Generate HTML for a single flight card."""
    html = f"""
    <div class="card flight">
        <h3>{flight.airline} - Flight {flight.flight_number}</h3>
        <p><strong>From:</strong> {flight.departure_airport} &rarr;
           <strong>To:</strong> {flight.arrival_airport}</p>
        <p><strong>Departure:</strong> 
           {flight.departure_time.strftime("%Y-%m-%d %H:%M")}</p>
        <p><strong>Arrival:</strong> 
           {flight.arrival_time.strftime("%Y-%m-%d %H:%M")}</p>
        <p><strong>Class:</strong> {flight.travel_class.value}</p>
        <p><strong>Duration:</strong> 
           {flight.duration_minutes // 60}h {flight.duration_minutes % 60}m</p>
"""

    # Add layovers if any
    if flight.layovers and len(flight.layovers) > 0:
        html += _generate_html_flight_layovers(flight)

    # Add price and booking link
    html += f"""
        <p class="price"><strong>Price:</strong> 
           {flight.currency} {flight.price:.2f}</p>
"""

    if flight.booking_link:
        html += f"""
        <p><a href="{flight.booking_link}" target="_blank">Booking Link</a></p>
"""

    html += """
    </div>
"""

    return html


def _generate_html_flight_layovers(flight: Flight) -> str:
    """Generate HTML for flight layovers."""
    html = f"""
        <p><strong>Layovers:</strong> {len(flight.layovers)}</p>
        <ul>
"""

    for layover in flight.layovers:
        html += f"""
            <li>{layover.get("airport", "Unknown")} - Duration: 
               {layover.get("duration_minutes", 0)} minutes</li>
"""

    html += """
        </ul>
"""

    return html


def _generate_html_accommodations(plan: TravelPlan) -> str:
    """Generate HTML accommodations section."""
    if not plan.accommodation or len(plan.accommodation) == 0:
        return ""

    html = """
    <h2>Accommodations</h2>
"""

    for acc in plan.accommodation:
        html += _generate_html_accommodation_card(acc)

    return html


def _generate_html_accommodation_card(acc: Accommodation) -> str:
    """Generate HTML for a single accommodation card."""
    html = f"""
    <div class="card accommodation">
        <h3>{acc.name} ({acc.type.value})</h3>
        <p><strong>Address:</strong> {acc.address}</p>
"""

    if acc.rating:
        html += f"""
        <p><strong>Rating:</strong> {acc.rating}/5</p>
"""

    html += f"""
        <p><strong>Check-in:</strong> {acc.check_in_time} - 
           <strong>Check-out:</strong> {acc.check_out_time}</p>
        <p class="price">
            <strong>Price per night:</strong> {acc.currency} {acc.price_per_night:.2f}
        </p>
        <p class="price">
            <strong>Total price:</strong> {acc.currency} {acc.total_price:.2f}
        </p>
"""

    if acc.amenities and len(acc.amenities) > 0:
        html += f"""
        <p><strong>Amenities:</strong> {", ".join(acc.amenities)}</p>
"""

    if acc.booking_link:
        html += f"""
        <p><a href="{acc.booking_link}" target="_blank">Booking Link</a></p>
"""

    html += """
    </div>
"""

    return html


def _generate_html_activities(plan: TravelPlan) -> str:
    """Generate HTML activities section."""
    if not plan.activities or len(plan.activities) == 0:
        return ""

    html = """
    <h2>Daily Itinerary</h2>
"""

    # Sort days by day number to ensure correct order
    sorted_days = sorted(plan.activities.items(), key=lambda x: x[1].day_number)

    for _day_key, day in sorted_days:
        html += _generate_html_day_card(day)

    return html


def _generate_html_day_card(day: DayItinerary) -> str:
    """Generate HTML for a day itinerary card."""
    html = f"""
    <div class="day">
        <div class="day-header">
            <h3>Day {day.day_number} - {day.date.strftime("%Y-%m-%d")}</h3>
"""

    if day.weather_forecast:
        weather = day.weather_forecast
        html += f"""
            <p><strong>Weather:</strong> {weather.get("description", "N/A")},
               {weather.get("temperature", "N/A")}°C</p>
"""

    html += """
        </div>
"""

    # Add activities
    html += _generate_html_day_activities(day)

    # Add transportation
    html += _generate_html_day_transportation(day)

    # Add notes
    if day.notes:
        html += f"""
        <div class="notes">
            <h4>Notes</h4>
            <p>{day.notes}</p>
        </div>
"""

    html += """
    </div>
"""

    return html


def _generate_html_day_activities(day: DayItinerary) -> str:
    """Generate HTML for day activities."""
    if not day.activities or len(day.activities) == 0:
        return ""

    html = """
        <h4>Activities</h4>
"""

    for activity in day.activities:
        duration_hours = activity.duration_minutes // 60
        duration_mins = activity.duration_minutes % 60
        duration = (
            f"{duration_hours}h {duration_mins}m"
            if duration_hours > 0
            else f"{duration_mins}m"
        )

        html += f"""
        <div class="activity card">
            <h5>{activity.name} ({activity.type.value})</h5>
            <p>{activity.description}</p>
            <p><strong>Location:</strong> {activity.location}</p>
            <p><strong>Duration:</strong> {duration}</p>
"""

        if activity.cost:
            html += f"""
            <p class="price">
                <strong>Cost:</strong> {activity.currency} {activity.cost:.2f}
            </p>
"""

        if activity.booking_required:
            html += """
            <p><strong>Booking required</strong></p>
"""
            if activity.booking_link:
                html += f"""
            <p><a href="{activity.booking_link}" target="_blank">Booking Link</a></p>
"""

        html += """
        </div>
"""

    return html


def _generate_html_day_transportation(day: DayItinerary) -> str:
    """Generate HTML for day transportation."""
    if not day.transportation or len(day.transportation) == 0:
        return ""

    html = """
        <h4>Transportation</h4>
"""

    for transport in day.transportation:
        html += f"""
        <div class="transportation card">
            <h5>{transport.type.value}</h5>
            <p>{transport.description}</p>
"""

        if transport.cost:
            html += f"""
            <p class="price">
                <strong>Cost:</strong> {transport.currency} {transport.cost:.2f}
            </p>
"""

        if transport.duration_minutes:
            dur_hours = transport.duration_minutes // 60
            dur_mins = transport.duration_minutes % 60
            dur_str = f"{dur_hours}h {dur_mins}m" if dur_hours > 0 else f"{dur_mins}m"
            html += f"""
            <p><strong>Duration:</strong> {dur_str}</p>
"""

        html += """
        </div>
"""

    return html


def _generate_html_budget(plan: TravelPlan) -> str:
    """Generate HTML budget section."""
    if not plan.budget:
        return ""

    html = """
    <h2>Budget Summary</h2>
    <div class="card">
"""

    html += f"""
        <p><strong>Total Budget:</strong> {plan.budget.currency} 
           {plan.budget.total_budget:.2f}</p>
        <p><strong>Spent:</strong> {plan.budget.currency} 
           {plan.budget.spent:.2f}</p>
        <p><strong>Remaining:</strong> {plan.budget.currency} 
           {plan.budget.remaining:.2f}</p>
"""

    # Add budget breakdown
    html += _generate_html_budget_breakdown(plan)

    # Add saving recommendations
    html += _generate_html_saving_recommendations(plan)

    html += """
    </div>
"""

    return html


def _generate_html_budget_breakdown(plan: TravelPlan) -> str:
    """Generate HTML for budget breakdown."""
    if not plan.budget or not plan.budget.breakdown or len(plan.budget.breakdown) == 0:
        return ""

    html = """
        <h3>Budget Breakdown</h3>
        <table>
            <tr>
                <th>Category</th>
                <th>Amount</th>
            </tr>
"""

    for category, amount in plan.budget.breakdown.items():
        html += f"""
            <tr>
                <td>{category}</td>
                <td>{plan.budget.currency} {amount:.2f}</td>
            </tr>
"""

    html += """
        </table>
"""

    return html


def _generate_html_saving_recommendations(plan: TravelPlan) -> str:
    """Generate HTML for saving recommendations."""
    if not plan.budget or not plan.budget.saving_recommendations:
        return ""

    if len(plan.budget.saving_recommendations) == 0:
        return ""

    html = """
        <h3>Saving Recommendations</h3>
        <ul>
"""

    for rec in plan.budget.saving_recommendations:
        html += f"""
            <li>{rec}</li>
"""

    html += """
        </ul>
"""

    return html


def _generate_html_recommendations(plan: TravelPlan) -> str:
    """Generate HTML recommendations section."""
    if not plan.recommendations or len(plan.recommendations) == 0:
        return ""

    html = """
    <h2>Recommendations</h2>
"""

    for rec in plan.recommendations:
        html += f"""
    <div class="recommendation">{rec}</div>
"""

    return html


def _generate_html_alerts(plan: TravelPlan) -> str:
    """Generate HTML alerts section."""
    if not plan.alerts or len(plan.alerts) == 0:
        return ""

    html = """
    <h2>Important Alerts</h2>
"""

    for alert in plan.alerts:
        html += f"""
    <div class="alert">{alert}</div>
"""

    return html


def _generate_html_footer() -> str:
    """Generate HTML footer section."""
    return f"""
    <footer>
        <p><small>Generated by AI Travel Planning System on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</small></p>
    </footer>
</body>
</html>
"""


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
                f"Invalid departure date format: {args.departure_date}. "
                f"Expected YYYY-MM-DD."
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
                f"Invalid budget format: {args.budget}. "
                f"Expected format like '1000-2000'."
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
            f"Travel plan saved to database with ID: "
            f"{result.data[0]['id'] if result.data else 'unknown'}"
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


async def main() -> int:
    """
    Main entry point function.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        # Parse arguments and initialize logging
        args = _parse_arguments_and_setup_basic_logging()

        # Initialize configuration
        system_config = await _initialize_system_configuration(args)
        if not system_config:
            return 1

        # Run in query or interactive mode
        return await _run_selected_mode(args)

    except TravelPlannerConfig.ConfigurationError as e:
        return _handle_configuration_error(e)
    except KeyboardInterrupt:
        return _handle_keyboard_interrupt()
    except FileNotFoundError as e:
        return _handle_file_not_found_error(e)
    except Exception as e:
        return _handle_general_exception(e)


def _parse_arguments_and_setup_basic_logging() -> argparse.Namespace:
    """Parse command-line arguments and set up basic logging."""
    parser = setup_argparse()
    args = parser.parse_args()

    # Setup basic logging first to capture any initialization errors
    setup_logging(log_level=args.log_level or "INFO", log_file=args.log_file)
    logger.info("Starting AI Travel Planning System")

    return args


async def _initialize_system_configuration(
    args: argparse.Namespace,
) -> TravelPlannerConfig | None:
    """Initialize system configuration and rate limiting."""
    # Initialize configuration with validation
    system_config = initialize_config(
        custom_config_path=args.config, validate=True, raise_on_error=False
    )

    # Enhance logging setup now that we have the full configuration
    setup_logging(
        log_level=args.log_level or system_config.system.log_level,
        log_file=args.log_file,
    )

    # Initialize rate limiting
    _initialize_rate_limiting(args)

    # Validate API keys
    if not system_config.api.validate():
        _display_missing_api_keys_error()
        return None

    # Initialize database if requested
    if args.init_db and not await _initialize_database_if_requested(system_config):
        return None

    return system_config


def _initialize_rate_limiting(args: argparse.Namespace) -> None:
    """Initialize rate limiting for external APIs."""
    if args.disable_rate_limits:
        logger.warning(
            "API rate limiting is disabled. This may cause API quota issues."
        )
        return

    logger.info("Initializing API rate limiting...")

    # Check for custom rate limit configuration
    if args.rate_limit_config and os.path.exists(args.rate_limit_config):
        _load_custom_rate_limits(args.rate_limit_config)
    else:
        # Use default rate limits
        initialize_rate_limiting()


def _load_custom_rate_limits(config_path: str) -> None:
    """Load custom rate limits from configuration file."""
    try:
        with open(config_path) as f:
            from travel_planner.utils.rate_limiting import (
                update_rate_limits_from_config,
            )

            rate_limit_config = json.load(f)
            update_rate_limits_from_config(rate_limit_config)
            logger.info(f"Loaded custom rate limit configuration from {config_path}")
    except Exception as e:
        logger.error(f"Error loading rate limit configuration: {e}")
        print(f"\nError loading rate limit configuration: {e}")


def _display_missing_api_keys_error() -> None:
    """Display error message for missing API keys."""
    print("\nERROR: Missing required API keys. Please configure your environment:")
    print("  - OPENAI_API_KEY: Required for AI agent functionality")
    print("  - SUPABASE_URL: Required for data persistence")
    print("  - SUPABASE_KEY: Required for data persistence")
    print("\nYou can set these in a .env file in the project root directory.")
    print("See the README.md for setup instructions.\n")


async def _initialize_database_if_requested(system_config: TravelPlannerConfig) -> bool:
    """Initialize database if requested by command-line arguments."""
    logger.info("Initializing Supabase database...")
    try:
        # Initialize database tables
        db_success = await initialize_database(system_config)
        if not db_success:
            logger.error("Failed to initialize Supabase database")
            print("\nERROR: Failed to initialize Supabase database.")
            print(
                "Please check logs for details or run "
                "'python supabase_setup.py init' for more information.\n"
            )
            return False
        logger.info("Supabase database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing Supabase database: {e}")
        print(f"\nERROR: Failed to initialize Supabase database: {e}")
        return False


async def _run_selected_mode(args: argparse.Namespace) -> int:
    """Run the travel planner in the selected mode (query or interactive)."""
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
        return await _run_query_mode_with_display(args)
    else:
        # Run in interactive mode
        await run_interactive_mode(args)
        return 0


async def _run_query_mode_with_display(args: argparse.Namespace) -> int:
    """Run the travel planner in query mode and display results."""
    # Run in query mode
    final_state = await run_query_mode(args)

    # Display the results
    if final_state and final_state.travel_plan:
        display_travel_plan(final_state.travel_plan)
        return 0
    elif final_state and final_state.error:
        print(f"Error: {final_state.error}")
        return 1
    else:
        print("No travel plan was generated.")
        return 1


def _handle_configuration_error(e: Exception) -> int:
    """Handle configuration errors."""
    logger.error(f"Configuration error: {e}")
    print(f"\nConfiguration Error: {e}")
    print("Please check your environment variables and configuration settings.")
    return 1


def _handle_keyboard_interrupt() -> int:
    """Handle keyboard interrupt (CTRL+C)."""
    logger.info("Travel planning session interrupted by user")
    print("\nTravel planning session interrupted. Goodbye!")
    return 0


def _handle_file_not_found_error(e: Exception) -> int:
    """Handle file not found errors."""
    logger.error(f"File not found: {e}")
    print(f"\nError: {e}")
    return 1


def _handle_general_exception(e: Exception) -> int:
    """Handle general exceptions."""
    logger.error(f"Error in main function: {e!s}\n{traceback.format_exc()}")
    print(f"\nError: {e!s}")
    print("An unexpected error occurred. Please check the logs for more details.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
