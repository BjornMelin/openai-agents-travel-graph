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
            # Write the header
            destination_name = plan.destination.get("name", "Unknown") if isinstance(plan.destination, dict) else plan.destination
            f.write(f"TRAVEL PLAN TO {destination_name.upper()}\n")
            f.write("=" * 50 + "\n\n")
            
            # Write overview if available
            if plan.overview:
                f.write("OVERVIEW\n")
                f.write("-" * 8 + "\n")
                f.write(f"{plan.overview}\n\n")
            
            # Write flight information
            if plan.flights and len(plan.flights) > 0:
                f.write("FLIGHTS\n")
                f.write("-" * 7 + "\n")
                for i, flight in enumerate(plan.flights):
                    f.write(f"{i+1}. {flight.airline}: {flight.flight_number}\n")
                    f.write(f"   From: {flight.departure_airport} - To: {flight.arrival_airport}\n")
                    f.write(f"   Departure: {flight.departure_time.strftime('%Y-%m-%d %H:%M')}\n")
                    f.write(f"   Arrival: {flight.arrival_time.strftime('%Y-%m-%d %H:%M')}\n")
                    f.write(f"   Class: {flight.travel_class.value}\n")
                    f.write(f"   Price: {flight.currency} {flight.price:.2f}\n")
                    if flight.layovers and len(flight.layovers) > 0:
                        f.write(f"   Layovers: {len(flight.layovers)}\n")
                        for j, layover in enumerate(flight.layovers):
                            f.write(f"      {j+1}. {layover.get('airport', 'Unknown')} - Duration: {layover.get('duration_minutes', 0)} min\n")
                    f.write(f"   Duration: {flight.duration_minutes} minutes\n")
                    if flight.booking_link:
                        f.write(f"   Booking: {flight.booking_link}\n")
                    f.write("\n")
            
            # Write accommodation information
            if plan.accommodation and len(plan.accommodation) > 0:
                f.write("ACCOMMODATIONS\n")
                f.write("-" * 14 + "\n")
                for i, acc in enumerate(plan.accommodation):
                    f.write(f"{i+1}. {acc.name} ({acc.type.value})\n")
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
            
            # Write daily itinerary
            if plan.activities and len(plan.activities) > 0:
                f.write("DAILY ITINERARY\n")
                f.write("-" * 15 + "\n")
                for day_key, day in plan.activities.items():
                    f.write(f"Day {day.day_number} - {day.date.strftime('%Y-%m-%d')}\n")
                    if day.weather_forecast:
                        weather = day.weather_forecast
                        f.write(f"   Weather: {weather.get('description', 'N/A')}, {weather.get('temperature', 'N/A')}°C\n")
                    
                    for i, activity in enumerate(day.activities):
                        duration_hours = activity.duration_minutes // 60
                        duration_mins = activity.duration_minutes % 60
                        duration = f"{duration_hours}h {duration_mins}m" if duration_hours > 0 else f"{duration_mins}m"
                        
                        f.write(f"   {i+1}. {activity.name} ({activity.type.value})\n")
                        f.write(f"      {activity.description}\n")
                        f.write(f"      Location: {activity.location}\n")
                        f.write(f"      Duration: {duration}\n")
                        if activity.cost:
                            f.write(f"      Cost: {activity.currency} {activity.cost:.2f}\n")
                        if activity.booking_required:
                            booking_info = f" - {activity.booking_link}" if activity.booking_link else ""
                            f.write(f"      Booking required{booking_info}\n")
                    
                    # Transportation for the day
                    if day.transportation and len(day.transportation) > 0:
                        f.write("   Transportation:\n")
                        for i, transport in enumerate(day.transportation):
                            f.write(f"      {i+1}. {transport.type.value}: {transport.description}\n")
                            if transport.cost:
                                f.write(f"         Cost: {transport.currency} {transport.cost:.2f}\n")
                    
                    # Notes for the day
                    if day.notes:
                        f.write(f"   Notes: {day.notes}\n")
                    
                    f.write("\n")
            
            # Write budget information
            if plan.budget:
                f.write("BUDGET SUMMARY\n")
                f.write("-" * 14 + "\n")
                f.write(f"Total budget: {plan.budget.currency} {plan.budget.total_budget:.2f}\n")
                f.write(f"Spent: {plan.budget.currency} {plan.budget.spent:.2f}\n")
                f.write(f"Remaining: {plan.budget.currency} {plan.budget.remaining:.2f}\n")
                
                if plan.budget.breakdown and len(plan.budget.breakdown) > 0:
                    f.write("Breakdown:\n")
                    for category, amount in plan.budget.breakdown.items():
                        f.write(f"   {category}: {plan.budget.currency} {amount:.2f}\n")
                
                if plan.budget.saving_recommendations and len(plan.budget.saving_recommendations) > 0:
                    f.write("Saving Recommendations:\n")
                    for i, rec in enumerate(plan.budget.saving_recommendations):
                        f.write(f"   {i+1}. {rec}\n")
                f.write("\n")
            
            # Write recommendations
            if plan.recommendations and len(plan.recommendations) > 0:
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 16 + "\n")
                for i, rec in enumerate(plan.recommendations):
                    f.write(f"{i+1}. {rec}\n")
                f.write("\n")
            
            # Write alerts
            if plan.alerts and len(plan.alerts) > 0:
                f.write("IMPORTANT ALERTS\n")
                f.write("-" * 16 + "\n")
                for i, alert in enumerate(plan.alerts):
                    f.write(f"{i+1}. {alert}\n")
    
    elif format_type == "html":
        destination_name = plan.destination.get("name", "Unknown") if isinstance(plan.destination, dict) else plan.destination
        
        # Build HTML content with CSS styling
        html_content = f'''
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
        .flight:last-child, .accommodation:last-child, .activity:last-child, .transportation:last-child {{
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
'''

        # Add overview if available
        if plan.overview:
            html_content += f'''
    <div class="card">
        <h2>Overview</h2>
        <p>{plan.overview}</p>
    </div>
'''

        # Add flights section
        if plan.flights and len(plan.flights) > 0:
            html_content += '''
    <h2>Flights</h2>
'''
            for flight in plan.flights:
                html_content += f'''
    <div class="card flight">
        <h3>{flight.airline} - Flight {flight.flight_number}</h3>
        <p><strong>From:</strong> {flight.departure_airport} &rarr; <strong>To:</strong> {flight.arrival_airport}</p>
        <p><strong>Departure:</strong> {flight.departure_time.strftime('%Y-%m-%d %H:%M')}</p>
        <p><strong>Arrival:</strong> {flight.arrival_time.strftime('%Y-%m-%d %H:%M')}</p>
        <p><strong>Class:</strong> {flight.travel_class.value}</p>
        <p><strong>Duration:</strong> {flight.duration_minutes // 60}h {flight.duration_minutes % 60}m</p>
'''
                if flight.layovers and len(flight.layovers) > 0:
                    html_content += f'''
        <p><strong>Layovers:</strong> {len(flight.layovers)}</p>
        <ul>
'''
                    for layover in flight.layovers:
                        html_content += f'''
            <li>{layover.get('airport', 'Unknown')} - Duration: {layover.get('duration_minutes', 0)} minutes</li>
'''
                    html_content += '''
        </ul>
'''
                html_content += f'''
        <p class="price"><strong>Price:</strong> {flight.currency} {flight.price:.2f}</p>
'''
                if flight.booking_link:
                    html_content += f'''
        <p><a href="{flight.booking_link}" target="_blank">Booking Link</a></p>
'''
                html_content += '''
    </div>
'''

        # Add accommodations section
        if plan.accommodation and len(plan.accommodation) > 0:
            html_content += '''
    <h2>Accommodations</h2>
'''
            for acc in plan.accommodation:
                html_content += f'''
    <div class="card accommodation">
        <h3>{acc.name} ({acc.type.value})</h3>
        <p><strong>Address:</strong> {acc.address}</p>
'''
                if acc.rating:
                    html_content += f'''
        <p><strong>Rating:</strong> {acc.rating}/5</p>
'''
                html_content += f'''
        <p><strong>Check-in:</strong> {acc.check_in_time} - <strong>Check-out:</strong> {acc.check_out_time}</p>
        <p class="price"><strong>Price per night:</strong> {acc.currency} {acc.price_per_night:.2f}</p>
        <p class="price"><strong>Total price:</strong> {acc.currency} {acc.total_price:.2f}</p>
'''
                if acc.amenities and len(acc.amenities) > 0:
                    html_content += f'''
        <p><strong>Amenities:</strong> {', '.join(acc.amenities)}</p>
'''
                if acc.booking_link:
                    html_content += f'''
        <p><a href="{acc.booking_link}" target="_blank">Booking Link</a></p>
'''
                html_content += '''
    </div>
'''

        # Add daily itinerary
        if plan.activities and len(plan.activities) > 0:
            html_content += '''
    <h2>Daily Itinerary</h2>
'''
            # Sort days by day number to ensure correct order
            sorted_days = sorted(plan.activities.items(), key=lambda x: x[1].day_number)
            
            for day_key, day in sorted_days:
                html_content += f'''
    <div class="day">
        <div class="day-header">
            <h3>Day {day.day_number} - {day.date.strftime('%Y-%m-%d')}</h3>
'''
                if day.weather_forecast:
                    weather = day.weather_forecast
                    html_content += f'''
            <p><strong>Weather:</strong> {weather.get('description', 'N/A')}, {weather.get('temperature', 'N/A')}°C</p>
'''
                html_content += '''
        </div>
'''
                # Activities for the day
                if day.activities and len(day.activities) > 0:
                    html_content += '''
        <h4>Activities</h4>
'''
                    for activity in day.activities:
                        duration_hours = activity.duration_minutes // 60
                        duration_mins = activity.duration_minutes % 60
                        duration = f"{duration_hours}h {duration_mins}m" if duration_hours > 0 else f"{duration_mins}m"
                        
                        html_content += f'''
        <div class="activity card">
            <h5>{activity.name} ({activity.type.value})</h5>
            <p>{activity.description}</p>
            <p><strong>Location:</strong> {activity.location}</p>
            <p><strong>Duration:</strong> {duration}</p>
'''
                        if activity.cost:
                            html_content += f'''
            <p class="price"><strong>Cost:</strong> {activity.currency} {activity.cost:.2f}</p>
'''
                        if activity.booking_required:
                            html_content += '''
            <p><strong>Booking required</strong></p>
'''
                            if activity.booking_link:
                                html_content += f'''
            <p><a href="{activity.booking_link}" target="_blank">Booking Link</a></p>
'''
                        html_content += '''
        </div>
'''
                
                # Transportation for the day
                if day.transportation and len(day.transportation) > 0:
                    html_content += '''
        <h4>Transportation</h4>
'''
                    for transport in day.transportation:
                        html_content += f'''
        <div class="transportation card">
            <h5>{transport.type.value}</h5>
            <p>{transport.description}</p>
'''
                        if transport.cost:
                            html_content += f'''
            <p class="price"><strong>Cost:</strong> {transport.currency} {transport.cost:.2f}</p>
'''
                        if transport.duration_minutes:
                            dur_hours = transport.duration_minutes // 60
                            dur_mins = transport.duration_minutes % 60
                            dur_str = f"{dur_hours}h {dur_mins}m" if dur_hours > 0 else f"{dur_mins}m"
                            html_content += f'''
            <p><strong>Duration:</strong> {dur_str}</p>
'''
                        html_content += '''
        </div>
'''
                
                # Notes for the day
                if day.notes:
                    html_content += f'''
        <div class="notes">
            <h4>Notes</h4>
            <p>{day.notes}</p>
        </div>
'''
                
                html_content += '''
    </div>
'''

        # Add budget section
        if plan.budget:
            html_content += '''
    <h2>Budget Summary</h2>
    <div class="card">
'''
            html_content += f'''
        <p><strong>Total Budget:</strong> {plan.budget.currency} {plan.budget.total_budget:.2f}</p>
        <p><strong>Spent:</strong> {plan.budget.currency} {plan.budget.spent:.2f}</p>
        <p><strong>Remaining:</strong> {plan.budget.currency} {plan.budget.remaining:.2f}</p>
'''
            
            # Budget breakdown
            if plan.budget.breakdown and len(plan.budget.breakdown) > 0:
                html_content += '''
        <h3>Budget Breakdown</h3>
        <table>
            <tr>
                <th>Category</th>
                <th>Amount</th>
            </tr>
'''
                for category, amount in plan.budget.breakdown.items():
                    html_content += f'''
            <tr>
                <td>{category}</td>
                <td>{plan.budget.currency} {amount:.2f}</td>
            </tr>
'''
                html_content += '''
        </table>
'''
            
            # Saving recommendations
            if plan.budget.saving_recommendations and len(plan.budget.saving_recommendations) > 0:
                html_content += '''
        <h3>Saving Recommendations</h3>
        <ul>
'''
                for rec in plan.budget.saving_recommendations:
                    html_content += f'''
            <li>{rec}</li>
'''
                html_content += '''
        </ul>
'''
            
            html_content += '''
    </div>
'''

        # Add recommendations
        if plan.recommendations and len(plan.recommendations) > 0:
            html_content += '''
    <h2>Recommendations</h2>
'''
            for rec in plan.recommendations:
                html_content += f'''
    <div class="recommendation">{rec}</div>
'''
        
        # Add alerts
        if plan.alerts and len(plan.alerts) > 0:
            html_content += '''
    <h2>Important Alerts</h2>
'''
            for alert in plan.alerts:
                html_content += f'''
    <div class="alert">{alert}</div>
'''
        
        # Close the HTML tags
        html_content += '''
    <footer>
        <p><small>Generated by AI Travel Planning System on ''' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '''</small></p>
    </footer>
</body>
</html>
'''
        
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
