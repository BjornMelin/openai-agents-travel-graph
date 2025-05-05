"""
Main entry point for the Travel Planner application.

This module serves as the application's entry point, initializing the necessary
components and providing a CLI interface for interacting with the travel planning system.
"""

import os
import sys
import argparse
import asyncio
from typing import Dict, Any, Optional

from loguru import logger

from travel_planner.config import initialize_config, config
from travel_planner.utils import setup_logging
from travel_planner.agents import (
    OrchestratorAgent,
    DestinationResearchAgent,
    FlightSearchAgent,
)


def setup_argparse() -> argparse.ArgumentParser:
    """
    Set up the argument parser for the CLI.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="AI Travel Planning System powered by OpenAI Agents SDK and LangGraph"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to write log file (optional)",
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Initial travel query to start planning",
    )
    
    return parser


async def run_interactive_mode() -> None:
    """
    Run the travel planner in interactive mode, allowing users to have a conversation.
    """
    logger.info("Starting interactive travel planning session")
    
    # Initialize the orchestrator agent
    orchestrator = OrchestratorAgent()
    
    print("\n=== AI Travel Planning System ===")
    print("Welcome! Describe your travel plans and preferences.")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    context = None
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "q", "bye"]:
            print("\nThank you for using the Travel Planner. Goodbye!")
            break
        
        try:
            # Process the user input
            result = await orchestrator.run(user_input, context)
            
            # Update the context for the next round
            context = result.get("context")
            
            # Display the response
            response = result.get("response", {})
            if isinstance(response, dict) and "content" in response:
                print(f"\nTravel Planner: {response['content']}")
            else:
                print(f"\nTravel Planner: {response}")
                
        except Exception as e:
            logger.error(f"Error in interactive session: {str(e)}")
            print(f"\nTravel Planner: I'm sorry, I encountered an error: {str(e)}")


async def run_query_mode(query: str) -> Dict[str, Any]:
    """
    Run the travel planner with a single query and return the results.
    
    Args:
        query: Travel query to process
        
    Returns:
        Results of the travel planning process
    """
    logger.info(f"Starting travel planning for query: {query}")
    
    # Initialize the orchestrator agent
    orchestrator = OrchestratorAgent()
    
    # Process the query
    result = await orchestrator.run(query)
    
    return result


def main() -> int:
    """
    Main entry point function.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Initialize configuration
    system_config = initialize_config()
    
    # Setup logging
    setup_logging(
        log_level=args.log_level or system_config.system.log_level,
        log_file=args.log_file,
    )
    
    logger.info("Starting AI Travel Planning System")
    
    try:
        if args.query:
            # Run in query mode
            result = asyncio.run(run_query_mode(args.query))
            
            # Print the result
            if result and "response" in result:
                response = result["response"]
                if isinstance(response, dict) and "content" in response:
                    print(response["content"])
                else:
                    print(response)
        else:
            # Run in interactive mode
            asyncio.run(run_interactive_mode())
        
        return 0
    except KeyboardInterrupt:
        logger.info("Travel planning session interrupted by user")
        print("\nTravel planning session interrupted. Goodbye!")
        return 0
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())