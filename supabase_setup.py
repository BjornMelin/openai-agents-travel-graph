#!/usr/bin/env python3
"""
Supabase Setup Tool

A command-line tool for initializing and managing the Supabase database
for the Travel Planner application.
"""

import argparse
import asyncio
import sys

from travel_planner.data.setup import initialize_database, create_test_data
from travel_planner.config import initialize_config
from travel_planner.data.supabase import SupabaseClient
from travel_planner.utils.logging import setup_logging, get_logger

# Initialize logger
logger = get_logger(__name__)


async def main(args: argparse.Namespace) -> int:
    """
    Main entry point for the Supabase setup tool.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Set up logging
    setup_logging(log_level=args.log_level)
    
    if args.command == "init":
        logger.info("Initializing Supabase database...")
        
        # Initialize configuration
        config = initialize_config(custom_config_path=args.config, raise_on_error=False)
        
        # Check if Supabase connection details are available
        if not config.api.supabase_url or not config.api.supabase_key:
            logger.error("Supabase URL and API key must be configured")
            logger.error("Please set SUPABASE_URL and SUPABASE_KEY environment variables")
            return 1
        
        # Initialize database
        success = await initialize_database(config, reset=args.reset)
        
        if not success:
            logger.error("Failed to initialize Supabase database")
            return 1
        
        # Create test data if requested
        if args.test_data:
            try:
                client = SupabaseClient(
                    url=config.api.supabase_url,
                    key=config.api.supabase_key
                )
                await create_test_data(client)
                logger.info("Test data created successfully")
            except Exception as e:
                logger.error(f"Error creating test data: {e}")
                # Continue execution even if test data creation fails
        
        logger.info("Supabase database initialized successfully")
        return 0
    
    elif args.command == "status":
        logger.info("Checking Supabase database status...")
        
        # Initialize configuration
        config = initialize_config(custom_config_path=args.config, raise_on_error=False)
        
        # Check connection
        try:
            client = SupabaseClient(
                url=config.api.supabase_url,
                key=config.api.supabase_key
            )
            
            # Check if tables exist
            tables_result = await client.client.rpc(
                "execute_sql", 
                {"query": "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"}
            ).execute()
            
            if not tables_result.data:
                logger.warning("No tables found in the database")
                return 1
            
            # Print tables
            print("Supabase connection successful")
            print("Tables found in database:")
            for table in tables_result.data:
                print(f"  - {table['table_name']}")
            
            # Try to count users
            users_result = await client.client.table("users").select("*", count='exact').execute()
            print(f"Users in database: {len(users_result.data)}")
            
            # Try to count travel plans
            plans_result = await client.client.table("travel_plans").select("*", count='exact').execute()
            print(f"Travel plans in database: {len(plans_result.data)}")
            
            return 0
        except Exception as e:
            logger.error(f"Error connecting to Supabase: {e}")
            return 1
    
    elif args.command == "reset":
        logger.warning("Resetting Supabase database - ALL DATA WILL BE LOST!")
        
        if not args.force:
            confirmation = input("Are you sure you want to reset the database? This will delete ALL data. Type 'yes' to confirm: ")
            if confirmation.lower() != "yes":
                logger.info("Database reset cancelled")
                return 0
        
        # Initialize configuration
        config = initialize_config(custom_config_path=args.config, raise_on_error=False)
        
        # Reset database
        success = await initialize_database(config, reset=True)
        
        if not success:
            logger.error("Failed to reset Supabase database")
            return 1
        
        # Create test data if requested
        if args.test_data:
            try:
                client = SupabaseClient(
                    url=config.api.supabase_url,
                    key=config.api.supabase_key
                )
                await create_test_data(client)
                logger.info("Test data created successfully")
            except Exception as e:
                logger.error(f"Error creating test data: {e}")
        
        logger.info("Supabase database reset successfully")
        return 0
    
    return 0


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Supabase database setup tool for Travel Planner"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file"
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Initialize command
    init_parser = subparsers.add_parser("init", help="Initialize the database")
    init_parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the database (drop and recreate tables)"
    )
    init_parser.add_argument(
        "--test-data",
        action="store_true",
        help="Create test data in the database"
    )
    
    # Status command
    subparsers.add_parser("status", help="Check database status")
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset the database (drop and recreate tables)")
    reset_parser.add_argument(
        "--force",
        action="store_true",
        help="Force reset without confirmation (dangerous!)"
    )
    reset_parser.add_argument(
        "--test-data",
        action="store_true",
        help="Create test data after reset"
    )
    
    # Set default command
    parser.set_defaults(command="init")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = asyncio.run(main(args))
    sys.exit(exit_code)