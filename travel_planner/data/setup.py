"""
Setup script for initializing the database schema and tables.

This script handles the initialization of the Supabase database,
applying migrations, and setting up required tables and indexes.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from travel_planner.config import TravelPlannerConfig, initialize_config
from travel_planner.data.supabase import SupabaseClient
from travel_planner.utils.logging import get_logger, setup_logging

# Initialize logger
logger = get_logger(__name__)


async def initialize_database(config: TravelPlannerConfig, reset: bool = False) -> bool:
    """
    Initialize the Supabase database with the required schema.

    Args:
        config: Application configuration
        reset: Whether to reset the database (drop and recreate tables)

    Returns:
        True if initialization was successful, False otherwise
    """
    try:
        # Initialize Supabase client
        supabase_client = SupabaseClient(
            url=config.api.supabase_url, key=config.api.supabase_key
        )

        logger.info("Initializing Supabase database...")

        # Apply migrations
        if reset:
            await apply_migrations(supabase_client, reset=True)
        else:
            await apply_migrations(supabase_client)

        logger.info("Supabase database initialization completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error initializing Supabase database: {e}")
        return False


async def apply_migrations(client: SupabaseClient, reset: bool = False) -> None:
    """
    Apply database migrations from the migrations directory.

    Args:
        client: Supabase client
        reset: Whether to reset the database (drop and recreate tables)
    """
    # Get the migrations directory
    migrations_dir = Path(__file__).parent / "migrations"

    if not migrations_dir.exists():
        logger.warning(f"Migrations directory does not exist: {migrations_dir}")
        return

    if reset:
        # Execute reset script to drop all tables
        logger.warning("Resetting database - all data will be lost!")
        reset_sql = """
        DROP TABLE IF EXISTS travel_plans CASCADE;
        DROP TABLE IF EXISTS user_preferences CASCADE;
        DROP TABLE IF EXISTS travel_queries CASCADE;
        DROP TABLE IF EXISTS users CASCADE;
        DROP VIEW IF EXISTS travel_plan_summaries;
        DROP FUNCTION IF EXISTS search_travel_plans;
        """
        await client.client.rpc("execute_sql", {"query": reset_sql})
        logger.info("Database reset completed")

    # Get all .sql files in the migrations directory, sorted by name
    migration_files = sorted([f for f in migrations_dir.glob("*.sql")])

    if not migration_files:
        logger.warning("No migration files found")
        return

    logger.info(f"Found {len(migration_files)} migration files")

    # Apply each migration
    for migration_file in migration_files:
        logger.info(f"Applying migration: {migration_file.name}")

        # Read the SQL file
        with open(migration_file) as f:
            sql = f.read()

        # Execute the SQL
        try:
            await client.client.rpc("execute_sql", {"query": sql})
            logger.info(f"Successfully applied migration: {migration_file.name}")
        except Exception as e:
            logger.error(f"Error applying migration {migration_file.name}: {e}")
            raise


async def create_test_data(client: SupabaseClient) -> None:
    """
    Create test data in the database.

    Args:
        client: Supabase client
    """
    logger.info("Creating test data...")

    # Create a test user
    user_result = (
        await client.client.table("users")
        .insert({"email": "test@example.com"})
        .execute()
    )

    if not user_result.data or len(user_result.data) == 0:
        logger.error("Failed to create test user")
        return

    user_id = user_result.data[0]["id"]
    logger.info(f"Created test user with ID: {user_id}")

    # Create a test travel query
    query_result = (
        await client.client.table("travel_queries")
        .insert(
            {
                "user_id": user_id,
                "raw_query": "I want to visit Tokyo for a week in October",
                "destination": "Tokyo",
                "origin": "New York",
                "departure_date": "2025-10-01",
                "return_date": "2025-10-08",
                "travelers": 2,
                "budget_min": 3000,
                "budget_max": 5000,
                "purpose": "vacation",
            }
        )
        .execute()
    )

    if not query_result.data or len(query_result.data) == 0:
        logger.error("Failed to create test travel query")
        return

    query_id = query_result.data[0]["id"]
    logger.info(f"Created test travel query with ID: {query_id}")

    # Create test user preferences
    await (
        client.client.table("user_preferences")
        .insert(
            {
                "user_id": user_id,
                "travel_query_id": query_id,
                "travel_class": "economy",
                "direct_flights_only": True,
                "accommodation_types": ["hotel", "apartment"],
                "hotel_rating": 4,
                "amenities": ["wifi", "pool", "gym"],
                "transportation_modes": ["public_transit", "taxi"],
                "activity_types": ["cultural", "sightseeing", "culinary"],
            }
        )
        .execute()
    )

    logger.info("Test data creation completed successfully")


async def main(args: argparse.Namespace) -> int:
    """
    Main function for the setup script.

    Args:
        args: Command-line arguments

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Set up logging
    setup_logging(log_level=args.log_level)

    logger.info("Starting database setup")

    # Load environment variables from .env file if it exists
    if os.path.exists(".env"):
        load_dotenv()

    # Initialize configuration
    config = initialize_config(custom_config_path=args.config)

    # Validate configuration
    if not config.api.validate():
        logger.error("Missing required API keys for Supabase")
        logger.error(
            "Please ensure SUPABASE_URL and SUPABASE_KEY are set in your environment"
        )
        return 1

    # Initialize database
    success = await initialize_database(config, reset=args.reset)

    if not success:
        logger.error("Database initialization failed")
        return 1

    # Create test data if requested
    if args.test_data:
        try:
            client = SupabaseClient(
                url=config.api.supabase_url, key=config.api.supabase_key
            )
            await create_test_data(client)
        except Exception as e:
            logger.error(f"Error creating test data: {e}")
            # Continue executing even if test data creation fails

    logger.info("Database setup completed successfully")
    return 0


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Set up the Supabase database for the travel planner"
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the database (drop and recreate tables)",
    )

    parser.add_argument(
        "--test-data", action="store_true", help="Create test data in the database"
    )

    parser.add_argument("--config", type=str, help="Path to custom configuration file")

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = asyncio.run(main(args))
    sys.exit(exit_code)
