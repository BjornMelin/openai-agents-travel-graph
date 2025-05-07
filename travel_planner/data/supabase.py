"""
Supabase integration for data persistence in the travel planner system.

This module implements the integration with Supabase for storing and retrieving
travel planning data, enabling persistence across sessions and real-time updates.
"""

import os
from datetime import date, datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from supabase import create_client

from travel_planner.data.models import (
    TravelPlan,
    TravelQuery,
    UserPreferences,
)
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class SupabaseClient:
    """Client for interacting with Supabase."""

    def __init__(self, url: str | None = None, key: str | None = None):
        """
        Initialize Supabase client.

        Args:
            url: Supabase project URL (defaults to SUPABASE_URL environment variable)
            key: Supabase API key (defaults to SUPABASE_KEY environment variable)
        """
        self.url = url or os.environ.get("SUPABASE_URL")
        self.key = key or os.environ.get("SUPABASE_KEY")

        if not self.url or not self.key:
            raise ValueError(
                "Supabase URL and key must be provided either as arguments or "
                "as SUPABASE_URL and SUPABASE_KEY environment variables"
            )

        self.client = create_client(self.url, self.key)

    async def initialize_tables(self):
        """
        Initialize database tables if they don't exist.

        This method is deprecated in favor of the setup.py script which
        applies migrations from the migrations directory.
        Use travel_planner.data.setup.initialize_database() instead.
        """
        logger.warning(
            "SupabaseClient.initialize_tables() is deprecated. "
            "Use travel_planner.data.setup.initialize_database() instead."
        )

        # Import and use the setup module to maintain consistency
        from travel_planner.config import TravelPlannerConfig
        from travel_planner.data.setup import initialize_database

        # Create a minimal config with just the necessary API information
        config = TravelPlannerConfig()
        config.api.supabase_url = self.url
        config.api.supabase_key = self.key

        # Initialize the database using the setup module
        try:
            await initialize_database(config)
            logger.info("Successfully initialized Supabase tables")
            return True
        except Exception as e:
            logger.error(f"Error initializing Supabase tables: {e!s}")
            raise

    async def get_or_create_user(self, email: str) -> dict[str, Any]:
        """
        Get an existing user or create a new one.

        Args:
            email: User email

        Returns:
            User data
        """
        # Check if user exists
        result = (
            await self.client.table("users").select("*").eq("email", email).execute()
        )

        if result.data and len(result.data) > 0:
            return result.data[0]

        # Create new user
        result = await self.client.table("users").insert({"email": email}).execute()

        if result.data and len(result.data) > 0:
            return result.data[0]

        raise RuntimeError(f"Failed to create user for email: {email}")

    async def save_travel_query(
        self, user_id: str, query: TravelQuery
    ) -> dict[str, Any]:
        """
        Save a travel query to the database.

        Args:
            user_id: User ID
            query: Travel query to save

        Returns:
            Saved travel query data
        """
        # Convert model to dict
        query_dict = query.model_dump()

        # Format dates for PostgreSQL
        if query.departure_date:
            query_dict["departure_date"] = query.departure_date.isoformat()
        if query.return_date:
            query_dict["return_date"] = query.return_date.isoformat()

        # Handle budget range
        if query.budget_range:
            query_dict["budget_min"] = query.budget_range.get("min")
            query_dict["budget_max"] = query.budget_range.get("max")
        del query_dict["budget_range"]

        # Add user ID
        query_dict["user_id"] = user_id

        # Save to database
        result = await self.client.table("travel_queries").insert(query_dict).execute()

        if result.data and len(result.data) > 0:
            return result.data[0]

        raise RuntimeError("Failed to save travel query")

    async def save_user_preferences(
        self, user_id: str, travel_query_id: str, preferences: UserPreferences
    ) -> dict[str, Any]:
        """
        Save user preferences to the database.

        Args:
            user_id: User ID
            travel_query_id: Travel query ID
            preferences: User preferences to save

        Returns:
            Saved user preferences data
        """
        # Convert model to dict
        preferences_dict = preferences.model_dump()

        # Add user and query IDs
        preferences_dict["user_id"] = user_id
        preferences_dict["travel_query_id"] = travel_query_id

        # Save to database
        result = (
            await self.client.table("user_preferences")
            .insert(preferences_dict)
            .execute()
        )

        if result.data and len(result.data) > 0:
            return result.data[0]

        raise RuntimeError("Failed to save user preferences")

    async def save_travel_plan(
        self, user_id: str, travel_query_id: str, plan: TravelPlan
    ) -> dict[str, Any]:
        """
        Save a travel plan to the database.

        Args:
            user_id: User ID
            travel_query_id: Travel query ID
            plan: Travel plan to save

        Returns:
            Saved travel plan data
        """
        # Convert model to dict
        plan_dict = plan.model_dump()

        # Add user and query IDs
        plan_dict["user_id"] = user_id
        plan_dict["travel_query_id"] = travel_query_id

        # Save to database
        result = await self.client.table("travel_plans").insert(plan_dict).execute()

        if result.data and len(result.data) > 0:
            return result.data[0]

        raise RuntimeError("Failed to save travel plan")

    async def get_travel_plan(self, plan_id: str) -> TravelPlan | None:
        """
        Get a travel plan by ID.

        Args:
            plan_id: Travel plan ID

        Returns:
            Travel plan if found, None otherwise
        """
        result = (
            await self.client.table("travel_plans")
            .select("*")
            .eq("id", plan_id)
            .execute()
        )

        if not result.data or len(result.data) == 0:
            return None

        plan_data = result.data[0]

        # Convert JSON data back to Pydantic model
        return TravelPlan.model_validate(plan_data)

    async def get_user_travel_plans(self, user_id: str) -> list[dict[str, Any]]:
        """
        Get all travel plans for a user.

        Args:
            user_id: User ID

        Returns:
            List of travel plans
        """
        result = (
            await self.client.table("travel_plans")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )

        if not result.data:
            return []

        return result.data

    async def get_travel_query(self, query_id: str) -> TravelQuery | None:
        """
        Get a travel query by ID.

        Args:
            query_id: Travel query ID

        Returns:
            Travel query if found, None otherwise
        """
        result = (
            await self.client.table("travel_queries")
            .select("*")
            .eq("id", query_id)
            .execute()
        )

        if not result.data or len(result.data) == 0:
            return None

        query_data = result.data[0]

        # Convert budget min/max back to budget_range
        if "budget_min" in query_data and "budget_max" in query_data:
            query_data["budget_range"] = {
                "min": query_data["budget_min"],
                "max": query_data["budget_max"],
            }
            del query_data["budget_min"]
            del query_data["budget_max"]

        # Convert date strings to date objects
        if query_data.get("departure_date"):
            query_data["departure_date"] = date.fromisoformat(
                query_data["departure_date"]
            )
        if query_data.get("return_date"):
            query_data["return_date"] = date.fromisoformat(query_data["return_date"])

        # Convert JSON data back to Pydantic model
        return TravelQuery.model_validate(query_data)

    async def delete_travel_plan(self, plan_id: str) -> bool:
        """
        Delete a travel plan.

        Args:
            plan_id: Travel plan ID

        Returns:
            True if deletion was successful
        """
        result = (
            await self.client.table("travel_plans").delete().eq("id", plan_id).execute()
        )

        return len(result.data) > 0

    async def update_travel_plan(
        self, plan_id: str, plan: TravelPlan
    ) -> dict[str, Any]:
        """
        Update an existing travel plan.

        Args:
            plan_id: Travel plan ID
            plan: Updated travel plan

        Returns:
            Updated travel plan data
        """
        # Convert model to dict
        plan_dict = plan.model_dump()

        # Update timestamp
        plan_dict["updated_at"] = datetime.now().isoformat()

        # Update in database
        result = (
            await self.client.table("travel_plans")
            .update(plan_dict)
            .eq("id", plan_id)
            .execute()
        )

        if result.data and len(result.data) > 0:
            return result.data[0]

        raise RuntimeError("Failed to update travel plan")

    async def search_travel_plans(
        self,
        user_id: str | None = None,
        destination: str | None = None,
        date_range: dict[str, date] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for travel plans with filters.

        Args:
            user_id: Filter by user ID
            destination: Filter by destination
            date_range: Filter by date range

        Returns:
            List of matching travel plans
        """
        query = self.client.table("travel_plans").select("*")

        if user_id:
            query = query.eq("user_id", user_id)

        if destination:
            # Use LIKE for partial matching on destination
            query = query.like("destination->name", f"%{destination}%")

        if date_range:
            # Join with travel_queries to filter by dates
            # This requires a more complex query using raw SQL
            pass

        result = await query.execute()

        if not result.data:
            return []

        return result.data

    async def create_realtime_subscription(
        self, table: str, callback: callable, user_id: str | None = None
    ) -> dict[str, Any]:
        """
        Create a real-time subscription to database changes.

        Args:
            table: Table to subscribe to
            callback: Callback function to handle changes
            user_id: Optional user ID to filter changes

        Returns:
            Subscription details
        """
        channel = self.client.channel("db-changes")

        if user_id:
            # Filter changes to only those for a specific user
            channel = channel.on(
                "postgres_changes",
                {
                    "event": "*",
                    "schema": "public",
                    "table": table,
                    "filter": f"user_id=eq.{user_id}",
                },
                callback,
            )
        else:
            # Subscribe to all changes
            channel = channel.on(
                "postgres_changes",
                {"event": "*", "schema": "public", "table": table},
                callback,
            )

        channel.subscribe()

        return {"channel": channel, "table": table}

    def close_subscription(self, subscription: dict[str, Any]):
        """
        Close a real-time subscription.

        Args:
            subscription: Subscription details returned by create_realtime_subscription
        """
        if "channel" in subscription:
            subscription["channel"].unsubscribe()


class SupabaseRepository(Generic[T]):
    """Generic repository for Supabase operations."""

    def __init__(
        self, supabase_client: SupabaseClient, table_name: str, model_class: type[T]
    ):
        """
        Initialize repository.

        Args:
            supabase_client: Supabase client
            table_name: Table name
            model_class: Pydantic model class
        """
        self.client = supabase_client
        self.table_name = table_name
        self.model_class = model_class

    async def create(self, item: T, **extra_fields) -> T:
        """
        Create a new item.

        Args:
            item: Item to create
            extra_fields: Additional fields to include

        Returns:
            Created item
        """
        # Convert model to dict
        item_dict = item.model_dump()

        # Add extra fields
        item_dict.update(extra_fields)

        # Save to database
        result = (
            await self.client.client.table(self.table_name).insert(item_dict).execute()
        )

        if result.data and len(result.data) > 0:
            return self.model_class.model_validate(result.data[0])

        raise RuntimeError(f"Failed to create {self.model_class.__name__}")

    async def get_by_id(self, item_id: str) -> T | None:
        """
        Get an item by ID.

        Args:
            item_id: Item ID

        Returns:
            Item if found, None otherwise
        """
        result = (
            await self.client.client.table(self.table_name)
            .select("*")
            .eq("id", item_id)
            .execute()
        )

        if not result.data or len(result.data) == 0:
            return None

        return self.model_class.model_validate(result.data[0])

    async def update(self, item_id: str, item: T) -> T:
        """
        Update an item.

        Args:
            item_id: Item ID
            item: Updated item

        Returns:
            Updated item
        """
        # Convert model to dict
        item_dict = item.model_dump()

        # Update timestamp
        item_dict["updated_at"] = datetime.now().isoformat()

        # Update in database
        result = (
            await self.client.client.table(self.table_name)
            .update(item_dict)
            .eq("id", item_id)
            .execute()
        )

        if result.data and len(result.data) > 0:
            return self.model_class.model_validate(result.data[0])

        raise RuntimeError(f"Failed to update {self.model_class.__name__}")

    async def delete(self, item_id: str) -> bool:
        """
        Delete an item.

        Args:
            item_id: Item ID

        Returns:
            True if deletion was successful
        """
        result = (
            await self.client.client.table(self.table_name)
            .delete()
            .eq("id", item_id)
            .execute()
        )

        return len(result.data) > 0

    async def list(self, filter_dict: dict[str, Any] | None = None) -> list[T]:
        """
        List items with optional filtering.

        Args:
            filter_dict: Filtering criteria

        Returns:
            List of matching items
        """
        query = self.client.client.table(self.table_name).select("*")

        if filter_dict:
            for key, value in filter_dict.items():
                query = query.eq(key, value)

        result = await query.execute()

        if not result.data:
            return []

        return [self.model_class.model_validate(item) for item in result.data]
