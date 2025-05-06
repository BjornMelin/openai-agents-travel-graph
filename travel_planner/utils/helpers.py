"""
Helper utilities for the Travel Planner system.

This module provides general utility functions used across the application.
"""

import json
import os
import re
import uuid
from collections.abc import Callable
from datetime import date, datetime, time
from typing import Any, TypeVar

import pycountry

# Type variables
T = TypeVar("T")


def generate_id(prefix: str = "") -> str:
    """
    Generate a unique ID with an optional prefix.

    Args:
        prefix: Optional prefix for the ID

    Returns:
        A unique ID string
    """
    unique_id = str(uuid.uuid4()).replace("-", "")
    if prefix:
        return f"{prefix}-{unique_id}"
    return unique_id


def generate_session_id() -> str:
    """
    Generate a unique session ID for a travel planning session.

    Returns:
        A unique session ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_part = str(uuid.uuid4())[:8]
    return f"trip-{timestamp}-{unique_part}"


def safe_serialize(obj: Any) -> Any:
    """
    Safely serialize an object to a JSON-compatible format.

    Args:
        obj: Object to serialize

    Returns:
        JSON-compatible representation of the object
    """
    # Handle different types of objects
    if obj is None or isinstance(obj, str | int | float | bool):
        return obj

    if isinstance(obj, datetime | date | time):
        return obj.isoformat()

    if isinstance(obj, list):
        return [safe_serialize(item) for item in obj]

    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}

    # Try to convert to dict if object has __dict__, otherwise use string representation
    result = safe_serialize(obj.__dict__) if hasattr(obj, "__dict__") else str(obj)
    return result


def safe_load_json(
    json_str: str, default: T | None = None
) -> dict[str, Any] | list[Any] | T:
    """
    Safely load a JSON string, returning a default value if parsing fails.

    Args:
        json_str: JSON string to parse
        default: Default value to return if parsing fails

    Returns:
        Parsed JSON data or default value
    """
    if not json_str:
        return default or {}

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return default or {}


def ensure_dir(directory: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory: Directory path

    Returns:
        Absolute path to the directory
    """
    abs_path = os.path.abspath(directory)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def get_country_code(country_name: str) -> str | None:
    """
    Get the ISO 3166-1 alpha-2 country code for a country name.

    Args:
        country_name: Country name

    Returns:
        ISO 3166-1 alpha-2 country code or None if not found
    """
    try:
        country = pycountry.countries.search_fuzzy(country_name)[0]
        return country.alpha_2
    except (LookupError, IndexError):
        return None


def get_country_name(country_code: str) -> str | None:
    """
    Get the country name for an ISO 3166-1 alpha-2 country code.

    Args:
        country_code: ISO 3166-1 alpha-2 country code

    Returns:
        Country name or None if not found
    """
    try:
        country = pycountry.countries.get(alpha_2=country_code)
        if country:
            return country.name
        return None
    except (LookupError, AttributeError):
        return None


def get_currency_symbol(currency_code: str) -> str:
    """
    Get the currency symbol for a currency code.

    Args:
        currency_code: ISO 4217 currency code

    Returns:
        Currency symbol or original code if not found
    """
    currency_symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
        "CNY": "¥",
        "AUD": "A$",
        "CAD": "C$",
        "CHF": "Fr",
        "HKD": "HK$",
        "NZD": "NZ$",
        # Add more as needed
    }
    return currency_symbols.get(currency_code, currency_code)


def format_price(amount: float, currency: str = "USD", decimal_places: int = 2) -> str:
    """
    Format a price with the appropriate currency symbol.

    Args:
        amount: Price amount
        currency: ISO 4217 currency code
        decimal_places: Number of decimal places to show

    Returns:
        Formatted price string
    """
    symbol = get_currency_symbol(currency)
    format_str = f"{symbol}{amount:.{decimal_places}f}"

    # Special handling for JPY, remove decimal places
    if currency in ["JPY"]:
        format_str = f"{symbol}{int(amount)}"

    return format_str


def extract_dates(text: str) -> list[datetime]:
    """
    Extract dates from text using regular expressions.

    Args:
        text: Text to extract dates from

    Returns:
        List of datetime objects
    """
    # Implement date extraction logic with regex
    # This is a simplified implementation that would need to be expanded
    date_patterns = [
        r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})",  # MM/DD/YYYY or DD/MM/YYYY
        r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})",  # YYYY/MM/DD
    ]

    dates = []
    for pattern in date_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            try:
                # Check for 4-digit year at the beginning (YYYY/MM/DD)
                year_length = 4
                if len(match.group(1)) == year_length:
                    year, month, day = match.groups()
                else:  # MM/DD/YYYY or DD/MM/YYYY
                    month, day, year = match.groups()

                date_obj = datetime(int(year), int(month), int(day))
                dates.append(date_obj)
            except (ValueError, TypeError):
                continue

    return dates


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length, adding a suffix if truncated.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def retry_with_fallback(
    primary_func: Callable[..., T],
    fallback_func: Callable[..., T],
    max_attempts: int = 3,
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Retry a primary function with a fallback function if all retries fail.

    Args:
        primary_func: Primary function to try
        fallback_func: Fallback function to use if primary fails
        max_attempts: Maximum number of attempts for primary function
        *args: Arguments to pass to both functions
        **kwargs: Keyword arguments to pass to both functions

    Returns:
        Result of primary function or fallback function
    """
    for attempt in range(max_attempts):
        try:
            return primary_func(*args, **kwargs)
        except Exception:
            if attempt == max_attempts - 1:
                # Last attempt failed, try fallback
                return fallback_func(*args, **kwargs)
            # Otherwise continue to next attempt
            continue


def is_valid_email(email: str) -> bool:
    """
    Check if a string is a valid email address.

    Args:
        email: Email address to check

    Returns:
        True if valid, False otherwise
    """
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(email_pattern, email))
