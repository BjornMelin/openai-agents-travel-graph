"""
Budget Management Agent for the travel planner system.

This module implements the specialized agent responsible for tracking,
optimizing, and managing the travel budget across all aspects of the trip.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from travel_planner.agents.base import AgentConfig, AgentContext, BaseAgent
from travel_planner.utils.error_handling import with_retry
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


class ExpenseCategory(str, Enum):
    """Categories of travel expenses."""

    FLIGHTS = "flights"
    ACCOMMODATION = "accommodation"
    TRANSPORTATION = "transportation"
    FOOD = "food"
    ACTIVITIES = "activities"
    SHOPPING = "shopping"
    MISCELLANEOUS = "miscellaneous"


@dataclass
class BudgetItem:
    """A single budget item."""

    category: ExpenseCategory
    name: str
    amount: float
    currency: str
    date: str | None = None
    description: str = ""
    is_estimate: bool = True
    is_required: bool = False
    alternatives: list[dict[str, Any]] = field(default_factory=list)

    @property
    def formatted_amount(self) -> str:
        """Get the formatted amount with currency symbol."""
        if self.currency == "USD":
            return f"${self.amount:.2f}"
        elif self.currency == "EUR":
            return f"€{self.amount:.2f}"
        else:
            return f"{self.amount:.2f} {self.currency}"


@dataclass
class BudgetAllocation:
    """Budget allocation for a category."""

    category: ExpenseCategory
    amount: float
    currency: str
    percentage: float
    items: list[BudgetItem] = field(default_factory=list)

    @property
    def total_spent(self) -> float:
        """Get the total amount spent in this category."""
        return sum(item.amount for item in self.items)

    @property
    def remaining(self) -> float:
        """Get the remaining budget for this category."""
        return self.amount - self.total_spent

    @property
    def formatted_amount(self) -> str:
        """Get the formatted amount with currency symbol."""
        if self.currency == "USD":
            return f"${self.amount:.2f}"
        elif self.currency == "EUR":
            return f"€{self.amount:.2f}"
        else:
            return f"{self.amount:.2f} {self.currency}"


@dataclass
class BudgetRecommendation:
    """A budget recommendation for an expense."""

    expense_name: str
    category: ExpenseCategory
    current_amount: float
    recommended_amount: float
    currency: str
    saving: float
    reasons: list[str] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BudgetContext(AgentContext):
    """Context for the budget management agent."""

    total_budget: float = 0.0
    currency: str = "USD"
    trip_duration_days: int = 0
    traveler_count: int = 1
    allocations: dict[ExpenseCategory, BudgetAllocation] = field(default_factory=dict)
    expenses: list[BudgetItem] = field(default_factory=list)
    recommendations: list[BudgetRecommendation] = field(default_factory=list)
    currency_conversions: dict[tuple[str, str], float] = field(default_factory=dict)
    category_preferences: dict[ExpenseCategory, int] = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)


class BudgetManagementAgent(BaseAgent[BudgetContext]):
    """
    Specialized agent for budget management.

    This agent is responsible for:
    1. Allocating the budget across different travel components
    2. Tracking expenses against the budget
    3. Optimizing spending to maximize value
    4. Providing cost-saving recommendations
    5. Generating budget reports and visualizations
    """

    def __init__(self, config: AgentConfig | None = None):
        """
        Initialize the budget management agent.

        Args:
            config: Configuration for the agent (optional)
        """
        default_config = AgentConfig(
            name="budget_management_agent",
            instructions="""
            You are a specialized budget management agent for travel planning.
            Your goal is to help travelers allocate, track, and optimize their budget
            to get the most value from their trip. Consider the traveler's preferences
            and priorities when making recommendations.
            """,
            model="gpt-4o",
            tools=[],  # No tools initially, they would be added in a real implementation
        )

        config = config or default_config
        super().__init__(config, BudgetContext)

        # Add tools for specific budget management functionality
        # These would typically be implemented as part of the full system
        # self.add_tool(convert_currency)
        # self.add_tool(compare_prices)
        # self.add_tool(find_deals)
        # self.add_tool(calculate_savings)

    async def run(
        self,
        input_data: str | list[dict[str, Any]],
        context: BudgetContext | None = None,
    ) -> Any:
        """
        Run the budget management agent with the provided input and context.

        Args:
            input_data: User input or conversation history
            context: Optional budget context

        Returns:
            Updated context and budget management results
        """
        try:
            # Initialize context if not provided
            if not context:
                context = BudgetContext()

            # Process the input
            result = await self.process(input_data, context)
            return result
        except Exception as e:
            error_msg = f"Error in budget management agent: {e!s}"
            logger.error(error_msg)
            return {"error": error_msg}

    async def process(
        self, input_data: str | list[dict[str, Any]], context: BudgetContext
    ) -> dict[str, Any]:
        """
        Process the budget management request.

        Args:
            input_data: User input or conversation history
            context: Budget context

        Returns:
            Budget management results
        """
        # Extract budget parameters if not already set
        if context.total_budget == 0.0:
            await self._extract_budget_parameters(input_data, context)

        # Create budget allocations if they don't exist
        if not context.allocations:
            await self._create_budget_allocations(context)

        # Process any new expenses
        new_expenses = await self._extract_expenses(input_data, context)
        if new_expenses:
            for expense in new_expenses:
                await self._add_expense(expense, context)

        # Generate budget recommendations
        recommendations = await self._generate_recommendations(context)
        context.recommendations = recommendations

        # Check budget alerts
        await self._check_budget_alerts(context)

        # Generate a budget report
        budget_report = await self._generate_budget_report(context)

        return {
            "context": context,
            "allocations": context.allocations,
            "expenses": context.expenses,
            "recommendations": context.recommendations,
            "alerts": context.alerts,
            "report": budget_report,
        }

    async def _extract_budget_parameters(
        self, input_data: str | list[dict[str, Any]], context: BudgetContext
    ) -> None:
        """
        Extract budget parameters from user input.

        Args:
            input_data: User input or conversation history
            context: Budget context
        """
        # In a real implementation, we would prepare prompts and extract data from user input
        # extraction_prompt = (
        #     "Please extract budget parameters from the following user input. "
        #     "Include total budget, currency, trip duration in days, traveler count, "
        #     "and any category preferences or priorities. "
        #     "Format your response as a JSON object.\n\n"
        #     "User input: {input}"
        # )
        #
        # user_input = input_data if isinstance(input_data, str) else self._get_latest_user_input(input_data)

        # Prepare messages for the model, but we don't actually use them in this demo
        # messages = [
        #     {"role": "system", "content": self.instructions},
        #     {"role": "user", "content": extraction_prompt.format(input=user_input)}
        # ]

        # In a real implementation, we would call the model and parse the JSON response
        # response = await self._call_model(messages)

        # and update the context with the extracted parameters

        # For now, we'll set some example values for demonstration
        context.total_budget = 3000.0
        context.currency = "USD"
        context.trip_duration_days = 7
        context.traveler_count = 2

        # Set category preferences (higher value means higher priority)
        context.category_preferences = {
            ExpenseCategory.ACCOMMODATION: 3,  # High priority
            ExpenseCategory.FLIGHTS: 2,  # Medium priority
            ExpenseCategory.FOOD: 3,  # High priority
            ExpenseCategory.ACTIVITIES: 2,  # Medium priority
            ExpenseCategory.TRANSPORTATION: 1,  # Lower priority
            ExpenseCategory.SHOPPING: 0,  # Lowest priority
            ExpenseCategory.MISCELLANEOUS: 0,  # Lowest priority
        }

        # Set some basic currency conversions
        context.currency_conversions = {
            ("USD", "EUR"): 0.92,
            ("EUR", "USD"): 1.09,
            ("USD", "GBP"): 0.79,
            ("GBP", "USD"): 1.27,
            ("EUR", "GBP"): 0.86,
            ("GBP", "EUR"): 1.16,
        }

    async def _create_budget_allocations(self, context: BudgetContext) -> None:
        """
        Create budget allocations across expense categories based on preferences.

        Args:
            context: Budget context
        """
        # Default allocation percentages if no preferences are specified
        default_percentages = {
            ExpenseCategory.ACCOMMODATION: 30,
            ExpenseCategory.FLIGHTS: 25,
            ExpenseCategory.FOOD: 20,
            ExpenseCategory.ACTIVITIES: 15,
            ExpenseCategory.TRANSPORTATION: 5,
            ExpenseCategory.SHOPPING: 3,
            ExpenseCategory.MISCELLANEOUS: 2,
        }

        # Adjust percentages based on preferences
        if context.category_preferences:
            # Calculate total preference points
            total_points = sum(context.category_preferences.values())
            if total_points == 0:
                # No preferences, use defaults
                adjusted_percentages = default_percentages
            else:
                # Adjust percentages based on preferences
                # Higher preference = higher percentage
                adjusted_percentages = {}
                for category in ExpenseCategory:
                    preference = context.category_preferences.get(category, 0)
                    # Base percentage from default, adjusted by preference
                    base_percentage = default_percentages.get(category, 0)
                    # Adjust percentage: preferred categories get more, others get less
                    adjustment_factor = (preference / total_points) * 2
                    adjusted_percentages[category] = max(
                        1, base_percentage * adjustment_factor
                    )

                # Normalize percentages to sum to 100
                total_percentage = sum(adjusted_percentages.values())
                for category in adjusted_percentages:
                    adjusted_percentages[category] = (
                        adjusted_percentages[category] / total_percentage
                    ) * 100
        else:
            adjusted_percentages = default_percentages

        # Create allocations based on the adjusted percentages
        for category, percentage in adjusted_percentages.items():
            amount = (context.total_budget * percentage) / 100
            context.allocations[category] = BudgetAllocation(
                category=category,
                amount=amount,
                currency=context.currency,
                percentage=percentage,
            )

    async def _extract_expenses(
        self, input_data: str | list[dict[str, Any]], context: BudgetContext
    ) -> list[BudgetItem]:
        """
        Extract expense items from user input.

        Args:
            input_data: User input or conversation history
            context: Budget context

        Returns:
            List of extracted expense items
        """
        # In a real implementation, we would prepare a specific prompt for expense extraction
        # extraction_prompt = (
        #     "Please extract expense items from the following user input. "
        #     "Include category, name, amount, currency, date, and description for each item. "
        #     "Format your response as a JSON array of expense objects.\n\n"
        #     "User input: {input}"
        # )

        # Extract the user input
        user_input = (
            input_data
            if isinstance(input_data, str)
            else self._get_latest_user_input(input_data)
        )

        # Check if the input likely contains expense information
        expense_keywords = [
            "cost",
            "price",
            "expense",
            "spend",
            "buy",
            "purchase",
            "book",
            "reserve",
            "paid",
            "$",
            "€",
            "£",
        ]
        if not any(keyword in user_input.lower() for keyword in expense_keywords):
            return []  # No expense information found

        # In a real implementation, we would prepare messages for the model
        # messages = [
        #     {"role": "system", "content": self.instructions},
        #     {"role": "user", "content": extraction_prompt.format(input=user_input)}
        # ]

        # In a real implementation, we would call the model and parse the JSON response
        # response = await self._call_model(messages)

        # and create BudgetItem objects from the extracted expenses

        # For demo purposes, we'll return some example expenses if the input seems expense-related
        new_expenses = []

        if (
            "hotel" in user_input.lower()
            or "stay" in user_input.lower()
            or "accommodation" in user_input.lower()
        ):
            new_expenses.append(
                BudgetItem(
                    category=ExpenseCategory.ACCOMMODATION,
                    name="Hotel Grand Paris",
                    amount=1200.0,
                    currency="USD",
                    date="2025-06-15",
                    description="7 nights at Hotel Grand Paris, standard room",
                    is_estimate=False,
                    is_required=True,
                )
            )

        if (
            "flight" in user_input.lower()
            or "airline" in user_input.lower()
            or "plane" in user_input.lower()
        ):
            new_expenses.append(
                BudgetItem(
                    category=ExpenseCategory.FLIGHTS,
                    name="Round-trip flights",
                    amount=850.0,
                    currency="USD",
                    date="2025-06-15",
                    description="Round-trip flights from New York to Paris",
                    is_estimate=False,
                    is_required=True,
                    alternatives=[
                        {
                            "name": "Budget airline",
                            "amount": 700.0,
                            "currency": "USD",
                            "note": "Less convenient times, 1 stop",
                        }
                    ],
                )
            )

        if (
            "tour" in user_input.lower()
            or "activity" in user_input.lower()
            or "visit" in user_input.lower()
        ):
            new_expenses.append(
                BudgetItem(
                    category=ExpenseCategory.ACTIVITIES,
                    name="Seine River Cruise",
                    amount=75.0,
                    currency="USD",
                    date="2025-06-16",
                    description="Evening Seine River cruise with dinner",
                    is_estimate=True,
                    is_required=False,
                    alternatives=[
                        {
                            "name": "Standard Seine Cruise",
                            "amount": 25.0,
                            "currency": "USD",
                            "note": "Without dinner, 1-hour tour",
                        }
                    ],
                )
            )

        return new_expenses

    async def _add_expense(self, expense: BudgetItem, context: BudgetContext) -> None:
        """
        Add an expense to the budget context.

        Args:
            expense: Expense to add
            context: Budget context
        """
        # Convert currency if needed
        converted_amount = expense.amount
        if expense.currency != context.currency:
            conversion_key = (expense.currency, context.currency)
            if conversion_key in context.currency_conversions:
                conversion_rate = context.currency_conversions[conversion_key]
                converted_amount = expense.amount * conversion_rate

                # Create a new expense with the converted amount
                expense = BudgetItem(
                    category=expense.category,
                    name=expense.name,
                    amount=converted_amount,
                    currency=context.currency,
                    date=expense.date,
                    description=expense.description,
                    is_estimate=expense.is_estimate,
                    is_required=expense.is_required,
                    alternatives=expense.alternatives,
                )

        # Add expense to the list
        context.expenses.append(expense)

        # Add to the corresponding allocation
        if expense.category in context.allocations:
            allocation = context.allocations[expense.category]
            allocation.items.append(expense)

            # Check if allocation is exceeded
            if allocation.total_spent > allocation.amount:
                context.alerts.append(
                    f"Budget alert: {expense.category.value.capitalize()} allocation exceeded by "
                    f"{(allocation.total_spent - allocation.amount):.2f} {context.currency}"
                )

    async def _generate_recommendations(
        self, context: BudgetContext
    ) -> list[BudgetRecommendation]:
        """
        Generate budget recommendations based on expenses and allocations.

        Args:
            context: Budget context

        Returns:
            List of budget recommendations
        """
        recommendations = []

        # Find overspent and surplus categories
        overspent_categories = self._find_overspent_categories(context)
        surplus_categories = self._find_surplus_categories(context)

        # Generate alternative expense recommendations
        alternative_recommendations = self._generate_alternative_recommendations(
            context, overspent_categories
        )
        recommendations.extend(alternative_recommendations)

        # Generate reallocation recommendations
        reallocation_recommendations = self._generate_reallocation_recommendations(
            context, overspent_categories, surplus_categories
        )
        recommendations.extend(reallocation_recommendations)

        return recommendations

    def _find_overspent_categories(
        self, context: BudgetContext
    ) -> list[ExpenseCategory]:
        """Find categories where spending exceeds allocation."""
        overspent_categories = []
        for category, allocation in context.allocations.items():
            if allocation.total_spent > allocation.amount:
                overspent_categories.append(category)
        return overspent_categories

    def _find_surplus_categories(self, context: BudgetContext) -> list[ExpenseCategory]:
        """Find categories with significant unused budget."""
        surplus_categories = []
        for category, allocation in context.allocations.items():
            # If more than 30% of the allocation is unspent and it's a lower priority category
            if (
                allocation.remaining > (allocation.amount * 0.3)
                and context.category_preferences.get(category, 0) <= 1
            ):
                surplus_categories.append(category)
        return surplus_categories

    def _generate_alternative_recommendations(
        self, context: BudgetContext, overspent_categories: list[ExpenseCategory]
    ) -> list[BudgetRecommendation]:
        """Generate recommendations to use cheaper alternatives."""
        recommendations = []

        # Generate recommendations for overspent categories
        for category in overspent_categories:
            allocation = context.allocations[category]

            # Look at expenses in this category
            for expense in allocation.items:
                if not expense.alternatives:
                    continue

                # Recommend the cheapest alternative
                cheapest_alt = min(expense.alternatives, key=lambda x: x["amount"])
                saving = expense.amount - cheapest_alt["amount"]

                if saving <= 0:
                    continue

                recommendations.append(
                    BudgetRecommendation(
                        expense_name=expense.name,
                        category=expense.category,
                        current_amount=expense.amount,
                        recommended_amount=cheapest_alt["amount"],
                        currency=expense.currency,
                        saving=saving,
                        reasons=[
                            f"This helps reduce overspending in {category.value}",
                            f"Alternative option: {cheapest_alt['name']}",
                            cheapest_alt.get("note", ""),
                        ],
                        alternatives=[cheapest_alt],
                    )
                )

        return recommendations

    def _generate_reallocation_recommendations(
        self,
        context: BudgetContext,
        overspent_categories: list[ExpenseCategory],
        surplus_categories: list[ExpenseCategory],
    ) -> list[BudgetRecommendation]:
        """Generate recommendations to reallocate budget between categories."""
        recommendations = []

        for surplus_category in surplus_categories:
            surplus_allocation = context.allocations[surplus_category]

            # Find highest priority overspent category to reallocate to
            target_category = self._find_highest_priority_target(
                context, overspent_categories
            )

            if not target_category:
                continue

            target_allocation = context.allocations[target_category]

            # Recommend reallocation of up to 50% of the surplus
            reallocation_amount = min(
                surplus_allocation.remaining * 0.5,
                target_allocation.total_spent - target_allocation.amount,
            )

            if reallocation_amount <= 0:
                continue

            # Create a virtual expense for the recommendation
            recommendations.append(
                BudgetRecommendation(
                    expense_name=f"Reallocate from {surplus_category.value} to {target_category.value}",
                    category=surplus_category,
                    current_amount=surplus_allocation.amount,
                    recommended_amount=surplus_allocation.amount - reallocation_amount,
                    currency=context.currency,
                    saving=0,  # Not a direct saving, but a reallocation
                    reasons=[
                        f"Surplus budget in {surplus_category.value}",
                        f"Overspending in higher priority {target_category.value}",
                        f"Reallocate {reallocation_amount:.2f} {context.currency}",
                    ],
                    alternatives=[],
                )
            )

        return recommendations

    def _find_highest_priority_target(
        self, context: BudgetContext, overspent_categories: list[ExpenseCategory]
    ) -> ExpenseCategory | None:
        """Find the highest priority overspent category."""
        target_category = None
        highest_priority = -1

        for category in overspent_categories:
            priority = context.category_preferences.get(category, 0)
            if priority > highest_priority:
                highest_priority = priority
                target_category = category

        return target_category

    async def _check_budget_alerts(self, context: BudgetContext) -> None:
        """
        Check for budget alerts and add them to the context.

        Args:
            context: Budget context
        """
        # Calculate total spent
        total_spent = sum(expense.amount for expense in context.expenses)

        # Check overall budget
        if total_spent > context.total_budget:
            context.alerts.append(
                f"Overall budget exceeded by {(total_spent - context.total_budget):.2f} {context.currency}"
            )
        elif total_spent > (context.total_budget * 0.9):
            context.alerts.append(
                f"Overall budget at {(total_spent / context.total_budget * 100):.1f}% of total "
                f"({(context.total_budget - total_spent):.2f} {context.currency} remaining)"
            )

        # Check individual category allocations
        for category, allocation in context.allocations.items():
            # Skip categories that have already triggered alerts
            if any(alert for alert in context.alerts if category.value in alert):
                continue

            # Check if allocation is close to being exceeded
            if (
                allocation.total_spent > (allocation.amount * 0.8)
                and allocation.total_spent <= allocation.amount
            ):
                context.alerts.append(
                    f"Budget warning: {category.value.capitalize()} allocation at "
                    f"{(allocation.total_spent / allocation.amount * 100):.1f}% "
                    f"({(allocation.amount - allocation.total_spent):.2f} {context.currency} remaining)"
                )

    async def _generate_budget_report(self, context: BudgetContext) -> str:
        """
        Generate a comprehensive budget report.

        Args:
            context: Budget context

        Returns:
            Budget report text
        """
        # Prepare a specific prompt for generating a report
        report_prompt = (
            "Create a comprehensive budget report for a {duration}-day trip with a total budget "
            "of {total_budget} {currency} for {traveler_count} traveler(s).\n\n"
            "Please include:\n"
            "1. An overview of the budget allocations across categories\n"
            "2. A summary of current expenses ({expense_count} items, {total_spent} {currency} spent)\n"
            "3. Budget recommendations and savings opportunities\n"
            "4. Budget alerts and warnings\n"
            "5. Remaining budget analysis\n\n"
            "{allocations_details}\n\n{expense_details}\n\n{recommendations_details}\n\n{alerts_details}"
        )

        # Calculate total spent
        total_spent = sum(expense.amount for expense in context.expenses)

        # Prepare allocation details
        allocations_details = []
        for category, allocation in context.allocations.items():
            allocations_details.append(
                f"{category.value.capitalize()}: {allocation.formatted_amount} "
                f"({allocation.percentage:.1f}% of total, "
                f"{allocation.total_spent:.2f} spent, "
                f"{allocation.remaining:.2f} remaining)"
            )

        # Prepare expense details
        expense_details = []
        for expense in context.expenses:
            expense_details.append(
                f"{expense.name}: {expense.formatted_amount} "
                f"({expense.category.value}, "
                f"{'estimate' if expense.is_estimate else 'confirmed'}, "
                f"{'required' if expense.is_required else 'optional'})"
            )

        # Prepare recommendation details
        recommendations_details = []
        for rec in context.recommendations:
            recommendations_details.append(
                f"Recommendation for {rec.expense_name}: "
                f"Change from {rec.current_amount:.2f} to {rec.recommended_amount:.2f} {rec.currency} "
                f"(Saving: {rec.saving:.2f} {rec.currency})\n"
                f"Reasons: {', '.join(rec.reasons)}"
            )

        # Prepare alert details
        alerts_details = (
            "\n".join(context.alerts)
            if context.alerts
            else "No budget alerts at this time."
        )

        messages = [
            {"role": "system", "content": self.instructions},
            {
                "role": "user",
                "content": report_prompt.format(
                    duration=context.trip_duration_days,
                    total_budget=context.total_budget,
                    currency=context.currency,
                    traveler_count=context.traveler_count,
                    expense_count=len(context.expenses),
                    total_spent=total_spent,
                    allocations_details="\n".join(allocations_details),
                    expense_details="\n".join(expense_details),
                    recommendations_details="\n".join(recommendations_details),
                    alerts_details=alerts_details,
                ),
            },
        ]

        response = await self._call_model(messages)

        # Return the generated report
        return response.get("content", "")

    def _get_latest_user_input(self, messages: list[dict[str, Any]]) -> str:
        """
        Extract the latest user input from a list of messages.

        Args:
            messages: List of message dictionaries

        Returns:
            Latest user input text
        """
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    @with_retry(max_attempts=3)
    async def _call_model(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Call the OpenAI API with the given messages.

        Args:
            messages: List of message dictionaries

        Returns:
            Model response
        """
        # Log inputs for debugging
        logger.debug(f"Calling model with messages: {messages}")

        # Call OpenAI API
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Log the response
        logger.debug(f"Model response: {response}")

        # Extract the content from the response
        content = response.choices[0].message.content
        return {"content": content}
