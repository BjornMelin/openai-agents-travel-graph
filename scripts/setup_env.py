#!/usr/bin/env python3
"""
Setup script for the travel planner development environment.

This script:
1. Checks if uv is installed and installs it if not
2. Creates a virtual environment
3. Installs dependencies using pyproject.toml
4. Sets up pre-commit hooks

Run with:
python scripts/setup_env.py
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Define colors for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_colored(text, color):
    """Print colored text to the terminal."""
    print(f"{color}{text}{RESET}")


def print_step(step_num, total_steps, description):
    """Print a step in the setup process."""
    print(f"\n{BOLD}{GREEN}Step {step_num}/{total_steps}: {description}{RESET}")


def run_command(command, description=None, check=True, shell=False, cwd=None):
    """Run a command and print its output."""
    if description:
        print(f"{YELLOW}{description}...{RESET}")

    if isinstance(command, str) and not shell:
        command = command.split()

    try:
        result = subprocess.run(
            command,
            check=check,
            shell=shell,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"{RED}{result.stderr}{RESET}")
        return result
    except subprocess.CalledProcessError as e:
        print(
            f"{RED}Error executing: {' '.join(command) if isinstance(command, list) else command}{RESET}"
        )
        print(f"{RED}{e}{RESET}")
        if description:
            print(f"{RED}Failed: {description}{RESET}")
        if check:
            sys.exit(1)
        return e


def check_uv_installed():
    """Check if uv is installed."""
    print_step(1, 5, "Checking if uv is installed")

    if shutil.which("uv"):
        print_colored("✅ uv is already installed", GREEN)
        return True

    print_colored("⚠️ uv is not installed", YELLOW)
    return False


def install_uv():
    """Install uv using the appropriate method for the platform."""
    system = platform.system().lower()

    if system == "linux" or system == "darwin":  # Linux or macOS
        print_colored("Installing uv using curl...", YELLOW)
        run_command(
            "curl -sSf https://astral.sh/uv/install.sh | sh",
            description="Installing uv",
            shell=True,
        )
    elif system == "windows":
        print_colored("Installing uv using PowerShell...", YELLOW)
        run_command(
            "irm https://astral.sh/uv/install.ps1 | iex",
            description="Installing uv",
            shell=True,
        )
    else:
        print_colored(
            f"❌ Unsupported platform: {system}. Please install uv manually from https://github.com/astral-sh/uv",
            RED,
        )
        sys.exit(1)

    # Add uv to PATH for the current session
    if system == "linux" or system == "darwin":
        uv_path = str(Path.home() / ".cargo" / "bin")
        os.environ["PATH"] = uv_path + os.pathsep + os.environ["PATH"]

    # Verify installation
    if not shutil.which("uv"):
        print_colored(
            "❌ Failed to install uv. Please install it manually from https://github.com/astral-sh/uv",
            RED,
        )
        sys.exit(1)

    print_colored("✅ uv installed successfully", GREEN)


def create_virtual_environment():
    """Create a virtual environment using uv."""
    print_step(2, 5, "Creating virtual environment")

    venv_path = Path(".venv")
    if venv_path.exists():
        print_colored("⚠️ Virtual environment already exists", YELLOW)
        overwrite = input("Do you want to recreate it? (y/N): ").lower() == "y"
        if overwrite:
            shutil.rmtree(venv_path)
        else:
            print_colored("✅ Using existing virtual environment", GREEN)
            return

    run_command("uv venv", description="Creating virtual environment")

    print_colored("✅ Virtual environment created successfully", GREEN)


def install_dependencies():
    """Install dependencies using uv."""
    print_step(3, 5, "Installing dependencies")

    run_command("uv pip install -e .[dev]", description="Installing dependencies")

    print_colored("✅ Dependencies installed successfully", GREEN)


def setup_pre_commit():
    """Set up pre-commit hooks."""
    print_step(4, 5, "Setting up pre-commit hooks")

    if not Path(".git").exists():
        print_colored("⚠️ Not a git repository, skipping pre-commit setup", YELLOW)
        return

    if not Path(".pre-commit-config.yaml").exists():
        print_colored("⚠️ No .pre-commit-config.yaml file found, creating one", YELLOW)
        # Create a basic pre-commit config
        pre_commit_config = """repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.2.1
    hooks:
    -   id: ruff
        args: [--fix, --exit-non-zero-on-fix]
    -   id: ruff-format

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
    -   id: mypy
        additional_dependencies: [types-requests, types-python-dateutil]
"""
        with open(".pre-commit-config.yaml", "w") as f:
            f.write(pre_commit_config)

    run_command("uv pip install pre-commit", description="Installing pre-commit")

    run_command("pre-commit install", description="Setting up pre-commit hooks")

    print_colored("✅ Pre-commit hooks set up successfully", GREEN)


def verify_setup():
    """Verify the setup by running some basic commands."""
    print_step(5, 5, "Verifying setup")

    # Verify Python version
    run_command("python --version", description="Checking Python version")

    # Verify uv version
    run_command("uv --version", description="Checking uv version")

    # Run ruff to check syntax
    run_command(
        "uv run ruff check --select=E .",
        description="Running ruff to check syntax (errors only)",
        check=False,
    )

    print_colored("\n✅ Setup completed successfully!", GREEN)
    print_colored("\nNext steps:", BOLD)

    # Determine the activate command based on the platform
    if platform.system().lower() == "windows":
        activate_cmd = ".venv\\Scripts\\activate"
    else:
        activate_cmd = "source .venv/bin/activate"

    print_colored(f"1. Activate the virtual environment: {activate_cmd}", YELLOW)
    print_colored(
        "2. Start developing! Try: uv run python -m travel_planner.main --help", YELLOW
    )


def main():
    """Main entry point for the setup script."""
    print_colored(
        f"\n{BOLD}🧳 ✈️ 🗺️ Setting up Travel Planner Development Environment 🧳 ✈️ 🗺️{RESET}\n",
        GREEN,
    )

    # Create the scripts directory if it doesn't exist
    os.makedirs("scripts", exist_ok=True)

    # Make sure we're in the project root
    if not Path("pyproject.toml").exists():
        print_colored("❌ Please run this script from the project root directory", RED)
        sys.exit(1)

    # Check if uv is installed
    if not check_uv_installed():
        install_uv()

    create_virtual_environment()
    install_dependencies()
    setup_pre_commit()
    verify_setup()


if __name__ == "__main__":
    main()
