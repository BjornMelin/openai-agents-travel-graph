# Contributing to OpenAI Agents Travel Graph 🤝

Thank you for your interest in contributing to OpenAI Agents Travel Graph! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Contributing to OpenAI Agents Travel Graph 🤝](#contributing-to-openai-agents-travel-graph-)
  - [Table of Contents](#table-of-contents)
  - [Code of Conduct](#code-of-conduct)
  - [Getting Started](#getting-started)
  - [Development Workflow](#development-workflow)
  - [Coding Standards](#coding-standards)
    - [Python Style Guide](#python-style-guide)
    - [Documentation](#documentation)
  - [Pull Request Process](#pull-request-process)
  - [Reporting Bugs](#reporting-bugs)
  - [Feature Requests](#feature-requests)
  - [Community](#community)
  - [License](#license)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

To get started with development:

1. Fork the repository
2. Clone your fork:

   ```bash
   git clone https://github.com/YOUR_USERNAME/openai-agents-travel-graph.git
   cd openai-agents-travel-graph
   ```

3. Set up the development environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate

   # Install uv package manager
   curl -sSf https://astral.sh/uv/install.sh | bash

   # Install dependencies
   uv pip install -r requirements.txt
   uv pip install -r requirements-dev.txt
   ```

4. Create a branch for your work:

   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

Our development workflow follows these steps:

1. **Pick an issue** - Select an open issue to work on or create a new one
2. **Discuss** - For major changes, please open an issue first to discuss what you would like to change
3. **Branch** - Create a feature branch for your work
4. **Code** - Make your changes, following our coding standards
5. **Test** - Add tests for your changes and ensure all tests pass
6. **Document** - Update documentation as needed
7. **PR** - Submit a pull request

## Coding Standards

We follow these principles in our codebase:

- **DRY (Don't Repeat Yourself)** - Avoid code duplication
- **KISS (Keep It Simple, Stupid)** - Prefer simple solutions over complex ones
- **YAGNI (You Aren't Gonna Need It)** - Don't implement features until they're needed
- **SOLID** - Follow SOLID principles for object-oriented design
- **File Size** - Keep file sizes reasonable (<300 lines) for better maintainability

### Python Style Guide

- Use [uv](https://github.com/astral-sh/uv) as the package manager and for running Python commands
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style
- Use type hints for all function parameters and return values
- Use Google-style docstrings
- Implement proper error handling with try/except blocks
- Use `uv run` to execute Python scripts and tests (e.g., `uv run pytest`, `uv run python script.py`)

### Documentation

- Keep README.md and other documentation up-to-date
- Document all public functions and classes
- Include examples in docstrings
- Add inline comments for complex logic

## Pull Request Process

1. Ensure your code follows our coding standards
2. Update documentation as needed
3. Add or update tests as appropriate
4. Ensure all tests are passing (`uv run pytest`)
5. Fill out the pull request template completely
6. Request review from maintainers
7. Address any feedback from code reviews

PRs will be merged once they:

- Pass all automated checks
- Receive approval from at least one maintainer
- Meet our code quality standards

## Reporting Bugs

When reporting bugs, please include:

- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Screenshots (if applicable)
- Environment details (OS, Python version, etc.)

Use the bug report template when creating new issues.

## Feature Requests

Feature requests are welcome! When submitting a feature request:

- Use a clear, descriptive title
- Provide a detailed description of the proposed feature
- Explain the benefit to users
- Suggest an implementation approach if possible

Use the feature request template when creating new issues.

## Community

Join our community to discuss the project:

- [GitHub Discussions](https://github.com/BjornMelin/openai-agents-travel-graph/discussions)

## License

By contributing to this project, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

Thank you for contributing to OpenAI Agents Travel Graph! 🙏
