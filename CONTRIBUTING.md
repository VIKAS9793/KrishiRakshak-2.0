# Contributing to KrishiSahayak

Thank you for your interest in contributing to KrishiSahayak! We welcome contributions from the community to help improve this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Issues](#reporting-issues)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment (see below)
4. Create a new branch for your changes
5. Make your changes
6. Run tests and ensure they pass
7. Submit a pull request

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/VIKAS9793/KrishiSahayak.git
   cd KrishiSahayak
   ```

2. **Set up a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   # Install the package in development mode with all development dependencies
   pip install -e ".[dev]"
   
   # This includes:
   # - All core dependencies
   # - Development tools (black, ruff, mypy, pre-commit)
   # - Testing frameworks (pytest, pytest-cov)
   # - Documentation tools (Sphinx)
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the code style guidelines.

3. Run tests to ensure everything works:
   ```bash
   pytest
   ```

4. Commit your changes with a descriptive commit message:
   ```bash
   git add .
   git commit -m "Add your detailed description of changes"
   ```

## Submitting a Pull Request

1. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a pull request against the `main` branch of the main repository.

3. Fill out the PR template with details about your changes.

4. Wait for the CI to run and address any issues that come up.

## Reporting Issues

When reporting issues, please include:

- A clear description of the issue
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)
- Any relevant error messages or logs

## Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for static type checking

Run the following commands to ensure your code adheres to our standards:

```bash
black .
isort .
flake8
mypy .
```

## Testing

We use `pytest` for testing. To run the test suite:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

## Documentation

We use Sphinx for documentation. To build the documentation locally:

```bash
cd docs
make html
```

The built documentation will be available in `docs/_build/html`.

## License

By contributing to KrishiSahayak, you agree that your contributions will be licensed under the [MIT License](LICENSE).
