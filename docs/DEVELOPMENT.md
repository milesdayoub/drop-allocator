# Development Guide

This guide covers how to set up the development environment, run tests, and contribute to the Drop Allocator project.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip
- git

### Initial Setup

1. **Clone the repository:**

```bash
git clone <repository-url>
cd drop-allocator
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode:**

```bash
pip install -e ".[dev]"
```

### Alternative Setup with Make

```bash
make setup-venv
source venv/bin/activate
make install-dev
```

## Development Workflow

### Code Quality Tools

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing

### Running Quality Checks

```bash
# Format code
make format

# Check formatting
make format-check

# Run linting
make lint

# Run tests
make test

# Run tests with coverage
make test-cov
```

### Pre-commit Hooks

Consider setting up pre-commit hooks to automatically run quality checks:

```bash
pip install pre-commit
pre-commit install
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_allocator.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Structure

- `tests/`: Test files
- `tests/test_allocator.py`: Core allocator tests
- `tests/test_utils.py`: Utility function tests
- `tests/conftest.py`: Test configuration and fixtures

### Writing Tests

Follow these guidelines:

1. **Test Naming**: Use descriptive test names that explain the expected behavior
2. **Test Isolation**: Each test should be independent
3. **Assertions**: Use specific assertions rather than generic ones
4. **Fixtures**: Use pytest fixtures for common test data

**Example:**

```python
def test_greedy_allocation_with_empty_data():
    """Test that greedy algorithm handles empty data gracefully"""
    caps = pd.DataFrame(columns=['contract_address', 'cap_face', 'is_sponsored'])
    elig = pd.DataFrame(columns=['user_id', 'contract_address', 'score'])

    result = greedy(caps, elig, k=5)

    assert result.empty
    assert list(result.columns) == ['user_id', 'contract_address']
```

## Code Style

### Python Style Guide

Follow PEP 8 with these project-specific rules:

- **Line Length**: 88 characters (Black default)
- **Import Sorting**: Use isort with Black profile
- **Type Hints**: Use type hints for all function parameters and return values

### Example

```python
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from .utils import validate_inputs


def allocate_resources(
    caps: pd.DataFrame,
    elig: pd.DataFrame,
    k: int,
    timeout: Optional[int] = None
) -> pd.DataFrame:
    """
    Allocate resources using the specified algorithm.

    Args:
        caps: Contract capacity data
        elig: User eligibility data
        k: Allocations per user
        timeout: Maximum solve time in seconds

    Returns:
        DataFrame with allocation assignments

    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    validate_inputs(caps, elig, k)

    # Implementation here
    pass
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def complex_function(param1: str, param2: int) -> bool:
    """Short description of function.

    Longer description if needed.

    Args:
        param1: Description of first parameter
        param2: Description of second parameter

    Returns:
        Description of return value

    Raises:
        ValueError: When something goes wrong

    Example:
        >>> complex_function("test", 42)
        True
    """
    pass
```

### API Documentation

- Keep `docs/API.md` up to date
- Include examples for all public functions
- Document data formats and requirements

## Contributing

### Pull Request Process

1. **Create a feature branch:**

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes:**

   - Write code following the style guide
   - Add tests for new functionality
   - Update documentation as needed

3. **Run quality checks:**

```bash
make format
make lint
make test
```

4. **Commit your changes:**

```bash
git add .
git commit -m "Add feature: brief description"
```

5. **Push and create PR:**

```bash
git push origin feature/your-feature-name
```

### Commit Message Format

Use conventional commit format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Maintenance tasks

**Examples:**

```
feat: add support for custom coverage weights
fix: resolve memory leak in OR-Tools solver
docs: update API documentation with examples
```

## Performance Considerations

### Optimization Guidelines

1. **Algorithm Selection**: Choose appropriate solver based on problem size
2. **Data Preprocessing**: Filter data early to reduce problem size
3. **Memory Management**: Monitor memory usage for large problems
4. **Warm Starts**: Use greedy solutions to improve optimization

### Profiling

Use profiling tools to identify bottlenecks:

```bash
# Install profiling tools
pip install line_profiler memory_profiler

# Profile specific functions
python -m line_profiler your_script.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're in the virtual environment
2. **Test Failures**: Check that all dependencies are installed
3. **Memory Issues**: Reduce problem size or use different solver
4. **Performance Issues**: Profile code and optimize bottlenecks

### Getting Help

- Check existing issues on GitHub
- Review the API documentation
- Run tests to verify functionality
- Use debugging tools to investigate issues

## Release Process

### Version Management

- Use semantic versioning (MAJOR.MINOR.PATCH)
- Update version in `pyproject.toml` and `src/allocator/__init__.py`
- Create release notes for significant changes

### Building and Publishing

```bash
# Build package
make build

# Publish to PyPI (requires twine)
make publish
```

### Release Checklist

- [ ] All tests pass
- [ ] Documentation is up to date
- [ ] Version numbers are updated
- [ ] Release notes are written
- [ ] Package builds successfully
- [ ] Release is tagged in git
