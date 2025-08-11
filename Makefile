.PHONY: help install install-dev test lint format clean build publish

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in development mode
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"

test: ## Run tests
	python -m pytest tests/ -v

test-cov: ## Run tests with coverage
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

lint: ## Run linting
	flake8 src/ tests/
	mypy src/

format: ## Format code
	black src/ tests/
	isort src/ tests/

format-check: ## Check code formatting
	black --check src/ tests/
	isort --check-only src/ tests/

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean ## Build the package
	python setup.py sdist bdist_wheel

publish: build ## Publish to PyPI (requires twine)
	twine upload dist/*

run-example: ## Run a basic example
	python -m src.allocator.main --help

setup-venv: ## Set up virtual environment
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "source venv/bin/activate  # On Unix/macOS"
	@echo "venv\\Scripts\\activate     # On Windows"

requirements: ## Generate requirements.txt from pyproject.toml
	pip-compile pyproject.toml --output-file=requirements.txt

requirements-dev: ## Generate requirements-dev.txt
	pip-compile pyproject.toml --extra=dev --output-file=requirements-dev.txt
