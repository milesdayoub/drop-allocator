# Drop Allocator

A Python-based optimization system for allocating drops (likely cryptocurrency or NFT distributions) with coverage encouragement and multiple solver backends.

## Overview

This project implements various allocation algorithms for distributing limited resources (drops) across users while optimizing for coverage and user preferences. It supports multiple optimization backends including Google OR-Tools, PuLP, and greedy algorithms.

## Features

- **Multiple Solver Backends**: OR-Tools (recommended), PuLP, and greedy algorithms
- **Coverage Optimization**: Configurable coverage modes with weighted penalties
- **Warm Start Support**: Uses greedy solutions as hints for better optimization
- **Flexible Input**: Supports CSV inputs for contract caps and user eligibility
- **Performance Monitoring**: Built-in timing and solution quality metrics

## Project Structure

```
├── src/                    # Source code
│   ├── allocator/         # Core allocation algorithms
│   ├── data/              # Data processing utilities
│   └── utils/             # Helper functions
├── sql/                   # Database queries
│   ├── clickhouse/        # ClickHouse queries
│   └── postgres/          # PostgreSQL queries
├── data/                  # Data files (CSV, etc.)
├── outputs/               # Generated outputs
├── tests/                 # Test suite
├── docs/                  # Documentation
├── scripts/               # Utility scripts
└── examples/              # Usage examples
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd drop-allocator
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python -m src.allocator.main \
    --caps data/contract_caps.csv \
    --elig data/elig_pairs.csv \
    --k 5 \
    --solver ortools \
    --timeout 300
```

### Key Parameters

- `--caps`: Contract capacity CSV file
- `--elig`: User eligibility pairs CSV file
- `--k`: Number of allocations per user
- `--solver`: Solver backend (ortools, pulp, greedy)
- `--timeout`: Maximum solve time in seconds
- `--cov_mode`: Coverage mode (z, shortfall)
- `--cov_w`: Coverage weight for optimization

### Input Format

**Contract Caps CSV:**

- `contract_address`: Contract identifier
- `cap_face`: Maximum capacity
- `is_sponsored`: Whether contract is sponsored

**Eligibility Pairs CSV:**

- `user_id`: User identifier
- `contract_address`: Contract identifier
- `score`: User preference score

## Solvers

### OR-Tools (Recommended)

- Fastest and most reliable
- Supports warm starts from greedy solutions
- Configurable coverage modes

### PuLP

- Fallback solver
- Good for smaller problems
- More memory intensive

### Greedy

- Fastest but lowest quality
- Useful for initial solutions
- Good for warm starts

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Quality

```bash
# Format code
black src/
isort src/

# Lint code
flake8 src/
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[Add your license here]

## Contact

[Add contact information]
