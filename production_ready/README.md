# Drop Allocator

A Python-based optimization system for allocating drops with coverage encouragement.

## Overview

The Drop Allocator is designed to efficiently distribute Claims (coupons/offers) to users during weekly Drop events. Each user receives 3 assignments of Claims, and they can choose 1 Claim to redeem by shopping at the brand.

## Key Features

- **Greedy Solver**: Enhanced allocation algorithm with dynamic sponsorship ratio guards
- **ILP Solvers**: PuLP (CBC) and OR-Tools CP-SAT for optimal solutions
- **Coverage Encouragement**: Ensures fair distribution across user groups
- **Sponsorship Management**: Handles sponsored vs. unsponsored Claims with ratio controls
- **Scarcity Weighting**: Prioritizes under-utilized sponsored contracts

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Command

```bash
python3 src/allocator/main.py \
  --caps data/contract_caps_2122_v3.csv \
  --elig data/elig_pairs_2122_include_zeros.csv \
  --user_groups data/user_groups_2122.csv \
  --group_ratios data/group_ratios_2122.csv \
  --out outputs/assignments.csv \
  --solver greedy \
  --timeout 1200
```

### Key Parameters

- `--caps`: Contract capacity and sponsorship information
- `--elig`: User-Contract eligibility pairs with scores
- `--user_groups`: User group assignments
- `--group_ratios`: Target sponsorship ratios per group
- `--solver`: Algorithm choice (greedy, pulp, or, both)
- `--timeout`: Maximum runtime in seconds
- `--cache_factor`: Per-user cached candidates multiplier
- `--sponsored_first_rounds`: Minimum sponsored slots before unsponsored fallback
- `--enforce_assignment_ratio`: Enable assignment-level ratio guards
- `--scarcity_alpha`: Scarcity weighting for under-utilized contracts
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Input Files

### Contract Caps (`--caps`)

- `contract_address`: Contract identifier
- `cap_face`: Available capacity
- `is_sponsored`: Boolean sponsorship flag

### Eligibility Pairs (`--elig`)

- `user_id`: User identifier
- `contract_address`: Contract identifier
- `score`: Assignment preference score

### User Groups (`--user_groups`)

- `user_id`: User identifier
- `group_id`: Group assignment

### Group Ratios (`--group_ratios`)

- `group_id`: Group identifier
- `sponsorship_ratio`: Target sponsored Claim ratio

## Output

The system generates an assignments CSV with:

- User assignments to Contracts
- Sponsorship ratios per group
- Capacity utilization analysis
- Performance metrics

## Architecture

- **main.py**: Core allocation logic and solver orchestration
- **utils/**: Helper functions for validation, reporting, and diagnostics
- **sql/**: Database queries for data extraction

## Dependencies

- pandas >= 2.2.0
- numpy >= 1.26
- ortools >= 9.9 (ILP solver)
- pulp >= 2.8 (fallback solver)

## License

MIT License
