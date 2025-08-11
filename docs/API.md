# Drop Allocator API Documentation

## Overview

The Drop Allocator provides a Python API for solving allocation problems with coverage optimization. It supports multiple solver backends and various optimization strategies.

## Core Functions

### `greedy(caps, elig, k, seed=42)`

Fast greedy allocation algorithm.

**Parameters:**

- `caps` (pd.DataFrame): Contract capacity data
- `elig` (pd.DataFrame): User eligibility data
- `k` (int): Allocations per user
- `seed` (int): Random seed for reproducibility

**Returns:**

- `pd.DataFrame`: Allocation assignments

**Example:**

```python
from src.allocator.main import greedy

result = greedy(caps_df, elig_df, k=5, seed=42)
```

### `ilp_ortools(caps, elig, k, timeout, or_workers, or_log, cov_mode, cov_w_str, shortfall_penalty, warm_start)`

OR-Tools-based integer linear programming solver.

**Parameters:**

- `caps` (pd.DataFrame): Contract capacity data
- `elig` (pd.DataFrame): User eligibility data
- `k` (int): Allocations per user
- `timeout` (int): Maximum solve time in seconds
- `or_workers` (int): Number of worker threads
- `or_log` (bool): Enable solver logging
- `cov_mode` (str): Coverage mode ('z' or 'shortfall')
- `cov_w_str` (str): Coverage weight string
- `shortfall_penalty` (float): Penalty for coverage shortfall
- `warm_start` (bool): Use greedy solution as warm start

**Returns:**

- `pd.DataFrame`: Optimal allocation assignments

**Example:**

```python
from src.allocator.main import ilp_ortools

result = ilp_ortools(
    caps_df, elig_df, k=5,
    timeout=300,
    or_workers=4,
    cov_mode='z',
    cov_w_str='1.0',
    warm_start=True
)
```

### `ilp_pulp(caps, elig, k, timeout, cov_w_str)`

PuLP-based integer linear programming solver.

**Parameters:**

- `caps` (pd.DataFrame): Contract capacity data
- `elig` (pd.DataFrame): User eligibility data
- `k` (int): Allocations per user
- `timeout` (int): Maximum solve time in seconds
- `cov_w_str` (str): Coverage weight string

**Returns:**

- `pd.DataFrame`: Optimal allocation assignments

### `summarize(df_assign, elig, k)`

Generate summary statistics for allocation results.

**Parameters:**

- `df_assign` (pd.DataFrame): Allocation assignments
- `elig` (pd.DataFrame): Original eligibility data
- `k` (int): Target allocations per user

**Returns:**

- `dict`: Summary statistics including:
  - `n_users`: Number of users
  - `dist`: Distribution of allocations per user
  - `fill_rate`: Overall fill rate

## Data Formats

### Contract Capacity Data (`caps`)

Required columns:

- `contract_address` (str): Unique contract identifier
- `cap_face` (int): Maximum capacity
- `is_sponsored` (bool): Whether contract is sponsored

**Example:**

```csv
contract_address,cap_face,is_sponsored
0x1234...,100,True
0x5678...,50,False
```

### Eligibility Data (`elig`)

Required columns:

- `user_id` (str): Unique user identifier
- `contract_address` (str): Contract identifier
- `score` (float): User preference score

**Example:**

```csv
user_id,contract_address,score
U001,0x1234...,0.95
U001,0x5678...,0.87
U002,0x1234...,0.76
```

## Coverage Modes

### Z-Mode (`cov_mode='z'`)

Uses binary coverage variables `z_{u,t}` for each user-time combination.

**Advantages:**

- More flexible coverage modeling
- Better optimization potential

**Disadvantages:**

- More variables and constraints
- Slower solve times

### Shortfall Mode (`cov_mode='shortfall'`)

Penalizes coverage shortfalls directly in the objective function.

**Advantages:**

- Faster solve times
- Simpler model structure

**Disadvantages:**

- Less flexible coverage control
- May produce suboptimal solutions

## Configuration

The allocator can be configured using a YAML configuration file:

```yaml
allocation:
  default_k: 5
  default_timeout: 300

solver:
  default: 'ortools'
  ortools:
    workers: 4
    max_memory_mb: 2048
```

## Error Handling

The allocator includes comprehensive error handling:

- **Input Validation**: Checks for required columns and data types
- **Solver Failures**: Graceful fallback to alternative solvers
- **Memory Limits**: Configurable memory constraints
- **Timeout Handling**: Respects user-specified time limits

## Performance Tips

1. **Use Warm Starts**: Enable `warm_start=True` for better performance
2. **Filter Data**: Use `min_score` and `top_n` to reduce problem size
3. **Solver Selection**: Use OR-Tools for large problems, PuLP for small ones
4. **Coverage Mode**: Use shortfall mode for faster solves, z-mode for better quality

## Examples

See the `examples/` directory for complete working examples:

- `basic_allocation.py`: Simple allocation example
- `coverage_optimization.py`: Coverage-focused optimization
- `performance_comparison.py`: Solver performance comparison
