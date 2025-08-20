# Production Deployment Guide

This guide covers deploying the Drop Allocator to your production environment.

## Prerequisites

- Python 3.8+ environment
- Access to production databases (PostgreSQL, ClickHouse)
- Sufficient compute resources for allocation runs

## Files to Copy

### Core Application

```
src/allocator/main.py          # Main allocation logic
src/allocator/__init__.py      # Package initialization
src/utils/                     # Utility functions
src/__init__.py               # Root package
```

### Configuration

```
requirements.txt               # Python dependencies
pyproject.toml                # Project configuration
setup.py                      # Installation script
```

### SQL Queries

```
sql/contract_caps_2122_v3.sql           # Contract capacity calculation
sql/elig_pairs_2122_include_zeros.sql   # Eligibility pairs extraction
sql/user_groups_2122.sql                 # User group assignments
sql/group_ratios_2122.sql               # Target sponsorship ratios
```

## Installation Steps

1. **Create Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   python3 -c "import allocator; print('Installation successful')"
   ```

## Database Setup

### PostgreSQL Tables Required

- `contract`: Contract information and budgets
- `contract_assignment`: User assignments and redemption status
- `payout`: Settlement records
- `conversion`: Redemption events
- `timed_drop_excluded_contract`: Drop-specific exclusions

### ClickHouse Tables Required

- Eligibility pairs with user-contract scores

## Configuration

### Environment Variables

Set these in your production environment:

```bash
export DB_HOST=your_db_host
export DB_PORT=5432
export DB_NAME=your_db_name
export DB_USER=your_db_user
export DB_PASSWORD=your_db_password
```

### Database Connection

Update SQL files with your production database connection details.

## Running in Production

### Command Structure

```bash
python3 src/allocator/main.py \
  --caps <contract_caps_file> \
  --elig <eligibility_file> \
  --user_groups <user_groups_file> \
  --group_ratios <group_ratios_file> \
  --out <output_file> \
  --solver greedy \
  --timeout 1200 \
  --cache_factor 40 \
  --sponsored_first_rounds 3 \
  --enforce_assignment_ratio \
  --assignment_ratio_guard hard \
  --ratio_slack 0.0 \
  --max_mixed_share 0.0 \
  --scarcity_alpha 0.05 \
  --log_level INFO
```

### Recommended Production Settings

- `--solver greedy`: Fast and reliable for production
- `--timeout 1200`: 20 minutes maximum runtime
- `--cache_factor 40`: Balance between speed and quality
- `--enforce_assignment_ratio`: Ensure sponsorship compliance
- `--assignment_ratio_guard hard`: Strict ratio enforcement

## Monitoring & Logging

### Output Files

- Assignments CSV: User-Contract assignments
- Summary statistics: Performance metrics and ratios

### Error Handling

- Check exit codes for success/failure
- Review timeout settings for large datasets
- Monitor memory usage during allocation

## Performance Considerations

### Memory Usage

- Large eligibility datasets may require significant RAM
- Consider chunking for very large user bases

### Runtime

- Greedy solver: Typically 5-15 minutes for 100K+ users
- ILP solvers: May take 1+ hours for complex scenarios

### Scaling

- For very large datasets, consider parallel processing
- Database query optimization may be needed

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `--cache_factor` or use smaller datasets
2. **Timeout Issues**: Increase `--timeout` or use greedy solver
3. **Database Errors**: Verify connection and table permissions
4. **Ratio Violations**: Check `--ratio_slack` and group ratio settings

### Debug Mode

Add `--or_log` for detailed OR-Tools logging when using ILP solvers.

### Logging Configuration

The allocator uses Python's logging module with configurable levels:

```bash
--log_level DEBUG    # Detailed debugging information
--log_level INFO     # General information (default)
--log_level WARNING  # Only warnings and errors
--log_level ERROR    # Only errors
```

Logs include timestamps and are formatted for easy parsing in production environments.

## Security Notes

- Ensure database credentials are properly secured
- Validate input files for data integrity
- Consider input sanitization for user-generated content
- Review file permissions on production systems

## Backup & Recovery

- Keep copies of input data files
- Archive output files for audit purposes
- Document any manual overrides or adjustments
- Test recovery procedures regularly
