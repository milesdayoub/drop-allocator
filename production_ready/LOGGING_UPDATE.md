# Logging Update Summary

The Drop Allocator has been updated to use Python's logging module instead of print statements for production-ready logging.

## 🔄 Changes Made

### 1. **Logging Configuration Added**

- Added `import logging` to imports
- Configured logging with timestamps and clean format
- Added `--log_level` command-line argument for flexible log level control

### 2. **Print Statements Replaced**

All print statements have been replaced with appropriate logging levels:

#### **INFO Level** (General Information)

- Input loading progress
- Phase transitions in greedy algorithm
- Summary statistics
- Success confirmations

#### **WARNING Level** (Important Issues)

- Fully utilized sponsored contracts
- Unused sponsored contracts
- Warm-start hint failures
- OR-Tools skipping reasons

#### **ERROR Level** (Critical Issues)

- Sponsorship ratio violations
- Mixed sponsorship type violations
- Assignment ratio violations
- General error conditions

### 3. **Log Format**

```
2025-08-20 10:45:30 - INFO - [greedy] Phase 1: Meeting sponsorship minima for 5 groups
2025-08-20 10:45:31 - WARNING - 🚨 FULLY UTILIZED sponsored contracts (3):
2025-08-20 10:45:32 - INFO - ✅ wrote outputs/assignments.csv (total wall time 45.2s)
```

## 🎛️ New Command Line Options

### `--log_level`

Control the verbosity of logging output:

```bash
--log_level DEBUG    # Most verbose - all messages
--log_level INFO     # Default - general information
--log_level WARNING  # Only warnings and errors
--log_level ERROR    # Only errors
```

## 📊 Log Level Examples

### DEBUG Level

```bash
python3 src/allocator/main.py --log_level DEBUG [other args...]
```

- Shows all internal processing details
- Useful for debugging allocation issues
- May impact performance slightly

### INFO Level (Default)

```bash
python3 src/allocator/main.py [other args...]
```

- Shows progress through allocation phases
- Displays summary statistics
- Reports violations and warnings
- Production-ready level

### WARNING Level

```bash
python3 src/allocator/main.py --log_level WARNING [other args...]
```

- Only shows important issues
- Minimal output for high-volume runs
- Good for monitoring systems

### ERROR Level

```bash
python3 src/allocator/main.py --log_level ERROR [other args...]
```

- Only shows critical errors
- Minimal output for production monitoring
- Good for alerting systems

## 🚀 Production Benefits

### **Structured Logging**

- Consistent timestamp format
- Easy to parse with log aggregation tools
- Standard log levels for filtering

### **Performance Monitoring**

- Track allocation phases and timing
- Monitor capacity utilization
- Identify bottlenecks

### **Error Tracking**

- Structured error reporting
- Easy to set up alerts
- Better debugging in production

### **Flexible Output**

- Can redirect logs to files
- Easy integration with monitoring systems
- Configurable verbosity per environment

## 📝 Usage Examples

### Basic Production Run

```bash
python3 src/allocator/main.py \
  --caps data/contract_caps.csv \
  --elig data/elig_pairs.csv \
  --user_groups data/user_groups.csv \
  --group_ratios data/group_ratios.csv \
  --out outputs/assignments.csv \
  --solver greedy \
  --log_level INFO
```

### Debug Run

```bash
python3 src/allocator/main.py \
  --caps data/contract_caps.csv \
  --elig data/elig_pairs.csv \
  --user_groups data/user_groups.csv \
  --group_ratios data/group_ratios.csv \
  --out outputs/assignments.csv \
  --solver greedy \
  --log_level DEBUG
```

### Silent Run (Errors Only)

```bash
python3 src/allocator/main.py \
  --caps data/contract_caps.csv \
  --elig data/elig_pairs.csv \
  --user_groups data/user_groups.csv \
  --group_ratios data/group_ratios.csv \
  --out outputs/assignments.csv \
  --solver greedy \
  --log_level ERROR
```

## 🔧 Integration with Monitoring Systems

### **File Logging**

```bash
python3 src/allocator/main.py [args...] 2>&1 | tee allocation.log
```

### **Log Rotation**

```bash
python3 src/allocator/main.py [args...] 2>&1 | rotatelogs allocation_%Y%m%d.log 86400
```

### **Systemd Journal**

```bash
python3 src/allocator/main.py [args...] 2>&1 | logger -t drop-allocator
```

The allocator is now production-ready with professional logging that makes monitoring, debugging, and operations much easier! 🎉
