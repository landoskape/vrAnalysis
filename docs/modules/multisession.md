# Multi-Session Analysis

The `vrAnalysis.multisession` module provides tools for analyzing data across multiple sessions, enabling population-level and longitudinal analyses.

## MultiSessionSpkmaps Class

The `MultiSessionSpkmaps` class manages multiple sessions and provides cross-session spike map analysis capabilities.

```python
from vrAnalysis.multisession import MultiSessionSpkmaps
from vrAnalysis.tracking import Tracker

# Create tracker for a mouse
tracker = Tracker("mouse001")

# Create multi-session object
multi = MultiSessionSpkmaps(tracker)

# Access processors (one per session)
processors = multi.processors

# Access sessions
sessions = tracker.sessions
```

## Cross-Session Analysis

Analyze data across sessions:

```python
# Get spike maps for a specific environment across sessions
envnum = 0
maps_list = multi.get_env_maps(envnum)

# Get reliability across sessions
reliability = multi.get_reliability(envnum)

# Get tracked ROIs
idx_tracked, extras = tracker.get_tracked_idx()
```

## See Also

- [Tracking Module](tracking.md) for cell tracking
- [Analysis Module](analysis.md) for analysis tools

