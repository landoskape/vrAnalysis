# Multi-Session Analysis

The `vrAnalysis2.multisession` module provides tools for analyzing data across multiple sessions, enabling population-level and longitudinal analyses.

## MultiSession Class

The `MultiSession` class manages multiple sessions and provides cross-session analysis capabilities.

```python
from vrAnalysis2.multisession import MultiSession
from vrAnalysis2.database import get_database

# Get sessions from database
db = get_database("vrSessions")
sessions_data = db.get_table(mouseName="mouse001")

# Create multi-session object
multi = MultiSession(sessions_data)

# Access sessions
sessions = multi.sessions
```

## Cross-Session Analysis

Analyze data across sessions:

```python
# Analyze place field stability
stability = multi.analyze_place_field_stability()

# Analyze population activity
population_activity = multi.analyze_population()

# Compare sessions
comparison = multi.compare_sessions(session1_idx=0, session2_idx=1)
```

## See Also

- [Tracking Module](tracking.md) for cell tracking
- [Analysis Module](analysis.md) for analysis tools

