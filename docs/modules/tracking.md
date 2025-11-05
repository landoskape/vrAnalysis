# Tracking

The `vrAnalysis2.tracking` module provides functionality for tracking cells across multiple sessions, enabling longitudinal analysis of individual neurons.

## Cell Tracking

Track the same cells across sessions:

```python
from vrAnalysis2.tracking import TrackedPair
from vrAnalysis2.sessions import create_b2session

# Load two sessions
session1 = create_b2session("mouse001", "2024-01-15", "001")
session2 = create_b2session("mouse001", "2024-01-16", "001")

# Create tracked pair
tracked = TrackedPair(session1, session2)

# Get matched pairs
matched_pairs = tracked.get_matched_pairs()

# Access matched ROIs
for roi1_idx, roi2_idx in matched_pairs:
    print(f"ROI {roi1_idx} in session 1 matches ROI {roi2_idx} in session 2")
```

## Tracking Methods

Tracking uses spatial correlation and other metrics to match ROIs:

```python
# Get tracking metrics
metrics = tracked.get_tracking_metrics()

# Access correlation scores
correlations = metrics["correlation"]

# Access distance scores
distances = metrics["distance"]
```

## Multi-Session Tracking

Track cells across multiple sessions:

```python
from vrAnalysis2.multisession import MultiSession

# Create multi-session object
multi = MultiSession(sessions_data)

# Track cells across all sessions
tracked_cells = multi.track_cells()

# Access tracking chains
for chain in tracked_cells:
    # chain contains ROI indices for each session
    print(f"Cell tracked across {len(chain)} sessions")
```

## See Also

- [Multi-session Analysis](multisession.md) for cross-session workflows
- [Analysis Module](analysis.md) for tracked cell analysis

