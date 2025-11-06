# Tracking

The `vrAnalysis.tracking` module provides functionality for tracking cells across multiple sessions, enabling longitudinal analysis of individual neurons.

## Cell Tracking

Track the same cells across sessions:

```python
from vrAnalysis.tracking import Tracker

# Create tracker for a mouse (automatically loads all tracked sessions)
tracker = Tracker("mouse001")

# Get tracked ROIs across specific sessions
idx_tracked, extras = tracker.get_tracked_idx(
    idx_ses=[0, 1],  # Track between first two sessions
    use_session_filters=True,
    keep_method="any"  # Keep ROI if valid in any session, or "all" for all sessions
)

# idx_tracked is a (num_sessions, num_tracked_rois) array
# Each column represents a tracked ROI across sessions
for roi_idx in range(idx_tracked.shape[1]):
    session1_roi = idx_tracked[0, roi_idx]
    session2_roi = idx_tracked[1, roi_idx]
    print(f"ROI {session1_roi} in session 1 matches ROI {session2_roi} in session 2")

# Access tracking metadata
cluster_ids = extras["cluster_ids"]
sample_silhouettes = extras["sample_silhouettes"]
cluster_silhouettes = extras["cluster_silhouettes"]
```

## Tracking Methods

Tracking uses ROICaT clustering to match ROIs across sessions:

```python
# Get cluster index for a specific cluster ID
cluster_id = 5
cluster_idx = tracker.get_cluster_idx(cluster_id)
# Returns array with ROI index in each session, or -1 if not present

# Access tracking files directly
labels = tracker.labels  # List of label arrays, one per session
sample_silhouettes = tracker.sample_silhouettes  # Sample silhouettes per session
cluster_silhouettes = tracker.cluster_silhouettes  # Cluster silhouettes (shared)
```

## Multi-Session Tracking

Track cells across multiple sessions using MultiSessionSpkmaps:

```python
from vrAnalysis.multisession import MultiSessionSpkmaps
from vrAnalysis.tracking import Tracker

# Create tracker
tracker = Tracker("mouse001")

# Create multi-session object for spike map analysis
multi = MultiSessionSpkmaps(tracker)

# Get tracked ROIs across all sessions
idx_tracked, extras = tracker.get_tracked_idx(
    idx_ses=None  # Use all sessions
)

# Access tracking chains
num_tracked = idx_tracked.shape[1]
print(f"Found {num_tracked} tracked ROIs across {len(tracker.sessions)} sessions")
```

## See Also

- [Multi-session Analysis](multisession.md) for cross-session workflows
- [Analysis Module](analysis.md) for tracked cell analysis

