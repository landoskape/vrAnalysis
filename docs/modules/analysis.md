# Analysis

The `vrAnalysis2.analysis` module provides tools for analyzing VR session data, including place cell analysis, reliability metrics, and plasticity measurements.

## Analysis Modules

### Place Cell Reliability

Analyze place cell reliability:

```python
from vrAnalysis.processors.spkmaps import SpkmapProcessor

# Create processor and get reliability
processor = SpkmapProcessor(session)
maps = processor.process(bin_size=5.0, by_environment=True)
reliability = processor.get_reliability(envnum=0)

# Access reliability scores
reliability_scores = reliability.reliability
```

### Changing Place Fields

Track how place fields change across sessions:

```python
from vrAnalysis.multisession import MultiSessionSpkmaps
from vrAnalysis.tracking import Tracker

# Create multi-session object
tracker = Tracker("mouse001")
multi = MultiSessionSpkmaps(tracker)

# Get maps for same environment across sessions
envnum = 0
maps_list = multi.get_env_maps(envnum)

# Compare place fields between sessions
# (implement comparison logic based on your needs)
```

### Tracked Plasticity

Analyze plasticity in tracked cells:

```python
from vrAnalysis.tracking import Tracker

# Get tracked ROIs
tracker = Tracker("mouse001")
idx_tracked, extras = tracker.get_tracked_idx()

# Analyze changes in tracked cells
# (implement analysis based on your needs)
```

## Common Analysis Workflows

### Single Session Analysis

```python
from vrAnalysis.sessions import create_b2session
from vrAnalysis.processors.spkmaps import SpkmapProcessor

# Load session
session = create_b2session("mouse001", "2024-01-15", "001")
session.load_data()

# Generate spike maps
processor = SpkmapProcessor(session)
maps = processor.process(bin_size=5.0)

# Perform analysis
# (analysis code here)
```

### Multi-Session Analysis

```python
from vrAnalysis.multisession import MultiSessionSpkmaps
from vrAnalysis.tracking import Tracker

# Create tracker for a mouse
tracker = Tracker("mouse001")

# Create multi-session object
multi = MultiSessionSpkmaps(tracker)

# Perform cross-session analysis
# (analysis code here)
```

## See Also

- [Multi-session Analysis](multisession.md) for cross-session workflows
- [Tracking Module](tracking.md) for cell tracking
- [Processors Module](processors.md) for data processing

