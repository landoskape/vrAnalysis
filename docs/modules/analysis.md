# Analysis

The `vrAnalysis2.analysis` module provides tools for analyzing VR session data, including place cell analysis, reliability metrics, and plasticity measurements.

## Analysis Modules

### Place Cell Reliability

Analyze place cell reliability across sessions:

```python
from vrAnalysis2.syd.placecell_reliability import analyze_reliability

# Analyze reliability for a session
reliability = analyze_reliability(session)
```

### Changing Place Fields

Track how place fields change across sessions:

```python
from vrAnalysis2.syd.changing_placefields import analyze_changes

# Analyze place field changes
changes = analyze_changes(session1, session2)
```

### Tracked Plasticity

Analyze plasticity in tracked cells:

```python
from vrAnalysis2.analysis.tracked_plasticity import analyze_plasticity

# Analyze plasticity for tracked pairs
plasticity = analyze_plasticity(tracked_pairs)
```

## Common Analysis Workflows

### Single Session Analysis

```python
from vrAnalysis2.sessions import create_b2session
from vrAnalysis2.processors.spkmaps import SpikeMapProcessor

# Load session
session = create_b2session("mouse001", "2024-01-15", "001")
session.load_data()

# Generate spike maps
processor = SpikeMapProcessor(session)
maps = processor.process(bin_size=5.0)

# Perform analysis
# (analysis code here)
```

### Multi-Session Analysis

```python
from vrAnalysis2.multisession import MultiSession
from vrAnalysis2.database import get_database

# Get sessions from database
db = get_database("vrSessions")
sessions_data = db.get_table(mouseName="mouse001")

# Create multi-session object
multi = MultiSession(sessions_data)

# Perform cross-session analysis
# (analysis code here)
```

## See Also

- [Multi-session Analysis](multisession.md) for cross-session workflows
- [Tracking Module](tracking.md) for cell tracking
- [Processors Module](processors.md) for data processing

