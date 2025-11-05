# Session Analysis Example

This example demonstrates how to perform analysis on a single session.

## Loading and Processing Data

```python
from vrAnalysis2.sessions import create_b2session
from vrAnalysis2.processors.spkmaps import SpkmapProcessor

# Create session
session = create_b2session(
    mouse_name="mouse001",
    date="2024-01-15",
    session_id="001"
)

# Load data
session.load_data()

# Generate spike maps
processor = SpkmapProcessor(session)
maps = processor.process(
    bin_size=5.0,
    by_environment=True,
    min_occupancy=0.1
)

# Access maps
for env_idx, env in enumerate(maps.environments):
    occ_map = maps.occmap[env_idx]
    spk_map = maps.spkmap[env_idx]
    
    # Perform analysis on maps
    # (analysis code here)
```

## Analyzing Place Cells

```python
from vrAnalysis2.syd.placecell_reliability import analyze_reliability

# Analyze place cell reliability
reliability = analyze_reliability(session)

# Access results
reliable_cells = reliability["reliable_cells"]
reliability_scores = reliability["scores"]
```

## Visualizing Results

```python
from vrAnalysis2.helpers.plotting import plot_spike_map

# Plot spike maps for reliable cells
for roi_idx in reliable_cells:
    plot_spike_map(
        maps.spkmap[roi_idx],
        maps.occmap,
        title=f"ROI {roi_idx}"
    )
```

