# Session Analysis Example

This example demonstrates how to perform analysis on a single session.

## Loading and Processing Data

```python
from vrAnalysis.sessions import B2Session
from vrAnalysis.processors.spkmaps import SpkmapProcessor

# Create session
session = B2Session.create(
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
from vrAnalysis.processors.spkmaps import SpkmapProcessor

# Generate spike maps and reliability
processor = SpkmapProcessor(session)
maps = processor.process(bin_size=5.0, by_environment=True)
reliability = processor.get_reliability(envnum=0)  # Get reliability for environment 0

# Access results
reliability_scores = reliability.reliability  # Array of reliability scores per ROI
```

## Visualizing Results

```python
# Plot spike maps for cells with high reliability
# (You'll need to implement plotting based on your visualization needs)
import matplotlib.pyplot as plt

# Example: plot spike map for a single ROI
roi_idx = 0
if maps.by_environment:
    spk_map = maps.spkmap[0][roi_idx]  # First environment, specific ROI
    occ_map = maps.occmap[0]
else:
    spk_map = maps.spkmap[roi_idx]
    occ_map = maps.occmap

# Create rate map (spikes per second)
rate_map = spk_map / (occ_map + 1e-6)  # Avoid division by zero

plt.imshow(rate_map, aspect='auto', origin='lower')
plt.colorbar(label='Firing rate (Hz)')
plt.title(f"ROI {roi_idx}")
plt.show()
```

