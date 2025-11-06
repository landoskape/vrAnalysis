# Processors

The `vrAnalysis.processors` module provides data processing pipelines that transform session data into analysis-ready formats.

## Spike Map Processor

The `SpkmapProcessor` creates spatial representations of neural activity.

### Maps Class

The `Maps` dataclass contains occupancy, speed, and spike maps:

- `occmap`: Occupancy map (time spent in each spatial bin)
- `speedmap`: Speed map (average speed in each spatial bin)
- `spkmap`: Spike map (spike count in each spatial bin)
- `by_environment`: Whether maps are separated by environment
- `rois_first`: Whether ROI dimension comes first in spkmap

### Processing Spike Maps

```python
from vrAnalysis.processors.spkmaps import SpkmapProcessor

# Create processor
processor = SpkmapProcessor(session)

# Process maps
maps = processor.process(
    bin_size=5.0,  # 5 cm bins
    by_environment=True,
    rois_first=True,
    min_occupancy=0.1  # Minimum occupancy threshold
)

# Access maps
if maps.by_environment:
    # Maps are lists, one per environment
    for env_idx, env in enumerate(maps.environments):
        occ = maps.occmap[env_idx]
        spk = maps.spkmap[env_idx]
        speed = maps.speedmap[env_idx]
else:
    # Maps are single arrays
    occ = maps.occmap
    spk = maps.spkmap
    speed = maps.speedmap
```

### Map Averaging

Average maps across ROIs or sessions:

```python
# Average across ROIs
averaged = maps.average_rois()

# Average across environments
averaged = maps.average_environments()
```

### Map Visualization

Maps can be visualized using helper functions:

```python
from vrAnalysis.helpers.plotting import plot_spike_map

# Plot spike map for a single ROI
plot_spike_map(
    maps.spkmap[roi_idx],
    maps.occmap,
    title=f"ROI {roi_idx}"
)
```

## Processing Options

### Bin Size

Control spatial resolution:

```python
# Fine bins (2 cm)
fine_maps = processor.process(bin_size=2.0)

# Coarse bins (10 cm)
coarse_maps = processor.process(bin_size=10.0)
```

### Environment Separation

Separate maps by environment:

```python
# Separate by environment
by_env = processor.process(by_environment=True)

# Combined across environments
combined = processor.process(by_environment=False)
```

### Occupancy Filtering

Filter out low-occupancy bins:

```python
# Only include bins with > 0.1 seconds occupancy
filtered = processor.process(min_occupancy=0.1)
```

## Caching

Processors cache results to speed up repeated operations:

```python
# First call processes data
maps1 = processor.process(bin_size=5.0)

# Second call uses cache
maps2 = processor.process(bin_size=5.0)  # Fast!
```

## Custom Processors

You can create custom processors by extending the base processor class:

```python
from vrAnalysis.processors.spkmaps import SpkmapProcessor

class CustomProcessor(SpkmapProcessor):
    def process_custom(self, **kwargs):
        # Custom processing logic
        maps = self.process(**kwargs)
        # Additional processing
        return processed_maps
```

## See Also

- [Sessions Module](sessions.md) for session data
- [Analysis Module](analysis.md) for analysis workflows
- [API Reference](../api/processors.md) for complete function signatures

