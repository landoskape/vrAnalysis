# Helpers

The `vrAnalysis.helpers` module provides utility functions for common operations.

## Helper Modules

### Plotting

Utilities for creating plots:

```python
from vrAnalysis.helpers.plotting import plot_spike_map, plot_traces

# Plot spike map
plot_spike_map(spike_map, occupancy_map)

# Plot calcium traces
plot_traces(traces, time_axis)
```

### Signals

Signal processing utilities:

```python
from vrAnalysis.helpers.signals import smooth, normalize

# Smooth signal
smoothed = smooth(signal, window_size=5)

# Normalize signal
normalized = normalize(signal)
```

### Indexing

Indexing utilities:

```python
from vrAnalysis.helpers.indexing import get_plane_indices

# Get indices for specific plane
indices = get_plane_indices(session, plane=0)
```

### VR Support

VR-specific utilities:

```python
from vrAnalysis.helpers.vrsupport import get_environment_transitions

# Get environment transition times
transitions = get_environment_transitions(behavior)
```

## See Also

- Individual helper modules for specific functionality
- [API Reference](../api/) for complete function listings

