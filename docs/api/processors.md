# Processors API Reference

The processors module provides data processing pipelines that transform session data into analysis-ready formats. The main component is the `SpkmapProcessor`, which creates spatial maps of neural activity, behavioral occupancy, and speed.

## Overview

The processors module contains:

- **SpkmapProcessor**: Main class for processing spike maps from session data
- **Maps**: Container class for occupancy, speed, and spike maps
- **Reliability**: Container class for reliability measurements
- **SpkmapParams**: Configuration parameters for spike map processing

## Core Classes

::: vrAnalysis.processors.spkmaps.SpkmapProcessor
  options:
    show_root_heading: true
    show_root_toc_entry: true
    heading_level: 3

::: vrAnalysis.processors.spkmaps.Maps
  options:
    show_root_heading: true
    show_root_toc_entry: true
    heading_level: 3

::: vrAnalysis.processors.spkmaps.Reliability
  options:
    show_root_heading: true
    show_root_toc_entry: true
    heading_level: 3

::: vrAnalysis.processors.spkmaps.SpkmapParams
  options:
    show_root_heading: true
    show_root_toc_entry: true
    heading_level: 3

## Usage Examples

### Basic Usage

```python
from vrAnalysis.processors import SpkmapProcessor, SpkmapParams
from vrAnalysis.sessions import B2Session

# Create a session
session = B2Session("path/to/session")

# Create a processor with default parameters
processor = SpkmapProcessor(session)

# Get raw maps (unsmoothed, not normalized)
raw_maps = processor.get_raw_maps()

# Get processed maps (smoothed and normalized by occupancy)
processed_maps = processor.get_processed_maps()

# Get environment-separated maps
env_maps = processor.get_env_maps()

# Calculate reliability
reliability = processor.get_reliability()
```

### Custom Parameters

```python
# Create custom parameters
params = SpkmapParams(
    dist_step=2.0,  # 2 cm bins
    speed_threshold=2.0,  # 2 cm/s minimum speed
    smooth_width=5.0,  # 5 cm smoothing
    standardize_spks=True,
)

# Use custom parameters
processor = SpkmapProcessor(session, params=params)

# Or update parameters temporarily
maps = processor.get_processed_maps(params={"smooth_width": 10.0})
```

### Working with Maps

```python
# Get processed maps
maps = processor.get_processed_maps()

# Filter to specific ROIs
maps.filter_rois([0, 1, 2, 3])

# Filter to specific positions
maps.filter_positions(np.arange(10, 50))

# Average across trials
maps.average_trials()

# Check memory usage
print(f"Maps use {maps.nbytes() / 1e6:.2f} MB")
```

### Environment-Separated Maps

```python
# Get maps separated by environment
env_maps = processor.get_env_maps()

# Access maps for a specific environment
env_idx = 0
occmap_env0 = env_maps.occmap[env_idx]
spkmap_env0 = env_maps.spkmap[env_idx]

# Filter to specific environments
env_maps.filter_environments([1, 2])  # Keep only environments 1 and 2
```

### Reliability Analysis

```python
# Calculate reliability with default method (leave_one_out)
reliability = processor.get_reliability()

# Access reliability values
# Shape: (num_environments, num_rois)
reliability_values = reliability.values

# Filter to specific ROIs
reliability_filtered = reliability.filter_rois([0, 1, 2])

# Filter to specific environments
reliability_env = reliability.filter_by_environment([1, 2])
```

### Place Field Predictions

```python
# Get place field predictions for each frame
prediction, extras = processor.get_placefield_prediction()

# prediction shape: (num_frames, num_rois)
# extras contains frame_position_index, frame_environment_index, idx_valid

# Use predictions to analyze neural activity
valid_predictions = prediction[extras["idx_valid"]]
```

### Traversals Analysis

```python
# Extract activity around place field peak
roi_idx = 5  # Neuron index
env_idx = 0  # Environment index

traversals, pred_travs = processor.get_traversals(
    idx_roi=roi_idx,
    idx_env=env_idx,
    width=10,  # 10 frames on each side
    placefield_threshold=5.0,  # 5 cm threshold
)

# traversals shape: (num_traversals, 21)  # 2*width + 1
# pred_travs shape: (num_traversals, 21)
```

## Caching

The SpkmapProcessor uses intelligent caching to avoid recomputing maps when parameters haven't changed:

```python
# Enable autosave to cache results
params = SpkmapParams(autosave=True)
processor = SpkmapProcessor(session, params=params)

# First call computes and caches
maps1 = processor.get_processed_maps()

# Second call loads from cache (much faster)
maps2 = processor.get_processed_maps()

# Force recomputation
maps3 = processor.get_processed_maps(force_recompute=True)

# View cache information
processor.show_cache()
processor.show_cache(data_type="processed_maps")
```

## Protocol Interface

The `SpkmapProcessor` works with any session class that implements the `SessionToSpkmapProtocol`. This protocol defines the required properties:

- `spks`: Spike data array
- `spks_type`: Type of spike data
- `idx_rois`: ROI filter mask
- `timestamps`: Imaging frame timestamps
- `env_length`: Environment length(s)
- `positions`: Position data tuple
- `trial_environment`: Environment for each trial
- `num_trials`: Number of trials
- `zero_baseline_spks`: Whether spikes are zero-baselined

See the protocol documentation for details on implementing custom session classes.
