# Sessions

The `vrAnalysis.sessions` module provides classes for loading and managing VR session data. Sessions are the core objects that represent individual experimental runs.

## Core Classes

### B2Session

The main session class that loads and provides access to experimental data.

**Key Attributes:**

- `mouse_name`: Mouse identifier
- `date`: Session date
- `session_id`: Session identifier
- `params`: Session parameters (B2SessionParams)

**Key Methods:**

- `load_data()`: Load behavioral and imaging data
- `get_spks()`: Get spike data
- `get_behavior()`: Get behavioral data
- `session_print()`: Print session information

**Example:**

```python
from vrAnalysis.sessions import create_b2session

# Create session with default parameters
session = create_b2session(
    mouse_name="mouse001",
    date="2024-01-15",
    session_id="001"
)

# Load data
session.load_data()

# Access spike data
spks = session.get_spks()

# Access behavioral data
behavior = session.get_behavior()
```

### B2SessionParams

Dataclass for configuring how session data is loaded.

**Parameters:**

- `spks_type`: Type of spike data to load (e.g., "significant")
- `keep_planes`: List of plane indices to keep (None = all)
- `good_labels`: List of ROI labels to keep (e.g., ["c", "d"])
- `fraction_filled_threshold`: Threshold for ROI fraction filled
- `footprint_size_threshold`: Threshold for ROI footprint size
- `exclude_silent_rois`: Whether to exclude ROIs with no activity
- `neuropil_coefficient`: Neuropil subtraction coefficient
- `exclude_redundant_rois`: Whether to exclude redundant ROIs

**Example:**

```python
from vrAnalysis.sessions import create_b2session, B2SessionParams

# Create custom parameters
params = B2SessionParams(
    spks_type="significant",
    keep_planes=[0, 1, 2],
    good_labels=["c", "d"],
    exclude_silent_rois=True
)

# Create session with custom parameters
session = create_b2session(
    mouse_name="mouse001",
    date="2024-01-15",
    session_id="001",
    params=params
)
```

## Session Data Structure

Sessions provide access to:

### Behavioral Data

- Position tracking
- Velocity
- Rewards
- Licks
- Environment information

### Imaging Data

- Calcium traces (F, Fneu, etc.)
- Spike data (deconvolved or raw)
- ROI information
- Neuropil signals
- Suite2p statistics

### Metadata

- Session information
- Processing status
- Quality control flags

## Loading Data

Data is loaded lazily by default. Access data properties to trigger loading:

```python
# Data is loaded when accessed
spks = session.spks  # Loads spike data
behavior = session.behavior  # Loads behavioral data
```

## Session Parameters

Session parameters control data filtering and processing:

```python
# Update parameters
session.params.update(
    good_labels=["c"],
    exclude_silent_rois=True
)

# Reload data with new parameters
session.load_data()
```

## File Paths

Sessions automatically construct paths to data files:

```python
# Suite2p path
s2p_path = session.s2p_path

# Timeline path
timeline_path = session.timeline_path

# Session root
session_root = session.session_root
```

## See Also

- [Quickstart Guide](../quickstart.md) for basic usage
- [Registration Module](registration.md) for preprocessing workflows
- [API Reference](../api/sessions.md) for complete function signatures

