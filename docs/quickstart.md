# Quickstart Guide

This guide will walk you through the basic workflows in vrAnalysis2.

## Creating a Session

The core object in vrAnalysis2 is the `B2Session`, which represents a single experimental session.

### Basic Session Creation

```python
from vrAnalysis.sessions import create_b2session

# Create a session with default parameters
session = create_b2session(
    mouse_name="mouse001",
    date="2024-01-15",
    session_id="001"
)
```

### Custom Session Parameters

You can customize how data is loaded using `B2SessionParams`:

```python
from vrAnalysis.sessions import create_b2session

# Create a session with custom parameters
params = {
    "spks_type": "significant",  # Use significant transients
    "keep_planes": [0, 1, 2],    # Only load specific planes
    "good_labels": ["c", "d"],   # Keep only "c" and "d" classified ROIs
    "exclude_silent_rois": True   # Exclude ROIs with no activity
}

session = create_b2session(
    mouse_name="mouse001",
    date="2024-01-15",
    session_id="001",
    params=params
)
```

## Working with the Database

### Querying Sessions

```python
from vrAnalysis.database import get_database

# Get database metadata
db = get_database("vrSessions")

# Query sessions
sessions_df = db.get_table(mouseName="mouse001", sessionQC=True)

# Access session data
for _, session_row in sessions_df.iterrows():
    print(f"Session: {session_row['sessionDate']} - {session_row['sessionID']}")

# Or create session objects directly
sessions = db.iter_sessions(mouseName="mouse001", sessionQC=True)
for session in sessions:
    print(f"Session: {session.session_print()}")
```

### Adding Sessions to Database

Use the GUI to add new sessions:

```python
from vrAnalysis.uilib.add_entry_gui import add_entry_gui

# Open the database entry GUI
add_entry_gui("vrSessions")
```

## Registration Workflow

Registration is the process of preprocessing and aligning behavioral and imaging data.

```python
from vrAnalysis.registration import B2Registration
from vrAnalysis.sessions.b2session import B2RegistrationOpts

# Create registration options
opts = B2RegistrationOpts(
    vrBehaviorVersion=1,
    imaging=True,
    oasis=True,  # Run OASIS deconvolution
    redCellProcessing=True,
    neuropilCoefficient=0.7,
    tau=1.5,
    fs=6
)

# Create registration object
registration = B2Registration(
    mouse_name="mouse001",
    date_string="2024-01-15",
    session_id="001",
    opts=opts
)

# Run registration
registration.register()
```

## Processing Spike Maps

Generate spatial representations of neural activity:

```python
from vrAnalysis.processors.spkmaps import SpkmapProcessor

# Create processor
processor = SpkmapProcessor(session)

# Generate maps
maps = processor.process(
    bin_size=5.0,  # 5 cm bins
    by_environment=True,  # Separate by environment
    rois_first=True
)

# Access maps
occupancy_map = maps.occmap
spike_map = maps.spkmap
speed_map = maps.speedmap
```

## Tracking Cells Across Sessions

Track the same cells across multiple sessions:

```python
from vrAnalysis.tracking import Tracker

# Create tracker for a mouse (tracks all sessions for that mouse)
tracker = Tracker("mouse001")

# Get tracked ROIs across sessions
idx_tracked, extras = tracker.get_tracked_idx(
    idx_ses=[0, 1],  # Track between first two sessions
    use_session_filters=True
)

# idx_tracked is a (num_sessions, num_tracked_rois) array
# Each column represents a tracked ROI across sessions
for roi_idx in range(idx_tracked.shape[1]):
    session1_roi = idx_tracked[0, roi_idx]
    session2_roi = idx_tracked[1, roi_idx]
    print(f"ROI {session1_roi} in session 1 matches ROI {session2_roi} in session 2")
```

## Multi-Session Analysis

Analyze data across multiple sessions:

```python
from vrAnalysis.multisession import MultiSessionSpkmaps
from vrAnalysis.tracking import Tracker

# Create tracker for a mouse
tracker = Tracker("mouse001")

# Create multi-session object for spike map analysis
multi = MultiSessionSpkmaps(tracker)

# Perform analysis across sessions
# (specific analysis methods depend on your needs)
```

## Next Steps

- Learn more about [Database Management](modules/database.md)
- Explore [Session Configuration](modules/sessions.md)
- Understand [Registration Workflows](modules/registration.md)
- Check out [Analysis Tools](modules/analysis.md)

