# Quickstart Guide

This guide will walk you through the basic workflows in vrAnalysis2.

## Creating a Session

The core object in vrAnalysis2 is the `B2Session`, which represents a single experimental session.

### Basic Session Creation

```python
from vrAnalysis2.sessions import create_b2session

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
from vrAnalysis2.sessions import create_b2session

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
from vrAnalysis2.database import get_database

# Get database metadata
db = get_database("vrSessions")

# Query sessions
sessions = db.get_sessions(
    conditions={"mouseName": "mouse001"},
    sessionQC=True  # Only get QC'd sessions
)

# Access session data
for session_row in sessions:
    print(f"Session: {session_row['sessionDate']} - {session_row['sessionID']}")
```

### Adding Sessions to Database

Use the GUI to add new sessions:

```python
from vrAnalysis2.uilib.add_entry_gui import add_entry_gui

# Open the database entry GUI
add_entry_gui("vrSessions")
```

## Registration Workflow

Registration is the process of preprocessing and aligning behavioral and imaging data.

```python
from vrAnalysis2.registration import B2Registration
from vrAnalysis2.sessions.b2session import B2RegistrationOpts

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
from vrAnalysis2.processors.spkmaps import SpkmapProcessor

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
from vrAnalysis2.tracking import TrackedPair

# Load two sessions
session1 = create_b2session("mouse001", "2024-01-15", "001")
session2 = create_b2session("mouse001", "2024-01-16", "001")

# Track cells
tracked = TrackedPair(session1, session2)
matched_pairs = tracked.get_matched_pairs()

# Access matched ROIs
for roi1_idx, roi2_idx in matched_pairs:
    print(f"ROI {roi1_idx} in session 1 matches ROI {roi2_idx} in session 2")
```

## Multi-Session Analysis

Analyze data across multiple sessions:

```python
from vrAnalysis2.multisession import MultiSession
from vrAnalysis2.database import get_database_metadata, SessionDatabase

# Get database
db_meta = get_database_metadata("vrSessions")
db = SessionDatabase(**db_meta)

# Get sessions for a mouse
sessions_data = db.get_sessions(conditions={"mouseName": "mouse001"})

# Create multi-session object
multi = MultiSession(sessions_data)

# Perform analysis across sessions
# (specific analysis methods depend on your needs)
```

## Next Steps

- Learn more about [Database Management](modules/database.md)
- Explore [Session Configuration](modules/sessions.md)
- Understand [Registration Workflows](modules/registration.md)
- Check out [Analysis Tools](modules/analysis.md)

