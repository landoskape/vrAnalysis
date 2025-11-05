# Registration

The `vrAnalysis2.registration` module handles preprocessing and registration of experimental data. Registration aligns behavioral and imaging data, runs deconvolution, and prepares data for analysis.

## Core Classes

### B2Registration

Extends `B2Session` with registration workflows. Handles the complete preprocessing pipeline.

**Key Methods:**

- `register()`: Run the complete registration workflow
- `register_behavior()`: Process and register behavioral data
- `register_imaging()`: Process and register imaging data
- `register_oasis()`: Run OASIS deconvolution
- `register_redcell()`: Process red cell annotations

**Example:**

```python
from vrAnalysis2.registration import B2Registration
from vrAnalysis2.sessions.b2session import B2RegistrationOpts

# Create registration options
opts = B2RegistrationOpts(
    vrBehaviorVersion=1,
    imaging=True,
    oasis=True,
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

### B2RegistrationOpts

Dataclass for configuring registration options.

**Parameters:**

- `vrBehaviorVersion`: Version of vrControl software used (1 or 2)
- `facecam`: Whether to process facecam data
- `imaging`: Whether to process imaging data
- `oasis`: Whether to run OASIS deconvolution
- `moveRawData`: Whether to move raw data files
- `redCellProcessing`: Whether to process red cell annotations
- `clearOne`: Whether to clear One files
- `neuropilCoefficient`: Coefficient for neuropil subtraction (default: 0.7)
- `tau`: OASIS tau parameter (default: 1.5)
- `fs`: Sampling frequency in Hz (default: 6)

## Registration Workflow

The registration process includes:

1. **Behavior Registration**: Load and process behavioral data from Timeline files
2. **Imaging Registration**: Load suite2p outputs and process imaging data
3. **OASIS Deconvolution**: Deconvolve calcium traces to get spike trains
4. **Red Cell Processing**: Process red cell classifier results
5. **Data Alignment**: Align behavioral and imaging data in time

## Behavior Processing

Behavior processing handles different versions of vrControl:

```python
from vrAnalysis2.registration.behavior import register_behavior

# Process behavior for version 1
registration = register_behavior(registration, behavior_type=1)

# Process behavior for version 2 (CR hippocampus)
registration = register_behavior(registration, behavior_type=2)
```

Different behavior versions may require different processing:
- Position tracking
- Reward delivery
- Lick detection
- Environment transitions

## OASIS Deconvolution

OASIS deconvolution converts calcium traces to spike trains:

```python
from vrAnalysis2.registration.oasis import oasis_deconvolution

# Run OASIS
deconvolved = oasis_deconvolution(
    traces,
    tau=1.5,
    fs=6
)
```

## Red Cell Processing

Process red cell classifier results:

```python
from vrAnalysis2.registration.redcell import RedCellProcessing

# Create processor
processor = RedCellProcessing(session)

# Process red cells
processor.process()
```

## Database Integration

Registration status is tracked in the database:

```python
from vrAnalysis2.database import get_database

# Get database
db = get_database("vrSessions")

# Find sessions needing registration
needs_reg = db.needs_registration(mouseName="mouse001")

# Run registration for each
for _, row in needs_reg.iterrows():
    registration = db.make_b2registration(row, opts)
    registration.register()
    
    # Update database
    db.update_database_field("vrRegistration", True, uSessionID=row["uSessionID"])
```

## See Also

- [Quickstart Guide](../quickstart.md) for basic usage
- [Sessions Module](sessions.md) for session management
- [Examples](../examples/registration.md) for detailed workflows
- [API Reference](../api/registration.md) for complete function signatures

