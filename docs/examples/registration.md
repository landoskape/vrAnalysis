# Registration Workflow Example

This example demonstrates a complete registration workflow for processing VR session data.

## Basic Registration

```python
from vrAnalysis.registration import B2Registration
from vrAnalysis.sessions.b2session import B2RegistrationOpts

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

# Create and run registration
registration = B2Registration(
    mouse_name="mouse001",
    date_string="2024-01-15",
    session_id="001",
    opts=opts
)

registration.register()
```

## Batch Registration

Register multiple sessions from the database:

```python
from vrAnalysis.database import get_database
from vrAnalysis.sessions.b2session import B2RegistrationOpts

# Get database
db = get_database("vrSessions")

# Find sessions needing registration
needs_reg = db.needs_registration(mouseName="mouse001")

# Create options
opts = B2RegistrationOpts(
    vrBehaviorVersion=1,
    imaging=True,
    oasis=True,
    redCellProcessing=True
)

# Register each session
for _, row in needs_reg.iterrows():
    try:
        registration = db.make_b2registration(row, opts)
        registration.register()
        
        # Update database
        db.update_database_field("vrRegistration", True, uSessionID=row["uSessionID"])
        print(f"Successfully registered: {registration.session_print()}")
    except Exception as e:
        print(f"Error registering {row['uSessionID']}: {e}")
        db.update_database_field("vrRegistrationError", True, uSessionID=row["uSessionID"])
```

## Custom Registration Options

Use different options for different sessions:

```python
# Standard registration
standard_opts = B2RegistrationOpts(
    vrBehaviorVersion=1,
    oasis=True,
    tau=1.5
)

# High-resolution registration
highres_opts = B2RegistrationOpts(
    vrBehaviorVersion=1,
    oasis=True,
    tau=0.8,  # Higher temporal resolution
    fs=10     # Higher sampling rate
)

# Apply based on session criteria
if session_date >= "2024-01-01":
    opts = highres_opts
else:
    opts = standard_opts
```

