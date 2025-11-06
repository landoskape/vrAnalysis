# Registration

Registration is the process of preprocessing and preparing experimental data from VR sessions for analysis. This includes processing behavioral data from Timeline and vrControl, aligning imaging data from suite2p, running deconvolution algorithms, and creating a unified data structure for analysis.

## Overview

The registration process transforms raw experimental data into a standardized format that can be used for analysis. It handles both behavioral and imaging data. It handles:

- **Timeline Data**: Raw hardware signals from Rigbox (rotary encoder, photodiode, lick detector, reward commands)
- **Behavioral Data**: VR environment data from vrControl (position tracking, trial information, rewards, licks)
- **Imaging Data**: Calcium imaging data from suite2p (ROI fluorescence traces, deconvolved spikes)
- **Data Alignment**: Synchronizing behavioral and imaging data in time

The registration process produces "onedata" files - standardized NumPy arrays stored in the session's `onedata/` directory that can be easily loaded for analysis.

## Prerequisites

Before registering a session, several prerequisites must be met:

### 1. Database Setup

You must have a SQL database (preferably Microsoft Access) set up with a table containing session information. The table should have the following columns (not all are required):

**Required Columns:**

- `uSessionID` (int): Unique session identifier
- `mouseName` (str): Mouse identifier
- `sessionDate` (datetime): Session date
- `sessionID` (int): Session ID number
- `vrRegistration` (bool): Registration status flag
- `imaging` (bool): Whether session has imaging data
- `behavior` (bool): Whether session has behavioral data
- `faceCamera` (bool): Whether session has face camera data
- `vrBehaviorVersion` (int): Version of vrControl software used (1 or 2)

**Optional Columns:**

- `sessionQC` (bool): Quality control flag
- `experimentType` (str)
- `experimentID` (int)
- `variableGain` (bool)
- `vrEnvironments` (int)
- `headPlateRotation` (float)
- `numPlanes` (int)
- `planeSeparation` (float)
- `pockelsPercentage` (float)
- `objectiveRotation` (float)
- `suite2p` (bool)
- `suite2pQC` (bool)
- `redCellQC` (bool)
- `scratchJustification` (str)
- `logtime` (datetime)
- `sessionNotes` (str)
- `suite2pDate` (datetime)
- `vrRegistrationDate` (datetime)
- `vrRegistrationError` (bool)
- `vrRegistrationException` (str)
- `redCellQCDate` (datetime)
- `dontTrack` (bool)

### 2. Adding Sessions to Database

The best way to add a new session entry to the database is using the GUI. The GUI will automatically build a form based on the database schema, so this will actually work for any database object you provide. 

```python
  from vrAnalysis.database import get_database
  from vrAnalysis.uilib.add_entry_gui import NewEntryGUI
  
  # Get database
  db = get_database("vrSessions")
  
  # Optionally create a session object to auto-populate fields
  from vrAnalysis.sessions import create_b2session
  session = create_b2session("mouse001", "2024-01-15", "001")
  
  # any other parameters you want to use to preload the GUI form.
  other_params = {} 

  # Open GUI to add entry
  gui = NewEntryGUI(db, ses=session, **other_params)
```

The GUI will automatically populate fields from the session object if provided, and validate all inputs before submission.

### 3. Suite2p Processing (for imaging sessions)

If the session has imaging data, suite2p must be run before registration. The suite2p output directory should be located at:

```
{session_path}/suite2p/
```

Where `session_path` follows the conventional structure: `{local_data_path}/{mouse_name}/{date}/{session_id}/`

The suite2p directory should contain subdirectories for each imaging plane (e.g., `plane0/`, `plane1/`, etc.), each containing the standard suite2p output files (`.npy` files like `F.npy`, `Fneu.npy`, `spks.npy`, `stat.npy`, `ops.npy`, `iscell.npy`).

### 4. Required Input Files

The session directory must contain the following files. Note that they are determined by other software - including both [vrControl](https://github.com/landoskape/vrControl) and [Rigbox](https://github.com/cortex-lab/Rigbox).

**Timeline Data:**
- `{date}_{session_id}_{mouse_name}_Timeline.mat`: Contains raw hardware signals from Rigbox

**Behavioral Data:**
- `{date}_{session_id}_{mouse_name}_VRBehavior_trial.mat`: Contains VR environment data from vrControl

These files are typically generated during the experiment by Rigbox and vrControl software.

## Data Structure

### Timeline Data

The Timeline.mat file contains hardware signals recorded by Rigbox:

- **timestamps**: Raw DAQ timestamps for all signals
- **rotaryEncoder**: Rotary encoder position counter (circular, needs conversion to linear position)
- **photoDiode**: Photodiode signal for detecting visual stimulus flips
- **lickDetector**: Edge counter signal for detecting licks
- **rewardCommand**: Voltage signal indicating reward delivery
- **neuralFrames**: Frame counter from ScanImage indicating imaging volume acquisition
- **mpepUDPTimes/mpepUDPEvents**: Trial start/end messages from vrControl

### Behavioral Data

The VRBehavior_trial.mat file contains VR environment data. The structure varies by `vrBehaviorVersion`:

**Version 1 (standard_behavior):**

- `expInfo`: Experiment information (room length, movement gain per trial)
- `trialInfo`: Sparse matrices containing trial-by-trial data:
  - `time`: Timestamps for each behavioral sample
  - `roomPosition`: Virtual position in VR environment
  - `rewardPosition`: Reward zone position
  - `rewardTolerance`: Reward zone half-width
  - `rewardAvailable`: Whether reward was available
  - `rewardDeliveryFrame`: Frame when reward was delivered
  - `activeLicking`: Whether active licking was required
  - `activeStopping`: Whether active stopping was required
  - `lick`: Lick counts per frame
  - `vrEnvIdx`: Environment index (optional)

**Version 2 (cr_hippocannula_behavior):**

- Similar structure but with different field names (`TRIAL` instead of `trialInfo`, `EXP` instead of `expInfo`)

Both versions also contain `rigInfo` with hardware configuration:

- `rotEncPos`: Rotary encoder position ("left" or "right")
- `rotEncSign`: Sign of rotary encoder (-1 or 1)
- `wheelToVR`: Conversion factor from wheel counts to VR units
- `wheelRadius`: Physical wheel radius in cm
- `rotaryRange`: Bit range of rotary encoder (typically 32)

### Imaging Data

Suite2p outputs are organized by plane:

```
suite2p/
  plane0/
    F.npy          # Fluorescence traces (nROIs x nFrames)
    Fneu.npy       # Neuropil fluorescence (nROIs x nFrames)
    spks.npy       # Deconvolved spikes (nROIs x nFrames)
    stat.npy        # ROI statistics (list of dicts)
    ops.npy         # Suite2p options (dict)
    iscell.npy      # Cell classifier output (nROIs x 2)
    redcell.npy     # Red cell classifier output (optional, nROIs x 2)
```

## Using the Database Pipeline

The recommended way to register sessions is through the database pipeline, which handles error tracking and status updates automatically. The database object for sessions includes some supporting methods that make it easy to identify sessions that need registration and register them all at once.

### Finding Sessions Needing Registration

```python
  from vrAnalysis.database import get_database
  
  db = get_database("vrSessions")
  
  # Get DataFrame of sessions needing registration
  needs_reg = db.needs_registration(return_df=True)
  
  # Filter by mouse
  needs_reg = db.needs_registration(mouseName="mouse001", return_df=True)
  
  # Print sessions (alternative to DataFrame)
  db.needs_registration(return_df=False, mouseName="mouse001")
```

### Registering a Single Session

```python
  from vrAnalysis.database import get_database
  
  db = get_database("vrSessions")
  
  # Register by identifiers
  success = db.register_single_session(
      mouse_name="mouse001",
      session_date="2024-01-15",
      session_id="001",
  )
  
  if success:
      print("Registration successful!")
  else:
      print("Registration failed - check database for error details")
```

### Registering Multiple Sessions

```python
  from vrAnalysis.database import get_database
  
  db = get_database("vrSessions")
  
  # Register all sessions needing registration
  # Stops when total data size exceeds max_data (default 30 GB)
  db.register_sessions(
      max_data=30e9,        # Maximum total data to process (bytes)
      skip_errors=True,     # Skip sessions with previous errors
      raise_exception=False, # Don't raise exceptions on failure
  )
```

The `register_sessions` method provides progress updates including:
- Accumulated onedata registered
- Average data size per session
- Estimated remaining data to process

### Error Handling

When registration fails, the database is automatically updated:

- `vrRegistrationError` set to `True`
- `vrRegistrationException` contains the error message
- All onedata files are cleared
- Registration status remains `False`

You can check for errors:

```python
  # Print all registration errors
  db.print_registration_errors()
  
  # Get sessions with errors
  errors = db.get_table(vrRegistrationError=True)
```

If you try to register sessions without setting the ``skip_errors`` parameter to ``False``, the bulk registration pipeline will simply skip sessions that had previous errors. You probably don't want this
unless you want to abandon the session! So use the ``needs_registration`` method above and the ``kwarg``
``vrRegistrationError=True`` to identify sessions that had errors. You'll need to debug them yourself.

## Manual Registration

You can also register sessions directly without using the database:

```python
  from vrAnalysis.registration import B2Registration
  from vrAnalysis.sessions.b2session import B2RegistrationOpts
  
  # Create registration options
  opts = B2RegistrationOpts(
      vrBehaviorVersion=1,      # Version of vrControl (1 or 2)
      facecam=False,             # Process face camera data
      imaging=True,              # Process imaging data
      oasis=True,                # Run OASIS deconvolution
      redCellProcessing=True,   # Process red cell features
      clearOne=True,            # Clear existing onedata before registration
      neuropilCoefficient=0.7,  # Neuropil subtraction coefficient
      tau=1.5,                  # OASIS tau parameter (seconds)
      fs=6                      # Sampling frequency (Hz)
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
  
  # Save session parameters (automatically called by register())
  # registration.save_session_prms()
```

## Registration Options

The `B2RegistrationOpts` dataclass controls registration behavior:

**Behavior Options:**
- `vrBehaviorVersion` (int): Version of vrControl software (1=standard, 2=CR hippocampus, software no longer exists!)
    - Note: this system will allow you to use this pipeline with new behavior software, just add a new version number and a new behavior processing function in the ``vrAnalysis.registration.behavior`` module.

**Data Processing Flags:**

- `imaging` (bool): Process imaging data (requires suite2p directory)
- `facecam` (bool): Process face camera data (not yet implemented)
- `redCellProcessing` (bool): Compute red cell features

**Deconvolution Options:**

- `oasis` (bool): Run OASIS deconvolution (recomputes spikes from corrected fluorescence)
- `tau` (float): OASIS decay time constant in seconds (default: 1.5)
- `fs` (int): Sampling frequency in Hz (default: 6)
- `neuropilCoefficient` (float): Coefficient for neuropil subtraction (default: 0.7)

**Data Management:**

- `clearOne` (bool): Clear existing onedata before registration (default: True)

## Output Data Structure

After registration, the session's `onedata/` directory contains standardized NumPy arrays:

### Timeline Data

- `wheelPosition.times.npy`: Timeline timestamps
- `wheelPosition.position.npy`: Wheel position in cm
- `licks.times.npy`: Lick event timestamps
- `rewards.times.npy`: Reward delivery timestamps
- `trials.startTimes.npy`: Trial start timestamps

### Behavioral Data

- `positionTracking.times.npy`: Behavioral timestamps
- `positionTracking.position.npy`: Virtual position in VR environment
- `positionTracking.mpci.npy`: Mapping from behavioral samples to imaging frames
- `trials.positionTracking.npy`: Start frame index for each trial
- `trials.environmentIndex.npy`: Environment ID for each trial
- `trials.roomlength.npy`: Room length for each trial
- `trials.movementGain.npy`: Movement gain for each trial
- `trials.rewardPosition.npy`: Reward zone position
- `trials.rewardZoneHalfwidth.npy`: Reward zone half-width
- `trials.rewardAvailability.npy`: Whether reward was available
- `trials.rewardPositionTracking.npy`: Reward delivery frame index
- `trials.activeLicking.npy`: Active licking requirement
- `trials.activeStopping.npy`: Active stopping requirement
- `licksTracking.positionTracking.npy`: Lick event indices in behavioral samples

### Imaging Data

- `mpci.times.npy`: Imaging frame timestamps
- `mpci.roiActivityF.npy`: Fluorescence traces (or LoadingRecipe reference)
- `mpci.roiNeuropilActivityF.npy`: Neuropil fluorescence (or LoadingRecipe reference)
- `mpci.roiActivityDeconvolved.npy`: Suite2p deconvolved spikes (or LoadingRecipe reference)
- `mpci.roiActivityDeconvolvedOasis.npy`: OASIS deconvolved spikes (if computed)
- `mpciROIs.isCell.npy`: Cell classifier output
- `mpciROIs.stackPosition.npy`: ROI positions (nROIs x 3: x, y, plane)
- `mpciROIs.redS2P.npy`: Red cell classifier output (if available)

### Red Cell Data (if processed)

- `mpciROIs.redDotProduct.npy`: Dot product feature
- `mpciROIs.redPearson.npy`: Pearson correlation feature
- `mpciROIs.redPhaseCorrelation.npy`: Phase correlation feature
- `mpciROIs.redCellIdx.npy`: Boolean array for red cell identification
- `mpciROIs.redCellManualAssignments.npy`: Manual assignment array
- `parametersRedDotProduct.keyValuePairs.npy`: Dot product parameters
- `parametersRedPearson.keyValuePairs.npy`: Pearson parameters
- `parametersRedPhaseCorrelation.keyValuePairs.npy`: Phase correlation parameters

### Session Metadata

- `vrExperimentOptions.json`: Registration options used
- `vrExperimentPreprocessing.json`: List of preprocessing steps completed
- `vrExperimentValues.json`: Session metadata values (numTrials, numROIs, etc.)

## Troubleshooting

### Common Issues

**"Session directory does not exist"**

- Ensure the session directory follows the expected structure: `{local_data_path}/{mouse_name}/{date}/{session_id}/`
- Check that `local_data_path()` is configured correctly

**"suite2p directory does not exist"**

- Run suite2p processing before registration
- Ensure suite2p output is in `{session_path}/suite2p/`

**"Missing required suite2p files"**

- Verify all planes have the required `.npy` files (F, Fneu, spks, stat, ops, iscell)
- Check for typos in file names

**"Frame count mismatch"**

- The registration process attempts to handle minor mismatches automatically
- Large mismatches may indicate a problem with the Timeline neuralFrames signal
- Check that ScanImage TTL signals were properly recorded

**"Behavior type not supported"**

- Ensure `vrBehaviorVersion` matches the version of vrControl used
- Currently supported: 1 (standard), 2 (CR hippocampus)

**"First flips in trial are not all down"**

- This assertion only applies to sessions after 2022-08-30
- May indicate a problem with photodiode signal or trial preparation

### Checking Registration Status

```python
  from vrAnalysis.database import get_database
  
  db = get_database("vrSessions")
  
  # Check if session is registered
  record = db.get_record("mouse001", "2024-01-15", "001")
  if record is not None:
      print(f"Registration status: {record['vrRegistration']}")
      print(f"Registration date: {record['vrRegistrationDate']}")
      if record['vrRegistrationError']:
          print(f"Error: {record['vrRegistrationException']}")
```

## See Also

- [Database Module](modules/database.md) for database management
- [Sessions Module](modules/sessions.md) for session data loading
- [Registration Module](modules/registration.md) for API reference
- [Registration Examples](examples/registration.md) for detailed workflows

