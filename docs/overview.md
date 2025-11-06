# Overview

## Architecture

vrAnalysis is organized around a few core concepts:

### Sessions

A **session** represents a single experimental run. It's the central object that enables accesss to data for an experiment. In general, all more complicated analyses are built on top of session objects - in other words, the interface enabled by session is the bottleneck to all analysis of the package. 

Each session contains:

- Behavioral data (position, velocity, rewards, etc.)
- Imaging data (calcium traces, ROIs, etc.)
- Metadata (mouse name, date, session ID, etc.)

Sessions are represented by the `B2Session` class, which provides methods to load and access data. (`B2Session` is one example of a possible `SessionData` class, but since all of my work was done at B2, 
this is really the only one that's used. The other ones are for converting external data into a format that can be used by this package, which are explained elsewhere).

### Database

The **database** tracks all sessions and their metadata. It's a simple interface that allows you to query your database easily and efficiently, and also to perform batch operations on sessions in the database. It uses Microsoft Access (`.accdb`) files by default, but can be adapted to other SQL databases.

The `SessionDatabase` class provides methods to:

- Query sessions based on criteria
- Add new sessions
- Update session metadata
- Track registration status and quality control flags

### Processors

**Processors** transform session data into analysis-ready formats. For example:

- `SpkmapProcessor`: Creates spatial maps of neural activity
- I haven't built any others yet, maybe one day!

### Tracking

**Tracking** identifies the same cells across multiple sessions. This enables longitudinal analysis of how individual cells change over time. This module is built with the ``Tracker`` class which is a nice wrapper around ROICaT tracking files. It will only work (or be relevant) if you've used ROICaT to track cells across sessions. 

### Registration

**Registration** is the process of preprocessing raw experimental data. This includes:

- Loading behavioral data from Timeline files
- Processing imaging data from suite2p outputs
- Running OASIS deconvolution
- Processing red cell annotations
- Aligning data in time

Registration creates standardized data structures that can be used for analysis.


## Data Flow

```
Raw Data (Timeline, vrControl, suite2p)
    ↓
Registration (B2Registration)
    ↓
Session Data (B2Session)
    ↓
Processing (Processors)
    ↓
Figures!
```

## Directory Structure

Not sure where else to put this, but vrAnalysis expects data organized in an Alyx-style structure. This is the structure that is used by the [alyx](https://github.com/cortex-lab/alyx) database and is generally a really nice way to organize sessions of data colleted on certain days from specific mice. 

```bash
localData/ # configure this in the `local_data_path()` function of `vrAnalysis/files.py`
├── mouse001/ # mouse name
│   ├── 2024-01-15/ # date of session 
│   │   ├── 001/ # session ID
│   │   │   ├── suite2p/ # where your suite2p output should go 
│   │   │   ├── onedata/ # where onedata will be stored after registration
│   │   │   └── ... # other files
│   │   └── 002/ # session ID
│   └── 2024-01-16/ # date of session
└── mouse002/ # mouse name
```

