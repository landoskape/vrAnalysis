# Overview

## Architecture

vrAnalysis2 is organized around a few core concepts:

### Sessions

A **session** represents a single experimental run. Each session contains:
- Behavioral data (position, velocity, rewards, etc.)
- Imaging data (calcium traces, ROIs, etc.)
- Metadata (mouse name, date, session ID, etc.)

Sessions are represented by the `B2Session` class, which provides methods to load and access data.

### Database

The **database** tracks all sessions and their metadata. It uses Microsoft Access (`.accdb`) files by default, but can be adapted to other SQL databases.

The `SessionDatabase` class provides methods to:
- Query sessions based on criteria
- Add new sessions
- Update session metadata
- Track registration status and quality control flags

### Registration

**Registration** is the process of preprocessing raw experimental data. This includes:
- Loading behavioral data from Timeline files
- Processing imaging data from suite2p outputs
- Running OASIS deconvolution
- Processing red cell annotations
- Aligning data in time

Registration creates standardized data structures that can be used for analysis.

### Processors

**Processors** transform session data into analysis-ready formats. For example:
- `SpkmapProcessor`: Creates spatial maps of neural activity
- Other processors generate various representations of the data

### Tracking

**Tracking** identifies the same cells across multiple sessions. This enables longitudinal analysis of how individual cells change over time.

## Data Flow

```
Raw Data (Timeline, suite2p)
    ↓
Registration (B2Registration)
    ↓
Session Data (B2Session)
    ↓
Processing (Processors)
    ↓
Analysis (Analysis Tools)
```

## Directory Structure

vrAnalysis2 expects data organized in an Alyx-style structure:

```
localData/
├── mouse001/
│   ├── 2024-01-15/
│   │   ├── 001/
│   │   │   ├── suite2p/
│   │   │   ├── timeline/
│   │   │   └── ...
│   │   └── 002/
│   └── 2024-01-16/
└── mouse002/
```

## Key Design Principles

1. **Session-Centric**: All operations revolve around session objects
2. **Database-Driven**: Session metadata is tracked in a database
3. **Lazy Loading**: Data is loaded on-demand to minimize memory usage
4. **Caching**: Intermediate results are cached to speed up repeated operations
5. **Modularity**: Each component can be used independently

## Common Workflows

### 1. Daily Registration Workflow

1. Add new sessions to database using GUI
2. Run registration for new sessions
3. Review and QC registered sessions
4. Update database with QC status

### 2. Analysis Workflow

1. Query database for sessions of interest
2. Load sessions into `B2Session` objects
3. Process data using processors
4. Perform analysis
5. Visualize and save results

### 3. Longitudinal Analysis Workflow

1. Query database for sessions from same mouse
2. Load sessions and track cells across sessions
3. Analyze changes in tracked cells
4. Compare across experimental conditions

## Integration with Other Tools

vrAnalysis2 integrates with:
- **suite2p**: For calcium imaging data processing
- **vrControl**: For behavioral data collection
- **Timeline (Rigbox)**: For experimental event tracking
- **ROICaT**: For ROI classification (via `roicat_support`)
- **OASIS**: For calcium trace deconvolution

## Next Steps

- Read the [Quickstart Guide](quickstart.md) for hands-on examples
- Explore [Module Documentation](modules/database.md) for detailed information
- Check the [API Reference](api/vrAnalysis2.md) for complete function signatures

