# Same Cell Candidates Module

This document provides a detailed overview of the `sameCellCandidates` module, which is designed to identify and analyze potential same-cell ROIs (Regions Of Interest) across different imaging planes in volumetric calcium imaging data.

## sameCellCandidates

The `sameCellCandidates` class is the core component that performs the analysis of potential same-cell candidates across imaging planes.

### Conceptual Overview

At a high level, the `sameCellCandidates` class:

1. **Identifies potential same-cell ROIs** across different imaging planes based on:
   - Spatial proximity (distance between ROIs in μm)
   - Activity correlation (temporal correlation of calcium signals)
   - Plane relationships (same plane, neighboring planes, distant planes)

2. **Provides analysis tools** to quantify and visualize these relationships:
   - Correlation vs. distance relationships
   - Cluster size distributions
   - Plane-pair histograms
   - Distance distributions

3. **Supports filtering** of ROI pairs based on multiple criteria:
   - Correlation thresholds
   - Distance cutoffs
   - Plane selection
   - ROI size (number of pixels)

### Implementation Details

#### Data Organization

The class stores several key data structures:
- `xcROIs`: Cross-correlation matrix between all ROI pairs
- `pwDist`: Pairwise distances between ROIs in μm
- `planePair1`, `planePair2`: Plane indices for each ROI pair
- `idxRoi1`, `idxRoi2`: ROI indices for each pair
- `npixPair1`, `npixPair2`: Number of pixels in each ROI's mask
- `xposPair1`, `xposPair2`, `yposPair1`, `yposPair2`: Spatial coordinates

These are stored as flattened arrays representing the upper triangle of the full matrices, using `scipy.spatial.distance.squareform` for efficient storage.

#### Key Methods

1. **`load_data()`**: Loads ROI data, calculates correlations and distances, and prepares pair-wise comparisons.

2. **`getPairFilter()`**: Creates boolean filters for ROI pairs based on multiple criteria (correlation, distance, planes, etc.).

3. **`filterPairs()`**: Applies filters to extract relevant subsets of ROI pairs.

4. **Analysis Methods**:
   - `planePairHistograms()`: Creates histograms of ROI pairs across planes meeting correlation thresholds
   - `scatterForThresholds()`: Visualizes correlation vs. distance relationships
   - `cdfForThresholds()`: Plots cumulative distributions of correlations
   - `clusterSize()`: Analyzes cluster size distributions under different correlation thresholds
   - `distanceDistribution()`: Analyzes the distribution of distances between correlated ROIs
   - `roiCountHandling()`: Analyzes statistics about ROI removal and connection graphs

5. **Graph Analysis**:
   - Uses `getConnectedGroups()` to identify clusters of connected ROIs
   - Implements both simple and Maximal Independent Set (MIS) algorithms for ROI removal

#### Important Implementation Considerations

- The class inherits from `standardAnalysis`, which provides common functionality for analysis and figure saving.
- Correlation calculations are performed on the activity measure specified by `onefile` (default: "mpci.roiActivityDeconvolvedOasis").
- Distance measurements are in micrometers (μm), calculated from stack positions.
- The class supports filtering by plane indices, allowing analysis of specific subsets of planes.
- Many methods support both interactive visualization (`withShow=True`) and batch processing (`withSave=True`).

### Refactoring Considerations

When refactoring:
1. Consider separating data loading from analysis to improve modularity.
2. The pair filtering mechanism is central and used by most methods - ensure it remains efficient.
3. The connected components analysis (`getConnectedGroups()`) is a critical function that could be optimized.
4. Many visualization methods have similar parameters - consider standardizing these.
5. The class currently mixes data processing and visualization - these could be separated.

## clusterExplorer

The `clusterExplorer` class provides an interactive visualization tool for exploring ROI clusters identified by the `sameCellCandidates` analysis.

### Core Functionality

1. **Interactive Visualization**:
   - Displays ROI activity traces
   - Shows neuropil signals
   - Visualizes spatial relationships between ROIs
   - Presents cluster memberships

2. **User Interface**:
   - Slider for selecting seed ROIs
   - Text input for direct ROI selection
   - Interactive plots for selecting ROIs by clicking
   - Keyboard navigation (left/right arrows)

3. **Display Components**:
   - Activity traces panel: Shows calcium activity of ROIs in the selected cluster
   - Neuropil panel: Displays neuropil signals for the same ROIs
   - Cluster activity panel: Heat map visualization of all ROIs in the cluster
   - ROI outlines panel: Spatial visualization of ROI positions and planes

4. **Filtering and Selection**:
   - Filters ROI pairs based on correlation thresholds
   - Applies distance cutoffs
   - Allows plane selection
   - Supports minimum/maximum correlation thresholds

### Interface Design

The interface consists of:
- Four main panels arranged horizontally
- A slider at the bottom for ROI selection
- A text box for direct ROI number input
- Color-coding of ROIs by plane and selection status
- Interactive highlighting of selected ROIs across all panels

The visualization updates dynamically when:
- A new ROI is selected via slider or text input
- An ROI is clicked in any panel
- Keyboard navigation is used

### Beyond Visualization

Beyond just visualizing the output of `sameCellCandidates`, the `clusterExplorer`:
1. Calculates ROI hulls for spatial visualization (either min-max hulls or convex hulls)
2. Dynamically identifies clusters of connected ROIs based on the selected seed
3. Normalizes and filters activity traces for better visualization
4. Provides conversion between global ROI indices and within-plane indices

## clusterExplorerROICaT

The `clusterExplorerROICaT` class is a variant of `clusterExplorer` specifically designed to work with ROICaT cluster labels.

### Key Differences

1. **Cluster Definition**:
   - Instead of dynamically identifying clusters based on correlation and distance, it uses pre-defined cluster labels from ROICaT
   - Clusters are identified by their label rather than by seed ROI

2. **Selection Mechanism**:
   - The slider selects cluster labels rather than seed ROIs
   - All ROIs with the same cluster label are displayed together

3. **Implementation**:
   - Most of the visualization code is identical to `clusterExplorer`
   - The main difference is in the `getCluster()` method, which selects ROIs based on ROICaT labels rather than correlation/distance

4. **Use Case**:
   - Specifically designed for exploring the results of ROICaT, a tool for identifying putative same-cell ROIs across imaging planes
   - Less flexible than `clusterExplorer` but more directly applicable to ROICaT results