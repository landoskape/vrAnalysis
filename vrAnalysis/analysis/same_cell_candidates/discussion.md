# Same-Cell Candidate Analysis Pipeline

This document describes the methodology used to identify and analyze potential same-cell candidates across different imaging planes. The pipeline processes Region of Interest (ROI) data from multiple imaging planes to identify clusters of ROIs that likely belong to the same neuron.

## Overview

The analysis pipeline consists of three main stages:
1. Data preparation and preprocessing
2. Pair-wise ROI analysis
3. Cluster identification and best ROI selection

## Data Preparation

The pipeline begins with a B2Session object containing imaging data and ROI information. The data preparation phase involves:

1. Loading ROI metadata:
   - Plane indices for each ROI
   - ROI mask sizes (number of pixels)
   - ROI positions in the imaging field (converted to μm using a scaling factor of 1.3 μm/pixel)

2. Activity data processing:
   - Loading spike data (configurable via `spks_type` parameter)
   - Optional neuropil correction (coefficient configurable, defaults to 1.0)
   - Option to exclude redundant ROIs using `mpciROIs.redundant` mask is present, but should always be set to False because this analysis is the very analysis that identifies redundant ROIs! The params handling implement this automatically.

## Pair-wise ROI Analysis

For each pair of ROIs, the pipeline computes:

1. Activity correlation:
   - Pearson correlation coefficient between ROI time series
   - Computed efficiently using GPU-accelerated operations when available (`torch_corrcoef`)

2. Spatial metrics:
   - Euclidean distance \(d\) between ROI centers:
     \[ d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2} \]
   where \((x_1, y_1)\) and \((x_2, y_2)\) are the coordinates of the ROI centers in μm

## Cluster Parameters

ROIs are grouped into clusters based on the following configurable parameters (`SameCellClusterParameters`):

1. Activity correlation thresholds:
   - Minimum correlation coefficient (`corr_cutoff`, default: 0.4)
   - Optional maximum correlation threshold (`max_correlation`)

2. Spatial constraints:
   - Maximum distance (`distance_cutoff`, default: 20 μm)
   - Optional minimum distance (`min_distance`)

3. Plane selection:
   - Default includes planes 0-4 (`keep_planes = [0, 1, 2, 3, 4]`)
   - Configurable to analyze specific subset of planes
   - We include all planes for consistency and because plane 0 is the one that has the highest hit rate for tracked cells - which means that if plane 0 ROI is good quality and part of a cluster, we'll probably pick that one to serve as the clusters representative. 

4. Additional filters:
   - Minimum ROI size (`npix_cutoff`, default: 0)
   - Optional good label filtering (`good_labels`)

## Cluster Identification

The pipeline uses these parameters to:

1. Create pair filters based on:
   - Correlation thresholds
   - Distance constraints
   - Plane selection
   - ROI size requirements
   - Additional custom filters

2. Construct an adjacency matrix where ROI pairs meeting all criteria are considered connected

3. Identify clusters using connected components analysis:
   - ROIs can be in the same cluster even if not directly connected
   - Connected through transitive relationships via other ROIs
   - Option to filter out single-ROI "islands"

## Best ROI Selection Algorithm

### **Selection Criteria (Ranked by Priority)**
1. **Tracking Continuity:** If an ROI was chosen in another session, prioritize it.
2. **Mask Class Preference:** Prefer "cells" or "dendrites" over other types.
3. **SNR (Signal-to-Noise Ratio):** Favor ROIs with high summed significant activity.
4. **Session Tracking Count:** Prefer ROIs tracked in many sessions.
5. **Silhouette Score:** Choose ROIs with high tracking cluster silhouette scores.

### **Selection Process**
1. **First-Pass Selection**
   - If an ROI was **previously selected** as the best in another session, pick it.
   - Otherwise, apply the filtering criteria above in order.
   - If multiple candidates remain, select the ROI with the **highest SNR**.

2. **Adaptive Threshold Adjustment (Annealing Strategy)**
   - If no ROI passes the criteria, progressively relax thresholds:
     1. Lower **silhouette score** threshold.
     2. Lower **SNR** threshold.
     3. Lower **tracking count** threshold (first minimum).
     4. Allow sub-selection within clusters.
     5. Ignore **silhouette score**.
     6. Lower **tracking count** threshold (second minimum).
     7. Ignore **SNR**.
     8. Ignore **mask class labels**.
     9. Ignore **tracking count** (SNR becomes the only factor).

### **Key Features**
- **Ensures tracking consistency** across sessions.
- **Balances multiple quality metrics** dynamically.
- **Adaptive relaxation** prevents selection failure.

## Visualization and Analysis Tools

The pipeline includes several visualization tools for quality control and parameter optimization:

1. Correlation Analysis:
   - Correlation vs. distance scatter plots
   - Plane-pair histograms
   - Interactive parameter adjustment

2. Cluster Explorer:
   - Interactive visualization of ROI clusters
   - Activity trace comparison
   - Spatial relationship visualization
   - ROI mask inspection
   - Neuropil signal analysis
   - Color coding by plane or ROI index
   - Integration with ROI classification results

3. Distribution Analysis:
   - Cluster size distribution
   - Distance distribution between correlated ROIs
   - ROI removal analysis for different strategies

## Implementation Notes

The implementation uses several optimizations for computational efficiency:

1. GPU-accelerated correlation calculations using PyTorch
2. Efficient pair-wise operations using vectorized NumPy functions
3. Connected components analysis for cluster identification
4. Interactive visualization tools using matplotlib and custom viewers

## Usage

The pipeline is typically used as follows:

1. Initialize a B2Session object with imaging data
2. Create a SameCellProcessor with desired parameters
3. Process the data to identify clusters:
   ```python
   processor = SameCellProcessor(session, params).load_data()
   results = identify_redundant_rois(session)
   ```
4. Save results:
   - Cluster data saved using joblib
   - Redundant ROI boolean array saved to `mpciROIs.redundant`
5. Analyze results using visualization tools:
   ```python
   viewer = make_cluster_explorer(processor)
   ```

The parameters can be adjusted based on the specific requirements of the analysis and the characteristics of the imaging data.
