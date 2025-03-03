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
   - Loading spike data ("corrected" by default)
   - Optional neuropil correction (coefficient = 1.0 by default)

## Pair-wise ROI Analysis

For each pair of ROIs, the pipeline computes:

1. Activity correlation:
   - Pearson correlation coefficient between ROI time series
   - Computed efficiently using GPU-accelerated operations when available

2. Spatial metrics:
   - Euclidean distance \(d\) between ROI centers:
     \[ d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2} \]
   where \((x_1, y_1)\) and \((x_2, y_2)\) are the coordinates of the ROI centers in μm

## Cluster Identification

ROIs are grouped into clusters based on the following default parameters:

1. Correlation threshold:
   - Minimum correlation coefficient: 0.4
   - ROI pairs must have correlation \(r > 0.4\) to be considered connected

2. Spatial constraints:
   - Maximum distance: 20 μm
   - ROI pairs must be within 20 μm of each other to be considered connected
   - Optional minimum distance can be specified to exclude very close ROIs

3. Plane selection:
   - Default analysis includes planes 1-4
   - Configurable to analyze specific subset of planes

4. Additional filters:
   - Minimum ROI size (optional)
   - Maximum correlation threshold (optional)

The pipeline uses these parameters to construct an adjacency matrix, where ROI pairs meeting all criteria are considered connected. Connected components analysis is then performed to identify clusters of ROIs that are likely to represent the same cell across different planes. Note that two ROIs can be considered part of the same cluster if they are connected via other cells - even if they themselves are not connected. 

## Best ROI Selection

For each identified cluster, the pipeline selects the best representative ROI using the "max_sum_significant" method by default. This method:

1. Calculates the sum of significant spike events for each ROI in the cluster
2. Selects the ROI with the highest total significant activity
3. Marks all other ROIs in the cluster as redundant

The redundant ROIs are stored in the session data as `mpciROIs.redundant` for downstream analysis.

## Visualization and Quality Control

The pipeline includes several visualization tools for quality control and parameter optimization:

1. Correlation vs. distance scatter plots
2. Plane-pair histograms showing the distribution of connections across different imaging planes
3. Cluster size distribution analysis
4. **BEST TOOL!!!** Interactive cluster explorer for detailed examination of individual clusters

## Implementation Notes

The implementation uses several optimizations for computational efficiency:

1. Numba-accelerated functions for distance calculations and pair-wise operations
2. GPU-accelerated correlation coefficient calculations when available
3. Efficient storage of pair-wise data using condensed matrix format
4. Parallel processing for computationally intensive operations

## Usage

The pipeline is typically used as follows:

1. Initialize a B2Session object with imaging data
2. Create a SameCellProcessor instance with desired parameters
3. Process the data to identify clusters:
   - Constructs adjacency matrix based on correlation and distance thresholds
   - Identifies connected components as clusters
   - Selects best ROI in each cluster
   - Marks other ROIs as redundant
4. Save results:
   - Cluster data is saved using joblib
   - Redundant ROI boolean array is saved to `mpciROIs.redundant`
5. Use the ClusterExplorer to examine and validate the results

The parameters can be adjusted based on the specific requirements of the analysis and the characteristics of the imaging data.
