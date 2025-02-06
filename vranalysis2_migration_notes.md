# Feature Comparison: vrExperiment vs B2Session

This document outlines key features present in `vrExperiment` that are not currently implemented in `B2Session`.

## 1. Behavioral Data Analysis

`vrExperiment` includes methods for handling behavioral data synchronization:
```python
python
def getBehaveTrialIdx(self, trialStartFrame)
def groupBehaveByTrial(self, data, trialStartFrame)
def get_frame_behavior(self, speedThreshold=5, use_average=True, return_speed=False)
def get_position_by_env(self, speedThreshold=5, use_average=True, idx_ignore=-100, return_speed=False)
```

These methods handle synchronization between imaging frames and behavioral data, including position tracking and trial structure.

## 2. ROI Volume/Mask Operations

Methods for working with ROI masks across different imaging planes:
```python
def getMaskVolume(self, cat_planes=False, keep_planes=None)
def getNumROIs(self, keep_planes=None)
def idxToPlanes(self, keep_planes=None)
def computeVolume(self, planeIdx=None)
```

These provide ways to work with ROI masks across different imaging planes and create volumetric representations.

## 3. Reference Image Processing

Methods for handling centered and filtered reference images:

```python
def centeredReferenceStack(self, planeIdx=None, width=15, fill=0.0, filtPrms=None)
def centeredMaskStack(self, planeIdx=None, width=15, fill=0.0)
```

These methods handle creation of centered and filtered reference images and masks.

## 4. Data Filtering

Built-in signal filtering capabilities:

```python
# Used within various methods
helpers.butterworthbpf(cReference, lowcut, highcut, order=filtPrms[2], fs=filtPrms[3])
```

## 5. Cache Management

More granular cache clearing options:

```python
def clearBuffer(self, *names)
```

B2Session only has `clear_cache()` which clears everything.

## 6. Plane-specific Operations

Methods for working with ROI indices within specific planes:

```python
def getRoiInPlaneIdx(self, idx)
def getRoiRange(self, idx)
```

## 7. Coordinate System Utilities

Utilities for working with centered coordinate systems:

```python
def createCenteredAxis(self, numElements, scale=1)
def getyref(self, yCenter)
def getxref(self, xCenter)
```

## Modernization Notes

To bring B2Session to feature parity while maintaining its modern architecture:

1. Add behavioral data synchronization capabilities
2. Implement ROI volume/mask operations
3. Add reference image processing functionality
4. Include coordinate system utilities
5. Add more granular cache management

The implementation should follow B2Session's patterns:
- Use of dataclasses
- Type hints
- Clear property definitions
- Inheritance from abstract base classes
- Clear separation of concerns

Handling spkmaps:
I want the session objects to handle spkmaps -- I want the spkmaps with smoothing etc to be saved as onedata,
but of course we need them to be recoverable too. So it's important that the parameters used for saving them
are also present -- including trial information! (I only save "full" trials). 