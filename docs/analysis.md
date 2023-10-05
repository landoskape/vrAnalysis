# vrAnalysis Documentation: analysis

This is some documentation for the analysis code of vrAnalysis. The main goal
of this file is to remind me what I've written so I don't forget. There may be 
room for explaining how it works, but at the moment, I'll just focus on using 
it.

## Basic Structure
The analysis of data is organized into modules that define analysis objects
which group together analyses that are about the same thing and usually depend
on the same basic data. This is very useful, because not only does it 
streamline the data loading and analysis process, but it provides a good 
organization for me to remember what analyses I've performed. 

The analysis library is in it's own directory which you can find 
[here](../vrAnalysis/analysis) if you want to look at the code in detail. 

Each analysis module defines an analysis object which is a child of the 
`standardAnalysis` class defined 
[here](../vrAnalysis/analysis/standardAnalysis.py). The standard analysis code
is pretty simple - it's job is just to add a few methods that are commonly 
used by most analyses. 

Key methods: 
1. `analysisDirectory()`: gets whatever path is hardcoded in the 
    [fileManagement](../vrAnalysis/fileManagement.py) module.
2. `saveDirectory()`: appends the name of the analysis type and the name of
    the specific analysis to the main analysis directory and returns it
3. `saveFigure()`: saves an open matplotlib figure in the saveDirectory()

## Analysis Modules
Here is a (hopefully mostly complete) list of analysis modules. It explains
the main analyses to do with each module and the standard usage for them. 

All modules require you to import the analysis package:
```python 
from vrAnalysis import analysis
from vrAnalysis import database # this is also used
```

### Same Cell Candidates
This analysis module attempts to identify imaging ROIs that are probably the
same cell. This can happen for two reasons: 
1. Suite2p sometimes oversplits cells within a plane - either erroneously or 
   because they are dendritic ROIs and far apart from each other.
2. Cells have ROIs in multiple planes, either because the planes are too close
   to each other or the cell's dendrite extends and is captured by suite2p. 

To open a sameCellCandidate analysis object, create a vrExperiment object and
choose whatever filters you want. 
```python
import random

# choose a session randomly using the vrDatabase module
vrexp = random.choice(vrdb.iterSessions(imaging=True, vrRegistration=True))
print(vrexp.sessionPrint()) # show which session you chose

# create sameCellCandidates object
# keepPlanes determines which planes to use, others will be considered the flyback and ignored in all further analyses
scc = analysis.sameCellCandidates(vrexp, keepPlanes=[1,2,3,4]) 
```

To observe clusters of cells based on a variety of criteria, use this block. 
It opens up a mediocre matplotlib GUI to look through ROI clusters. 
```python
clusterExplorer = analysis.clusterExplorer(scc, corrCutoff=0.5, maxCutoff=None, distanceCutoff=30, minDistance=None, keepPlanes=[1,2,3,4])
```

To look at how many pairs of ROIs are found across planes that are near each
other and above a certain correlation, use this:
```python
# withSave and withShow can be reversed if doing this programmatically to save figures for many sessions
scc.planePairHistograms(corrCutoff=[0.5, 0.6, 0.7, 0.8], distanceCutoff=50, withSave=False, withShow=True) 
```        

To look at a scatter plot of the distance between ROIs vs. their correlation 
coefficient, use this. Beware of the alpha value.
```python
scc.scatterForThresholds(keepPlanes=[1,2,3,4], distanceCutoff=250);
``` 








