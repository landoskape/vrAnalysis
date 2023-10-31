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
vrdb = database.vrDatabase() # open a database manager
```

## Place Cell Multi Session -- [link to module](../vrAnalysis/analysis/placeCellMultiSession.py)
Analyzing place cells across sessions requires a "tracker" object which 
processes ROI tracking data from the ROICaT package. The tracker module is 
[here](../vrAnalysis/tracking.py). Tracker objects operate for an entire mouse
so open one up for whatever mouse you want to analyze, then use that as input
to the `placeCellMultiSession` analysis object (`pcm`). Here, I'm using 
``autoload=False`` because otherwise it'll analyze all the place cell data 
from all the tracked sessions, which takes a while. 
```python
track = tracking.tracker('ATL022') # get tracker object for mouse
pcm = analysis.placeCellMultiSession(track, autoload=False) # open up place cell multi session analysis object (don't autoload!!!)
pcm.track.session_table() # this prints a dataframe of the sessions in the tracker!
```

The simplest analysis to do with the `pcm` object is to plot snakes across 
sessions using tracked cells that are reliable on a particular session (in a
particular environment). Here, I plot snakes from environment 4 in sessions 12
to 16, sorting and choosing ROIs by their activity on session 16. The other 
inputs are similar to those in place cell single session, which is described
below. Note that this will load whatever sessions you request, which can take
a bit of time, as I mentioned. This method calls another method named 
`make_snake_data` which retrieves the data used by `plot_snake`. 
```python
# choose range of sessions based on dataframe, pick environment, and plot snake
envnum = 4
idx_ses = range(12,17)
sortby = 16 # which session to sort by
pcm.plot_snake(envnum, idx_ses=idx_ses, sortby=sortby, cutoffs=(0.5, 0.8), method='max', normalize=10, rewzone=True, interpolation='none', withShow=True, withSave=False)
```

*NOTE THAT THIS SECTION IS NOW OBSOLETE, BUT WILL BE REPLACED!!!*
This module can also analyze how place fields change across sessions. The
`plot_pfplasticity` method does so by making a histogram of the changes in 
place field position (with `'max'` or `'com'`, see method for details) across
sessions. You can do it for all cells or split by control or red cells with
`split_red`. You can plot absolute difference or plot actual difference 
(including negative changes) with `abs_val`. This calls `measure_pfplasticity`
to make the data.
```python
pcm.plot_pfplasticity(self, envnum, idx_ses=idx_ses, cutoffs=(0.5, 0.8), method='max', absval=True, split_red=False, withShow=True, withSave=False):
```

This contains a program for measuring place field plasticity across sessions 
as well. *This needs more explanation!!!!* but for now I'll just show the main
workflow:
```python
track = tracking.tracker('CR_Hippocannula7') # get tracker object for mouse
pcm = analysis.placeCellMultiSession(track, autoload=False) # open up place cell multi session analysis object (don't autoload!!!)
for envnum in [1, 2]:
    for present in ['r2', 'pc']:
        pcm.plot_pfplasticity(envnum, idx_ses=None, cutoffs=(0.3, 0.6), present=present, split_red=True, withShow=False, withSave=True)
```

## Place Cell Single Session -- [link to module](../vrAnalysis/analysis/placeCellSingleSession.py)
This analysis module is used for analyzing place fields in single sessions
(with some support for cross-session analysis as methods). The `pcss` object
loads behavioral and spiking data, remaps it to a spatial map (rather than the
x-axis being time, the x-axis is space and the data is warped from temporal to
spatial coordinates). It automatically divides data by which virtual 
environment the mouse was in. After loading data, it has methods for analyzing
spatial reliability of spiking, creating cross-validated "snake plots" of 
neural activity in different environments, and more.

The main uses of `pcss` objects are to:
1. Generate spatial maps of behavioral data (occupancy, speed, licking).
2. Generate spatial maps of firing patterns for each neuron.
3. Measure reliability for each ROI within each environment.
4. Create cross-validated snake plots within and across environments.
5. Compare data across sessions using extensions in the module.


To open a placeCellSingleSession analysis object (`pcss`), do this:
```python
import random

# choose a session randomly that has registered imaging data
vrexp = random.choice(vrdb.iterSessions(imaging=True, vrRegistration=True))
print(vrexp.sessionPrint()) # show which session you chose

# create place cell analysis object
pcss = analysis.placeCellSingleSession(vrexp)
```

There are several options for creating a `pcss` object. Most of these options
are utilized in the [`functions`](../vrAnalysis/functions.py) module, for more
explanation look there. 
- keepPlanes: a list of planes to keep, usually ignoring the flyback plane
- distStep: a tuple describing how to manage spatial resolution. It's a little
  complicated so here's a full explanation. If only one value is provided, no
  downsampling or gaussian smoothing is used. If only two values are provided,
  then the second value is used to set the standard deviation of the
  gaussian smoothing.
  - the first value describes the initial spatial resolution to measure
    occupancy, speed, licks, and spiking maps with.
  - the second value describes how much to downsample the aforementioned maps.
    If provided, it must be an integer multiple of the first value.
  - the last value describes the standard deviation of a gaussian smoothing
    kernel used to smooth the data (except the lick map).
- speedThreshold: as is standard in the field, this sets a threshold at how
  fast the mouse is moving (in centimeters) for including data points.
- standardizeSpks: if True, will use a median/normalization (subtract by the
  median, divide by the standard deviation). It's a good idea to use this
  because deconvolved spikes don't have clear units.

To retrieve snake plots from a session, there are two methods. One makes train
and test snakes for each environment independently. The ROIs used for the 
snakes are determined based on their reliability on training trials, and the
sort of both train & test snakes is determined based on training trials.
```python
# envnum: requested environments to make snake for, if None, uses all
# with_reliable: whether or not to use only reliable cells as defined by cutoffs
# cutoffs: [0]=relmse cutoff, [1]=relcor cutoff (see below)
# method: how to sort -- if 'max', uses max activity, if 'com', uses center of mass
train_snake, test_snake = pcss.make_snake(envnum=None, with_reliable=True, cutoffs=(0.5, 0.8), method='max')
```

Alternatively, you can retrieve cross-validated snakes comparing activity 
across environments. This retrieves an (N x N) grid of snakes where the ROIs 
and sort in each row are determined by the environment index of the row, and 
each column uses trials from the environment index of the column. Train/test 
splits are used for the diagonal, and full trial lists are used for 
off-diagonal. For example, row `i` column `j` uses ROIs that are reliable in 
environment `i`, sorts them based on activity in environment `i`, and plots
their activity on environment `j`.
```python
# kwargs same as above, automatically uses all environments in the session. 
remap_snakes = pcss.make_remap_data(with_reliable=True, cutoffs=(0.5, 0.8), method='max')
```

To plot the data, there are convenience methods. These use the standard 
`withSave=False` and `withShow=True` kwargs so you can choose whether to save
or show the figure (or both). 
```python
# normalize: determines what colormap limits to use (-norm, +norm), if 0 uses max activity
# rewzone: whether or not to plot the reward zone as a patch
# interpolation: how to make the imshow plot
# withShow: whether or not to show the figure
# withSave: whether or not to save the figure using the standard analysis method (see above)
# force_single_env: whether or not to make a remap_snake_figure even if there is only a single environment in the session 
plot_snake(self, envnum=None, with_reliable=True, cutoffs=(0.5, 0.8), method='max', normalize=0, rewzone=True, interpolation='none', withShow=True, withSave=False)
plot_remap_snakes(self, with_reliable=True, cutoffs=(0.5, 0.8), method='max', normalize=0, rewzone=True, interpolation='none', force_single_env=False, withShow=True, withSave=False)
```

To get a list of which ROIs are reliable for each environment, `pcss` uses two
measures of reliability, one based on mean-square error and one based on 
correlation. For both methods, the trials are divided into training and 
testing sets, and the average activity (across trials) is compared. 
For MSE, a measure based on deviance is used:
$$\large relmse = 1 - \frac{\sum_i(testProfile_i - trainProfile_i)^2)}{\sum_i(testProfile_i - \frac{1}{N}\sum_jtrainProfile_j)^2}$$

For COR, the pearson correlation is used:
$$\large relcor = correlation_{pearson}(trainProfile, testProfile)$$

To retrieve indices of reliable cells in each environment, use this:
```python
# envnum is a list of environment indices
# cutoffs is a two-element tuple of reliability cutoffs
idx_reliable = get_reliable(envnum, cutoffs=None)
```

There are additional tools for analyzing data across sessions that are in the 
[module](../vrAnalysis/analysis/placeCellSingleSession.py). Generally, these 
tools require `pcss` objects for many sessions. This can take a bit of time 
and memory, so I start by doing the following:
```python
# this is a session iterable (a list of vrExperiment objects) that I want to 
# analyze, in this case I'm ignoring mouse ATL023 and only getting sessions 
# that used an experimentID > 0. 
ises = vrdb.iterSessions(imaging=True, vrRegistration=True, mouseName=('ATL023','!='), experimentID=(0, '>'))

# then for each session, I make a `pcss` object. 
ipcss = []
for ses in tqdm(ises):
    ipcss.append(analysis.placeCellSingleSession(ses))
    ses.clearBuffer() # clearing the buffer removes data loaded in each session object which is unnecessary and usually quite large
```

To plot the difference in the fraction of reliable cells between red cells 
(that express tdTomato and have a conditional KO) and control cells, use this:
```python
# make sure to feed in the preloaded `ises` and `ipcss` objects. 
# cutoffs determine the reliability cutoff (see above)
# include_manual determines whether to use manually annotated cells in the red idx. 
# note: you can also request to only use the suite2p index -- see the module for details
analysis.plot_reliable_difference(ises=ises, ipcss=ipcss, cutoffs=(0.5, 0.8), include_manual=True, withSave=False, withShow=True)
```

To plot the difference in the fraction of reliable cells between red cells 
(that express tdTomato and have a conditional KO) and control cells, use this.
This will make a separate plot for each mouse in `ises`/`ipcss` and 
automatically make different curves for each environment.
```python
# make sure to feed in the preloaded `ises` and `ipcss` objects. 
# cutoffs determine the reliability cutoff (see above)
# include_manual determines whether to use manually annotated cells in the red idx. 
# note: you can also request to only use the suite2p index -- see the module for details
analysis.plot_reliable_difference(ises=ises, ipcss=ipcss, cutoffs=(0.5, 0.8), include_manual=True, withSave=False, withShow=True)
```

To plot the fraction of red reliable cells and control reliable cells for each
environment, use this:
```python
# same input structure as above
analysis.plot_reliable_fraction(ises=ises, ipcss=ipcss, cutoffs=(0.5, 0.8), include_manual=True, withSave=False, withShow=True)
```


## Session Stats -- [link to module](../vrAnalysis/analysis/sessionStats.py)

This analysis module is used for analyzing basic session data using a session 
iterable. As of now, this entire module is based on non-class methods, which I
hope to change in the near future but I needed them quickly and this was 
easiest. 

The data are generally produced by the `sessionStats` method as follows. It 
measures various things including which environments were used in each 
session, how many trials per environment, how many ROIs per session, and how 
many of those ROIs are considered red, according to the standard red detection
method or by suite2p alone. It organizes the outputs by mouse and envrionment.
```python
ises = vrdb.iterSessions(imaging=True, vrRegistration=True) # get iterable of all registered imaging sessions
stats = analysis.sessionStats(ises=ises, keepPlanes=[1,2,3,4], include_manual=True, use_s2p=False, s2p_cutoff=0.65) # measure session stats
miceInSession, env_counter, trial_counter, total_cell_count, red_cell_count = stats # unpack tuple
```

To plot the stats about which environments were visited and in how many trials
by each mouse, use this. Additional input arguments are the same as 
`sessionStats`.
```python
analysis.plot_environmentStats(ises=ises, keepPlanes=[1,2,3,4])
```

To plot the stats about how many ROIs were imaged and how many of those are 
red in each session, use this. Additional input arguments are the same as 
`sessionStats`.
```python
analysis.plot_roiCountStats(ises=ises, keepPlanes=[1,2,3,4])
```

## Same Cell Candidates -- [link to module](../vrAnalysis/analysis/sameCellCandidates.py)

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








