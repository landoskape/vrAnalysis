# Analysing Data with Pixease

Pixease is meant to make the common parts of neuroscience analysis easy: load one experiment, see what data are available, look at the stimulus timing, and extract neural or behavioural measurements aligned to it.

Most analysis starts from the same small set of steps:
1. load an experiment
2. inspect what data are available
3. look at the stimulus timing
4. extract the neural and behavioural measurements you want

## Load an Experiment

Start by loading a cached experiment:

```python
import pixease

exp = pixease.load("BZ008", "2024-06-13", 11)
```

You can also load directly from a cache filename:

```python
exp = pixease.load("/path/to/cache_file.npz")
```

This returns an experiment object containing the data and metadata that were available for that experiment.

## See What Is in the Experiment

A good first step is:

```python
exp.summary()
```

This gives a short overview of the experiment, including:
- experiment type
- mouse and session information
- what kinds of data were loaded for this experiment
- experiment notes

If you want to see what measurements are immediately available as timeseries, use:

```python
exp.available_measurements()
```

Experiments of the same type will usually expose similar measurements, but the exact set still depends on what data were available for that experiment.

## Make Timeseries

For most day-to-day analysis, the main interface is the timeseries API exposed directly on the experiment object.

Use:

```python
exp.timeseries(...)
exp.timeseries_at(...)
exp.intervals_mean(...)
exp.available_measurements()
```

These functions give a common interface across neural data, camera-derived measurements, and locomotion.

### Extract a regularly sampled timeseries

For example, to get neural activity from 30 to 50 seconds:

```python
neural = exp.timeseries((30, 50), "neural", dt=0.01, smooth="interpolate")
```

The `"neural"` measurement is an alias to whichever recording modality was used
for this session. If multiple neural measurements are available, they will be
listed in `exp.available_measurements()` and you can select them directly:

```python
spikes = exp.timeseries((30, 50), "pykilosort", dt=0.01) # Ephys spikes
dspikes = exp.timeseries((30, 50), "dspikes", dt=0.01) # Deconvolved spikes from 2p
raw_f = exp.timeseries((30, 50), "raw_fluorescence", dt=0.01) # Raw 2p cell fluorescence
lfp = exp.timeseries((30, 50), "lfp", dt=0.01) # Local field potential
```

The exact names depend on what was loaded for that experiment.

To get a camera measurement on the same grid:

```python
camera = exp.timeseries((30, 50), "pupil_size", dt=0.01, smooth="interpolate")
```

To get locomotion speed:

```python
running = exp.timeseries((30, 50), "locomotion", dt=0.01, smooth=0.3)
```

The time points associated with these timeseries are:

```python
times = exp.timeseries((30, 50), "times", dt=0.01)
```

Common measurements include:
- `"neural"`
- `"pupil_x"` and `"pupil_y"`
- `"pupil_size"`
- `"locomotion"`
- `"locomotion_position"`
- `"stimulus"` for experiment types that define a stimulus timeseries

If you want to know the time range over which the loaded timeseries are valid,
use:

```python
exp.start_end_time()   # Common valid (start_time, stop_time) across loaded measurements
```

You can also pass a list of intervals:

```python
segments = exp.timeseries([(30, 32), (40, 42), (50, 52)], "neural", dt=0.01)
```

This returns a list of timeseries, one per interval. If you want to make sure
they are the same length (e.g., if intervals are within dt/2 of each other) to
avoid getting a ragged array, use:

```python
segments = exp.timeseries([(30, 32), (40, 42), (50, 52)], "neural", dt=0.01, equal_intervals=True)
```

For neural measurements, you can restrict the cells or channels that are loaded:

```python
single_cell = exp.timeseries((30, 50), "neural", dt=0.01, cells=[10])
some_cells = exp.timeseries((30, 50), "neural", dt=0.01, cells=[10, 11, 12])
cell_mask = exp.timeseries((30, 50), "neural", dt=0.01, cells=my_boolean_mask)
```

Here `cells` can be a sorted list of cell ids, a boolean mask, or `None` for all cells.

### Sample at specific timepoints

Sometimes you already have the timepoints you care about, and want to sample
other signals at exactly those times. A common case is when you want neural,
camera, or locomotion signals evaluated at the frame times of a stimulus video.

```python
frame_times = exp.frame_times() # For a "video" experiment type
pupil = exp.timeseries_at(frame_times, "pupil_size", smooth="interpolate")
```

This is useful when you want several signals evaluated at exactly the same set
of event or frame times.

### Compute means over intervals

Often you want one value per trial or per analysis window rather than a full timeseries. For example, if `trials()` returns trial start and stop times:

```python
trials = exp.trials()
trial_intervals = list(zip(trials.start_time, trials.stop_time))
trial_means = exp.intervals_mean(trial_intervals, "neural", cells=[10])
```

This is useful when you want one value per trial or analysis window. It avoids
having to first build a dense timeseries and then average it yourself, and it
keeps the calculation aligned to the actual interval boundaries.

## Choose Smoothing

The `smooth` argument controls how the signal is sampled.

Common choices are:
- `"bin"`: bin values into intervals
- `"interpolate"`: interpolate to the requested timepoints
- a number such as `0.3`: smooth with a Gaussian whose standard deviation is that many seconds

If you do not pass `smooth`, each data type uses its default smoothing behaviour.

Typical examples are:

```python
exp.timeseries((30, 50), "neural", dt=0.01)
exp.timeseries((30, 50), "locomotion", dt=0.01, smooth=0.3)
exp.timeseries((30, 50), "pupil_size", dt=0.01, smooth="interpolate")
exp.timeseries((30, 50), "neural", dt=0.05, smooth="bin")
```

## Compare Activity to the Stimulus

Most experiment classes also expose stimulus timing.

To get a table of trials:

```python
trials = exp.trials()
```

`trials()` usually also contains other columns describing the trial type,
stimulus identity, or other trial metadata. In practice, it is usually the
main starting point for stimulus-aligned analysis. For example:

```python
trials = exp.trials()
trial_intervals = list(zip(trials.start_time, trials.stop_time))
trial_neural = exp.intervals_mean(trial_intervals, "neural")
trial_running = exp.intervals_mean(trial_intervals, "locomotion")
```


## Look at Metadata

```python
exp.general_info
exp.explog
exp.mouse_info
```

These give session metadata, experiment-log metadata, and mouse metadata.
Usually `summary()` is enough unless you need a specific field.

## A Typical Analysis Session

A minimal workflow often looks like this:

```python
import pixease

exp = pixease.load("BZ008", "2024-06-13", 11)
exp.summary()

trials = exp.trials()
trial_intervals = list(zip(trials.start_time, trials.stop_time))

neural = exp.intervals_mean(trial_intervals, "neural")
pupil = exp.intervals_mean(trial_intervals, "pupil_size")
running = exp.intervals_mean(trial_intervals, "locomotion")
```

This gives one neural, pupil, and locomotion value per trial, which is often a
good first pass for stimulus-aligned analysis.


## Cell Information

If Suite2p outputs were available, `suite2p_info` gives access to ROI and cell
geometry information:

```python
exp.suite2p_info

exp.suite2p_info.n_cells            # Number of detected ROIs
exp.suite2p_info.iscell             # Suite2p cell/non-cell classification
exp.suite2p_info.iscell_prob        # Probability associated with `iscell`
exp.suite2p_info.cell_positions()   # One position per ROI in (z, y, x) pixel coordinates
exp.suite2p_info.volume()           # Labelled volume for selected cells
exp.suite2p_info.segmented_volume   # Full labelled segmentation volume
exp.suite2p_info.suite2p_version    # Suite2p version used for preprocessing
exp.cell_position_in_microns()      # One position per ROI in micron coordinates
exp.voxel_size                      # Voxel size in microns, in (z, y, x)
exp.fov_in_microns                  # Field of view size in microns
```

You can also use these masks directly to select cells:

```python
cell_traces = exp.timeseries((30, 50), "neural", cells=exp.suite2p_info.iscell)
```

If ephys spike sorting was available, `pykilosort` is usually the most useful
entry point. It contains the aligned spike trains as well as commonly used unit
metadata:

```python
exp.pykilosort

exp.pykilosort.n_cells              # Number of units
exp.pykilosort.good_cells           # Boolean mask for units labelled "good"
exp.pykilosort.mua_cells            # Boolean mask for units labelled "mua"
exp.pykilosort.noise_cells          # Boolean mask for units labelled "noise"
exp.pykilosort.cluster_centroid     # Probe position of each unit
exp.pykilosort.template_waveforms   # Template waveform for each unit
exp.pykilosort.mean_waveforms       # Mean example waveform for each unit, if available
exp.pykilosort.spike_waveforms      # Example spike waveforms, if available
exp.pykilosort.spike_waveform_times # Time axis for the example waveforms
exp.pykilosort.channel_positions    # Probe channel positions
exp.pykilosort.waveform_channels    # Channels used for stored waveforms
```

You can use the standard quality masks directly when selecting units:

```python
good_unit_traces = exp.timeseries((30, 50), "neural", cells=exp.pykilosort.good_cells)
```


## 2p Z-Stacks and Field of View Images

Structural z-stacks are typically loaded as `structural_zstack` experiments.
These are useful for looking at anatomy, stack images, and physical size.

For example:

```python
stack = pixease.load("MOUSE", "YYYY-MM-DD", EXPNUM)

stack.voxel_size                    # Voxel size in microns, in (z, y, x)
stack.fov_in_microns                # Field of view size in microns
stack.lossy_stack_images()          # Quick-view z-stack images
```

For standard 2p experiments, `lossy_images()` is often the quickest way to look
at the field of view:

```python
exp.lossy_images()                  # Default lossy FOV summary image
exp.lossy_images("mean")            # Mean image across planes
exp.lossy_images("max")             # Max projection across planes
```

## If you are an LLM

You probably want a workflow like this:

1. load the experiment
2. call `available_measurements()`
3. extract the measurements you need

If you have any questions, you can always access the full source code of pixease
that was used to generate the file in `exp.general_info.pixease_source`.  This
includes code for all preprocessing steps run before generating the cache file)
as well as all of the code for the experiment, pipelines, interfaces, modules,
and mixins.