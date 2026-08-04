# Figure-script design principles

How the figure modules are built, and why. `figure2/` is the reference implementation;
`figure1.py`, `figure3.py`, and `figure4.py` predate it.

The goal is that **each fact about a panel is stated once**, in the place that owns it. Most of
the bulk in the old modules was not drawing code — it was the same fact restated in three or four
places, which is both the reading cost and the source of drift.

---

## 1. A panel is a Viewer. There is no caller function.

The old shape was a `Viewer` subclass *plus* a public function that re-declared every knob as a
keyword argument, re-documented it, and re-assigned it with a typed `update_*` call. Four
statements per knob; `figure2.py` had 161 `viewer.update_*` calls, `figure1.py` has 192.

The caller existed only to offer `return_syd_viewer`. A notebook can do that itself:

```python
viewer = ModelPerformanceViewer(results, activity_parameters_name="std", fontsize=12)
viewer.show()                     # interactive
fig = viewer.plot(viewer.state)   # static, for save_figure
```

So: **`__init__` keyword arguments are the widget defaults.** One statement per knob.

```python
def __init__(self, results, *, fontsize: float = 12.0, ylim=(-0.03, 0.12), ...):
    self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)
    self.add_float_range("ylim", value=ylim, min=-1.0, max=2.0, step=0.001)
```

This also deletes a class of bug: the old callers had to seed data-selection params, then call
`refresh_data` explicitly, because "pre-deploy `update_*` may not fire `on_change`". With the
values present before the first `refresh_data`, that whole dance disappears.

**Migration note.** Where you do need to set a parameter generically, `Viewer.set_parameter_value(name, value)`
is type-agnostic and fires callbacks — you never need to dispatch between `update_float` /
`update_boolean` / `update_float_range`.

---

## 2. Style is widgets, not global state

No module sets `plt.rcParams`. Importing a figure module must not change how the rest of the
notebook draws. Every font size is a `fontsize` widget threaded explicitly into `set_xlabel`,
`set_title`, `format_spines(tick_fontsize=...)`, and `legend(fontsize=...)`.

If a panel has text whose size you cannot control from its state, that is a bug, not a default.

---

## 3. Data selection comes from the results, not from hard-coded lists

A panel should offer exactly the variations its results were stored over. That information is
already on the aggregator — don't re-derive it from `VALID_*` constants that can drift.

```python
self.selection_names = add_data_selection_widgets(self, results, skip=("model_name",), defaults=kwargs)
...
selection = data_selection(state, self.results, self.selection_names)
scores = results.sel(model_name=name, avg_by_mouse=True, **selection)
```

- `param_axes` gives the options. Tuple-valued axes are skipped automatically (the panel supplies
  its own label encoding — see `latents.py`'s `MODEL_PAIR_LABELS`); two-valued boolean axes become
  checkboxes.
- `skip=` is for axes the panel fixes by design (`model_name`, when the panel already knows which
  models it draws).
- Several aggregators can be passed at once; the widgets cover their union, and `data_selection`
  forwards to each `sel` only the axes *that* aggregator declares.
- Wire `on_change` from the **returned names**, never from a hard-coded tuple.

### The config is the source of default values

`_param_grid()` says what is *swept*. The config class says what each parameter *is*. These
diverge: `RegressionConfig` still declares `spks_type: SpksTypes = "sigrebase"` after
`_param_grid()` stopped sweeping it. So a widget's starting value resolves as:

> caller default → the config's field value (if selectable) → the axis's first option

Never hard-code a default like `spks_type: str = "sigrebase"` in a viewer signature. That is a
second copy of a fact the config already owns, and it silently goes stale.

### `require=` for values used outside `results.sel`

Some panels pass a parameter to something other than `sel` — `get_model_predictions`,
`registry.get_population`. Those need a concrete value whether or not the config sweeps it.
Declaring it `require=` guarantees a widget exists (pinned to a single option when unswept), so
`state[name]` and `on_change(name, ...)` work without any caller checking which axes are live:

```python
# These go to get_model_predictions, not results.sel, so we need a value either way.
self.selection_names = add_data_selection_widgets(
    self, results, skip=("model_name",), defaults=selection_defaults,
    require=("spks_type", "activity_parameters_name"),
)
```

Dropping an axis from a `_param_grid()` should never break a panel that still needs the value.

---

## 4. Shared widget bundles, following the `legends.py` shape

Groups of knobs that recur across panels live in `panels.py` as an
`add_*_widgets(viewer, **defaults)` / `draw_*(ax, state, ...)` pair. The adder registers the
widgets with the caller's defaults; the drawer consumes them from `state`. `legends.py` is the
original of this pattern; `panels.py` generalizes it:

| bundle | knobs | drawer |
| --- | --- | --- |
| `add_data_selection_widgets` | one per param axis | `data_selection` |
| `add_trace_style_widgets` | `markersize`, `mean_linewidth`, `subject_*` | — |
| `add_dot_legend_widgets` | `show_legend`, `legend_x/y/dy/...` | `draw_dot_legend` |
| `add_score_inset_widgets` | `show_inset`, `inset_*` | `draw_score_inset` |

Keep bundles in `panels.py` only when they are figure-agnostic. A helper that knows about
external/internal/neural role colours belongs in the figure's own `_shared` module, not here.

**Test for whether something is a bundle:** if two panels would otherwise contain the same ~20
lines, it is. `figure2` had the score-inset block duplicated character-for-character across two
viewers; they had already drifted in three places.

---

## 5. `plot()` only draws

Selection and expensive computation go in `refresh_data(state)`, wired to the data-selection
widgets via `on_change` and called once at the end of `__init__`. `plot()` reads `self._scores`
and puts marks on axes. Syd re-runs `plot()` on *every* widget change, including pure style
knobs — anything expensive in there is paid on every slider drag.

Sorting and display transforms that depend only on the data selection get memoised on the viewer.
**Key those caches on the selection that identifies the data, never on `id(array)`** — `id()` is
reused after GC, so a freed array's id can collide and return a stale result. The data key
already identifies which array is in play.

---

## 6. Figure lifecycle: `FigureViewer`

Subclass `panels.FigureViewer` and build via `self.new_figure(...)` / `self.new_subplots(...)`.
It closes only the figure *this viewer* made last. `plt.close("all")` in a `plot()` method also
discards figures the notebook made elsewhere; omitting it leaks one figure per widget change.

---

## 7. One module per panel

A 4000-line figure module is a packaging problem, not a complexity problem — the `# ====` banners
are already the seams. `figure2/` splits as:

```
figure2/
  __init__.py       # re-exports every public viewer
  model_style.py    # colours / linestyles shared by every panel
  _predictions.py   # cross-panel data: caches, transforms
  _scores.py        # cross-panel data + the trace shapes built on it
  performance.py    # one module per panel
  familiarity.py
  ...
```

`__init__.py` re-exports, so `from ...figure_scripts.figure2 import X` keeps working. Do this
*last*, after the shared layer exists — otherwise you are sorting duplication into folders.

---

## Critical implementation details worth sharing

These are load-bearing and easy to get wrong:

- **`format_spines` ordering.** It positions the offset spines from the axis's *current* limits,
  so `set_ylim` must come first; and it calls `tick_params` itself, so font sizes must be applied
  *after*. Use `panels.style_model_axis`, which packages
  `format_spines` → `set_xticks` → `tick_params` with that ordering (and includes
  `which="both"`, which log axes need for their minor labels).

- **`sharey` and offset spines.** With shared axes, *both* panels must be populated before either
  is formatted — formatting `ax[0]` while `ax[1]` is empty converts the fractional spine offset
  against stale limits. Draw everything, then format in a second loop.

- **Syd validates selections.** `add_selection` rejects a value outside its options, so a stale
  default fails loudly instead of quietly plotting the wrong variation. Don't add your own
  membership checks for selectable parameters; do check things Syd can't see (a fixed model list
  against `param_axes`).

- **Docstrings.** Since `__init__` is now the public API, the parameter documentation lives on the
  viewer class, not on a wrapper.

---

## Migration checklist

For each `figure*.py`:

1. Delete the caller functions; move their defaults into the viewer's `__init__` signature and
   their docstrings onto the class. Drop the seed-then-`refresh_data` workaround.
2. Remove `plt.rcParams` mutation; add a `fontsize` widget wherever text was relying on it.
3. Replace hard-coded option lists with `add_data_selection_widgets`; wire `on_change` from the
   returned names; add `require=` for values passed outside `results.sel`.
4. Replace duplicated widget/draw blocks with `panels.py` bundles; add a bundle when two panels
   share one.
5. `Viewer` → `FigureViewer`; `plt.close("all")` → `new_figure`/`new_subplots`.
6. Move computation out of `plot()`; drop `id()` from any cache key.
7. Split into a package, re-exporting from `__init__.py`.

Steps 1–6 are behaviour-preserving. Verify by rendering every panel before and after and diffing
the images — and call `plot()` twice, asserting an unrelated figure survives.
