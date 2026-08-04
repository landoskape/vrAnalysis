"""One mouse's experiment as a session timeline: when each environment entered the protocol."""

from collections import defaultdict

import numpy as np
from matplotlib.patches import Rectangle

from vrAnalysis.sessions import B2Session

from dimensionality_manuscript.env_order import _session_sort_key
from dimensionality_manuscript.figure_scripts.panels import FigureViewer

from ._shared import env_slot_color, ordinal


def _boolean_runs(present: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` runs of True in a 1-D boolean array."""
    padded = np.concatenate([[False], present, [False]]).astype(int)
    change = np.diff(padded)
    return list(zip(np.where(change == 1)[0], np.where(change == -1)[0]))


class ExperimentTimeline(FigureViewer):
    """One mouse's experiment as a session timeline: when each environment entered the protocol.

    A phase bar on top splits the timeline into training (not in the dataset -- its length is the
    ``num_training`` knob) and imaging (one slot per session in the data). Below it, one bar per
    environment spanning the sessions in which that environment was actually run, ordered and
    colored by experience order -- the same palette as :class:`~.schematic.VREnvironmentSchematic`
    and the ``by_env`` panels of figure 3. An arrow at the bottom carries the session-axis label.

    Environment identities do not carry meaning across cohorts (the CR mice run environments 1 and
    2, the ATL mice 1, 3 and 4), which is why the labels use experience order (``"1st Env"``,
    ``"2nd Env"``, ...) unless ``label_by_env_id`` is set.

    Everything except the training block comes from the data: the number of imaging sessions and
    which environments each session ran. Training sessions predate the imaging dataset, so their
    count is a knob.

    The layout is fully relative: the vertical stack is built in arbitrary units and normalized to
    fill the figure height, and the horizontal axis is the session count normalized to [0, 1]. So
    the panel keeps its proportions at any ``figsize``, and every gap knob is a fraction of the
    whole rather than an absolute size. Font sizes are the exception -- they are in points, so
    scaling the figure up makes the text read relatively smaller.

    Parameters
    ----------
    sessions : list of B2Session
        The full session list; grouped by mouse and sorted chronologically internally.
    mouse : str or None
        Mouse to draw. None takes the first alphabetically.
    num_training : int
        Number of training sessions to draw ahead of the imaging sessions. These are not in the
        dataset, so this only sets how much of the timeline the training block occupies.
    merge_gaps : bool
        Draw each environment's bar from its first to its last session, ignoring sessions in
        between where it was not run. False draws one rectangle per contiguous run instead, which
        exposes the dropouts (e.g. CR_Hippocannula6 skips environment 1 mid-way).
    train_in_first_env : bool
        Extend the first environment's bar back through the training block -- the mouse is trained
        in the environment it later sees first.
    label_by_env_id : bool
        Label bars with the raw environment index instead of ordinal experience labels.
    figsize : tuple[float, float]
        Figure size in inches.
    phase_label_height, phase_bar_height, phase_gap : float
        Heights of the phase-label row, the phase bar, and the gap below them.
    env_bar_height, env_bar_gap : float
        Height of one environment bar and the gap between consecutive bars.
    axis_gap, axis_label_height : float
        Gap above the session-axis arrow, and the height of the band holding the arrow and its
        label. All of the above are in arbitrary units, normalized so the stack fills the figure --
        only their ratios matter.
    margin_x, margin_y : float
        Padding around the layout as a fraction of the panel width/height.
    phase_split_gap : float
        Gap between the training and imaging segments of the phase bar, in session units.
    env_label_pad : float
        Inset of the environment label from the right edge, in session units.
    fontsize : float
        Base font size in points.
    phase_label_scale, env_label_scale, axis_label_scale : float
        Multipliers on ``fontsize`` for the three text groups.
    training_color, imaging_color : str
        Colors of the two phase-bar segments. Grey by default so they do not compete with the
        environment palette.
    env_label_color : str
        Color of the labels drawn inside the environment bars.
    show_phase_bar, show_phase_labels, show_env_labels : bool
        Toggles for the phase bar, its "training"/"imaging" labels, and the in-bar env labels.
    show_training_divider : bool
        Draw a dotted line at the training/imaging boundary across the environment bars.
    show_session_ticks : bool
        Mark each imaging session with a tick above the axis arrow.
    axis_arrowstyle : str
        Matplotlib arrowstyle for the session axis; ``"<|-|>"`` heads both ends.
    """

    def __init__(
        self,
        sessions: list[B2Session],
        *,
        mouse: str | None = None,
        num_training: int = 6,
        merge_gaps: bool = True,
        train_in_first_env: bool = True,
        label_by_env_id: bool = False,
        figsize: tuple[float, float] = (3.6, 2.0),
        phase_label_height: float = 0.55,
        phase_bar_height: float = 0.3,
        phase_gap: float = 0.22,
        env_bar_height: float = 1.0,
        env_bar_gap: float = 0.06,
        axis_gap: float = 0.3,
        axis_label_height: float = 0.6,
        margin_x: float = 0.02,
        margin_y: float = 0.02,
        phase_split_gap: float = 0.4,
        env_label_pad: float = 0.25,
        fontsize: float = 9.0,
        phase_label_scale: float = 1.0,
        env_label_scale: float = 1.0,
        axis_label_scale: float = 1.0,
        training_color: str = "0.65",
        imaging_color: str = "0.25",
        env_label_color: str = "w",
        show_phase_bar: bool = True,
        show_phase_labels: bool = True,
        show_env_labels: bool = True,
        show_training_divider: bool = False,
        show_session_ticks: bool = False,
        axis_arrowstyle: str = "-|>",
    ):
        by_mouse: dict[str, list[B2Session]] = defaultdict(list)
        for session in sessions:
            by_mouse[session.mouse_name].append(session)
        # Chronological within a mouse; the incoming list order is not trusted.
        self.sessions_by_mouse = {name: sorted(ms, key=_session_sort_key) for name, ms in by_mouse.items()}
        self.mice = sorted(self.sessions_by_mouse)

        # --- data ---
        self.add_selection("mouse", value=mouse if mouse is not None else self.mice[0], options=self.mice)
        self.add_integer("num_training", value=num_training, min=0, max=60)
        self.add_boolean("merge_gaps", value=merge_gaps)
        self.add_boolean("train_in_first_env", value=train_in_first_env)
        self.add_boolean("label_by_env_id", value=label_by_env_id)

        # --- figure size in inches; everything else below is relative to it ---
        self.add_float("fig_width", value=figsize[0], min=0.5, max=20.0, step=0.01)
        self.add_float("fig_height", value=figsize[1], min=0.5, max=20.0, step=0.01)

        # --- vertical layout, in arbitrary units normalized to fill the figure height ---
        self.add_float("phase_label_height", value=phase_label_height, min=0.0, max=3.0)
        self.add_float("phase_bar_height", value=phase_bar_height, min=0.02, max=3.0)
        self.add_float("phase_gap", value=phase_gap, min=0.0, max=3.0)
        self.add_float("env_bar_height", value=env_bar_height, min=0.05, max=3.0)
        self.add_float("env_bar_gap", value=env_bar_gap, min=0.0, max=2.0)
        self.add_float("axis_gap", value=axis_gap, min=0.0, max=3.0)
        self.add_float("axis_label_height", value=axis_label_height, min=0.05, max=3.0)
        self.add_float("margin_x", value=margin_x, min=0.0, max=0.5)
        self.add_float("margin_y", value=margin_y, min=0.0, max=0.5)

        # --- horizontal layout, in session units (1.0 = the width of one session) ---
        self.add_float("phase_split_gap", value=phase_split_gap, min=0.0, max=5.0)
        self.add_float("env_label_pad", value=env_label_pad, min=0.0, max=5.0)

        # --- text ---
        self.add_float("fontsize", value=fontsize, min=3.0, max=30.0)
        self.add_float("phase_label_scale", value=phase_label_scale, min=0.2, max=3.0)
        self.add_float("env_label_scale", value=env_label_scale, min=0.2, max=3.0)
        self.add_float("axis_label_scale", value=axis_label_scale, min=0.2, max=3.0)

        # --- style ---
        self.add_text("training_color", value=training_color)
        self.add_text("imaging_color", value=imaging_color)
        self.add_text("env_label_color", value=env_label_color)
        self.add_boolean("show_phase_bar", value=show_phase_bar)
        self.add_boolean("show_phase_labels", value=show_phase_labels)
        self.add_boolean("show_env_labels", value=show_env_labels)
        self.add_boolean("show_training_divider", value=show_training_divider)
        self.add_boolean("show_session_ticks", value=show_session_ticks)
        self.add_text("axis_arrowstyle", value=axis_arrowstyle)

        self.on_change("mouse", self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Session count, environment experience order, and per-session presence for one mouse."""
        mouse_sessions = self.sessions_by_mouse[state["mouse"]]
        self.num_imaging = len(mouse_sessions)

        # Experience order: first appearance walking the sessions chronologically. Same rule as
        # ``env_order.build_env_order``, but kept per-session here because the bars need to know
        # *which* sessions used each environment, not just the order.
        self.env_order: list[int] = []
        for session in mouse_sessions:
            for env in session.environments:
                env = int(env)
                if env < 0:  # negative environmentIndex is the invalid/unlabeled sentinel
                    continue
                if env not in self.env_order:
                    self.env_order.append(env)

        self.env_present = np.zeros((len(self.env_order), self.num_imaging), dtype=bool)
        for isession, session in enumerate(mouse_sessions):
            for env in session.environments:
                env = int(env)
                if env >= 0:
                    self.env_present[self.env_order.index(env), isession] = True

    def _band_edges(self, state, num_envs: int) -> dict[str, tuple[float, float]]:
        """Top/bottom of every horizontal band, normalized so the stack fills the figure height."""
        bands: list[tuple[str, float]] = []
        if state["show_phase_labels"]:
            bands.append(("phase_label", state["phase_label_height"]))
        if state["show_phase_bar"]:
            bands.append(("phase_bar", state["phase_bar_height"]))
        if state["show_phase_bar"] or state["show_phase_labels"]:
            bands.append(("phase_gap", state["phase_gap"]))
        for slot in range(num_envs):
            bands.append((f"env{slot}", state["env_bar_height"]))
            if slot < num_envs - 1:
                bands.append((f"env_gap{slot}", state["env_bar_gap"]))
        bands.append(("axis_gap", state["axis_gap"]))
        bands.append(("axis_label", state["axis_label_height"]))

        total_units = sum(height for _, height in bands)
        edges: dict[str, tuple[float, float]] = {}
        cumulative = 0.0
        for name, height in bands:
            edges[name] = (1.0 - cumulative / total_units, 1.0 - (cumulative + height) / total_units)
            cumulative += height
        return edges

    def plot(self, state):
        fontsize = state["fontsize"]
        num_training = state["num_training"]
        num_envs = len(self.env_order)
        num_sessions = num_training + self.num_imaging
        edges = self._band_edges(state, num_envs)

        def x(session_units: float) -> float:
            """Session units -> [0, 1] across the panel width."""
            return session_units / num_sessions

        fig = self.new_figure(figsize=(state["fig_width"], state["fig_height"]))
        # One bare axes filling the figure: data coordinates are the relative layout itself.
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_xlim(-state["margin_x"], 1.0 + state["margin_x"])
        ax.set_ylim(edges["axis_label"][1] - state["margin_y"], 1.0 + state["margin_y"])
        ax.set_axis_off()

        def bar(left: float, right: float, band: tuple[float, float], color) -> None:
            top, bottom = band
            ax.add_patch(Rectangle((x(left), bottom), x(right) - x(left), top - bottom, facecolor=color, edgecolor="none", zorder=3))

        # ------------------------------------------------------------------- phase bar --
        # Training and imaging segments, separated by a gap so the split reads without a border.
        half_gap = state["phase_split_gap"] / 2
        if num_training > 0:
            segments = [
                ("training", 0.0, num_training - half_gap, state["training_color"]),
                ("imaging", num_training + half_gap, num_sessions, state["imaging_color"]),
            ]
        else:
            segments = [("imaging", 0.0, num_sessions, state["imaging_color"])]

        if state["show_phase_bar"]:
            for _, start, stop, color in segments:
                bar(start, stop, edges["phase_bar"], color)

        if state["show_phase_labels"]:
            top, bottom = edges["phase_label"]
            for label, start, stop, _ in segments:
                ax.text(
                    x((start + stop) / 2),
                    (top + bottom) / 2,
                    label,
                    ha="center",
                    va="center",
                    fontsize=fontsize * state["phase_label_scale"],
                    zorder=4,
                )

        # ------------------------------------------------------------------- env bars --
        for slot, env in enumerate(self.env_order):
            top, bottom = edges[f"env{slot}"]
            present = self.env_present[slot]
            if state["merge_gaps"]:
                # First appearance -> last appearance, ignoring any dropout in between; the
                # un-merged form draws one rectangle per contiguous run instead.
                runs = [(int(present.argmax()), int(self.num_imaging - present[::-1].argmax()))]
            else:
                runs = _boolean_runs(present)

            for start, stop in runs:
                left = num_training + start
                # The mouse is trained in the environment it sees first, so slot 0's bar covers
                # the training block too.
                if slot == 0 and start == 0 and state["train_in_first_env"]:
                    left = 0.0
                bar(left, num_training + stop, edges[f"env{slot}"], env_slot_color(slot))

            if state["show_env_labels"]:
                label = f"Env {env}" if state["label_by_env_id"] else f"{ordinal(slot + 1)} Env"
                ax.text(
                    x(num_sessions - state["env_label_pad"]),
                    (top + bottom) / 2,
                    label,
                    ha="right",
                    va="center",
                    color=state["env_label_color"],
                    fontsize=fontsize * state["env_label_scale"],
                    zorder=5,
                )

        if state["show_training_divider"] and num_training > 0:
            ax.plot(
                [x(num_training)] * 2,
                [edges["env0"][0], edges[f"env{num_envs - 1}"][1]],
                color="w",
                linestyle=":",
                linewidth=1.0,
                zorder=6,
            )

        # ----------------------------------------------------------------- session axis --
        top, bottom = edges["axis_label"]
        ax.annotate(
            "",
            xy=(1.0, top),
            xytext=(0.0, top),
            arrowprops=dict(
                arrowstyle=state["axis_arrowstyle"],
                color="k",
                linewidth=1.0,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=fontsize * 1.2,
            ),
            zorder=4,
        )

        if state["show_session_ticks"]:
            # One tick per imaging session, at the center of its slot.
            tick_height = (top - bottom) * 0.25
            for isession in range(self.num_imaging):
                xt = x(num_training + isession + 0.5)
                ax.plot([xt, xt], [top, top + tick_height], color="k", linewidth=0.8, zorder=4)

        ax.text(
            0.5,
            (top + bottom) / 2,
            f"{self.num_imaging} imaging sessions",
            ha="center",
            va="center",
            fontsize=fontsize * state["axis_label_scale"],
            zorder=4,
        )
        return fig
