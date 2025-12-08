"""Panel functions for the six CA1 prediction models."""

import numpy as np

from primitives import (
    Track,
    PositionDot,
    PFsArray,
    NeuronBlock,
    Arrow,
    Thermometer,
    HighDimBarplot,
    LatentArray,
)
from layout import PanelContext
from config import LAYOUT, SIZES, N_PFS, PALETTE


def panel_external_pf(ax, ctx=None, opt=None):
    """Panel 1: External PF - Purely spatial; measured position only.

    Equation: ŷ(t) = f(x(t))
    """
    ctx = ctx or PanelContext(ax)
    ctx.setup_panel()

    # Track with measured position
    track_y = LAYOUT["track_y"]
    Track(ax, y=track_y)
    pos_x = LAYOUT["pos_x_external"]
    gold_point_y = track_y

    # Gold point (position dot) in the middle
    PositionDot(ax, x=pos_x, y=gold_point_y, color="external")

    # x(t) label below the gold point (closer vertically, no arrow)
    x_t_y = gold_point_y - LAYOUT["x_label_offset_below"]
    ctx.text_center(pos_x, x_t_y, r"$x(t)$", size="small")

    # Array of place fields above track
    pf_y_base = track_y + LAYOUT["pf_offset_above_track"]
    pf_width = LAYOUT["pf_width_1d"]
    pf_height = SIZES["pf_height"] * LAYOUT["target_pf_scale"]
    PFsArray(ax, n_pfs=N_PFS, x_pos=pos_x, width=pf_width, height=pf_height, y_base=pf_y_base, color="target", lw=SIZES["lw"] / 2.0)

    # y hat (t) label above the place fields
    yhat_y = pf_y_base + pf_height + LAYOUT["yhat_offset_above_pf"]
    ctx.text_center(pos_x, yhat_y, r"$\hat y(t)$", size="small")

    # Vertical arrow from PFs array to y hat (t)
    Arrow(ax, pos_x, pf_y_base + pf_height, pos_x, yhat_y - LAYOUT["arrow_offset_from_pf_top"], style="solid")

    # Labels
    ctx.title("External PF (1-D)")


def panel_internal_pf(ax, ctx=None, opt=None):
    """Panel 2: Internal PF (split-half) - Decode position from peers; PF lookup.

    Equation: x̂ = D y_src; ŷ = f(x̂)
    """
    ctx = ctx or PanelContext(ax)
    ctx.setup_panel()

    # Fixed track position (same as external panels)
    track_y = LAYOUT["track_y"]
    pos_x = LAYOUT["pos_x_internal"]

    # Calculate source PF position to fit below track
    source_pf_height = SIZES["pf_height"] * LAYOUT["source_pf_scale"]
    source_pf_y_base = track_y - LAYOUT["source_pf_spacing_above"] - source_pf_height

    pf_width = LAYOUT["pf_width_1d"]
    PFsArray(ax, n_pfs=N_PFS, x_pos=pos_x, width=pf_width, height=source_pf_height, y_base=source_pf_y_base, color="source", lw=SIZES["lw"] / 2.0)

    # Track with decoded position (at fixed Y position)
    Track(ax, y=track_y)
    gold_point_y = track_y

    # Gold point (decoded position x hat (t)) on the track
    PositionDot(ax, x=pos_x, y=gold_point_y, color="external")

    # x hat (t) label below the track (closer vertically, no arrow to ball)
    xhat_y = gold_point_y - LAYOUT["x_label_offset_below"]
    ctx.text_center(pos_x, xhat_y, r"$\hat x(t)$", size="small")

    # Arrow from source PFs to xhat (t) label (labeling connection)
    Arrow(ax, pos_x, source_pf_y_base + source_pf_height, pos_x, xhat_y - LAYOUT["arrow_offset_from_label"], style="solid")

    # Target place fields above track (same position as external panels)
    target_pf_y_base = track_y + LAYOUT["pf_offset_above_track"]
    target_pf_height = SIZES["pf_height"] * LAYOUT["target_pf_scale"]
    PFsArray(ax, n_pfs=N_PFS, x_pos=pos_x, width=pf_width, height=target_pf_height, y_base=target_pf_y_base, color="target", lw=SIZES["lw"] / 2.0)

    # y hat (t) label above the target PFs
    yhat_y = target_pf_y_base + target_pf_height + LAYOUT["yhat_offset_above_pf"]
    ctx.text_center(pos_x, yhat_y, r"$\hat y(t)$", size="small")

    # Arrow from target PFs to y hat (t) label (labeling connection)
    Arrow(ax, pos_x, target_pf_y_base + target_pf_height, pos_x, yhat_y - LAYOUT["arrow_offset_from_pf_top"], style="solid")

    # Labels
    ctx.title("Internal PF (1-D)")


def panel_external_pf_gain(ax, ctx=None, opt=None):
    """Panel 3: External PF + Gain - Global modulation on spatial prediction.

    Equation: ŷ(t) = g f(x(t))
    """
    ctx = ctx or PanelContext(ax)
    ctx.setup_panel()

    # Track with measured position
    track_y = LAYOUT["track_y"]
    Track(ax, y=track_y)
    pos_x = LAYOUT["pos_x_external"]
    gold_point_y = track_y
    track_x0 = LAYOUT["track_x0"]

    # Calculate source PF position to fit below track
    source_pf_height = SIZES["pf_height"] * LAYOUT["source_pf_scale"]
    source_pf_y_base = track_y - LAYOUT["source_pf_spacing_above"] - source_pf_height

    # Gold point (position dot) in the middle
    PositionDot(ax, x=pos_x, y=gold_point_y, color="external")

    # x(t) label below the gold point (closer vertically, no arrow)
    x_t_y = gold_point_y - LAYOUT["x_label_offset_below"]
    ctx.text_center(pos_x, x_t_y, r"$x(t)$", size="small")

    # Array of place fields above track (normal amplitude with fill)
    pf_y_base = track_y + LAYOUT["pf_offset_above_track"]
    pf_width = LAYOUT["pf_width_1d"]
    pf_height = SIZES["pf_height"] * LAYOUT["target_pf_scale_gain"]
    PFsArray(ax, n_pfs=N_PFS, x_pos=pos_x, width=pf_width, height=pf_height, y_base=pf_y_base, color="target", lw=SIZES["lw"] / 2.0)

    # Thermometer to the left of place field array (offset to left of track)
    thermometer_x = track_x0 - LAYOUT["thermometer_x_offset"]
    thermometer_width = LAYOUT["thermometer_width"]
    ball_radius = thermometer_width
    thermometer_y_base = pf_y_base - ball_radius  # Adjust for ball at bottom
    thermometer_total_height = pf_height * LAYOUT["thermometer_height_multiplier"] + ball_radius
    thermometer_fill_height = pf_height  # Fill height matches place field height
    Thermometer(
        ax,
        x=thermometer_x,
        y_base=thermometer_y_base,
        total_height=thermometer_total_height,
        fill_height=thermometer_fill_height,
        width=thermometer_width,
    )

    # y hat (t) label above the place fields (with gain)
    yhat_y = pf_y_base + pf_height + LAYOUT["yhat_offset_above_pf_gain"]
    ctx.text_center(pos_x, yhat_y, r"$g \hat y(t)$", size="small")

    # Vertical arrow from PFs array to y hat (t)
    Arrow(ax, pos_x, pf_y_base + pf_height, pos_x, yhat_y - LAYOUT["arrow_offset_from_pf_top"], style="solid")

    # Arrow from thermometer fill height to y hat (t)
    arrow_thermometer_x_offset = LAYOUT["thermometer_arrow_offset"]
    thermometer_fill_top_y = pf_y_base + thermometer_fill_height
    Arrow(
        ax,
        thermometer_x + arrow_thermometer_x_offset,
        thermometer_fill_top_y,
        pos_x - 2 * arrow_thermometer_x_offset,
        yhat_y - LAYOUT["arrow_offset_from_pf_top"],
        style="solid",
    )

    # Neuron block below thermometer (source neurons controlling gain)
    neuron_block_size = 0.18
    neuron_block_y = source_pf_y_base + source_pf_height - neuron_block_size / 2  # Space between thermometer and neuron block
    neuron_block_x = track_x0 - neuron_block_size / 2  # Center on track
    neuron_block_center_x = neuron_block_x + neuron_block_size / 2
    neuron_block_center_y = neuron_block_y + neuron_block_size / 2
    NeuronBlock(
        ax,
        x=neuron_block_x,
        y=neuron_block_y,
        w=neuron_block_size,
        h=neuron_block_size,
        color="source",
    )

    # "g" label below thermometer (similar to x(t) below track)
    g_label_y = thermometer_y_base - LAYOUT["g_label_offset_below_thermometer"]
    ctx.text_center(thermometer_x, g_label_y, r"$g$", size="small")

    # Arrow from neuron block to "g" label
    Arrow(
        ax,
        neuron_block_center_x,  # Center of neuron block (x)
        neuron_block_center_y + neuron_block_size / 2,  # Center of neuron block (y)
        thermometer_x,  # Center of thermometer (x)
        g_label_y - 0.01,  # "g" label position
        style="solid",
    )

    # Labels
    ctx.title("External PF + Gain")


def panel_internal_pf_gain(ax, ctx=None, opt=None):
    """Panel 4: Internal PF + Gain - Combine internal decoding with gain modulation.

    Equation: x̂ = D y_src; ŷ = g f(x̂)
    """
    ctx = ctx or PanelContext(ax)
    ctx.setup_panel()

    # Fixed track position (same as external panels)
    track_y = LAYOUT["track_y"]
    pos_x = LAYOUT["pos_x_internal"]
    track_x0 = LAYOUT["track_x0"]

    # Calculate source PF position to fit below track
    source_pf_height = SIZES["pf_height"] * LAYOUT["source_pf_scale"]
    source_pf_y_base = track_y - LAYOUT["source_pf_spacing_above"] - source_pf_height

    pf_width = LAYOUT["pf_width_1d"]
    PFsArray(ax, n_pfs=N_PFS, x_pos=pos_x, width=pf_width, height=source_pf_height, y_base=source_pf_y_base, color="source", lw=SIZES["lw"] / 2.0)

    # Track with decoded position (at fixed Y position)
    Track(ax, y=track_y)
    gold_point_y = track_y

    # Gold point (decoded position x hat (t)) on the track
    PositionDot(ax, x=pos_x, y=gold_point_y, color="external")

    # x hat (t) label below the track (closer vertically, no arrow to ball)
    xhat_y = gold_point_y - LAYOUT["x_label_offset_below"]
    ctx.text_center(pos_x, xhat_y, r"$\hat x(t)$", size="small")

    # Arrow from source PFs to xhat (t) label (labeling connection)
    Arrow(ax, pos_x, source_pf_y_base + source_pf_height, pos_x, xhat_y - LAYOUT["arrow_offset_from_label"], style="solid")

    # Target place fields above track (same position as external panels)
    target_pf_y_base = track_y + LAYOUT["pf_offset_above_track"]
    target_pf_height = SIZES["pf_height"] * LAYOUT["target_pf_scale_gain"]
    PFsArray(ax, n_pfs=N_PFS, x_pos=pos_x, width=pf_width, height=target_pf_height, y_base=target_pf_y_base, color="target", lw=SIZES["lw"] / 2.0)

    # Thermometer to the left of place field array (offset to left of track)
    thermometer_x = track_x0 - LAYOUT["thermometer_x_offset"]
    thermometer_width = LAYOUT["thermometer_width"]
    ball_radius = thermometer_width
    thermometer_y_base = target_pf_y_base - ball_radius  # Adjust for ball at bottom
    thermometer_total_height = target_pf_height * LAYOUT["thermometer_height_multiplier"] + ball_radius
    thermometer_fill_height = target_pf_height  # Fill height matches place field height
    Thermometer(
        ax,
        x=thermometer_x,
        y_base=thermometer_y_base,
        total_height=thermometer_total_height,
        fill_height=thermometer_fill_height,
        width=thermometer_width,
    )

    # y hat (t) label above the target PFs (with gain)
    yhat_y = target_pf_y_base + target_pf_height + LAYOUT["yhat_offset_above_pf_gain"]
    ctx.text_center(pos_x, yhat_y, r"$g \hat y(t)$", size="small")

    # Arrow from target PFs to y hat (t) label (labeling connection)
    Arrow(ax, pos_x, target_pf_y_base + target_pf_height, pos_x, yhat_y - LAYOUT["arrow_offset_from_pf_top"], style="solid")

    # Arrow from thermometer fill height to y hat (t)
    arrow_thermometer_x_offset = LAYOUT["thermometer_arrow_offset"]
    thermometer_fill_top_y = target_pf_y_base + thermometer_fill_height
    Arrow(
        ax,
        thermometer_x + arrow_thermometer_x_offset,
        thermometer_fill_top_y,
        pos_x - 2 * arrow_thermometer_x_offset,
        yhat_y - LAYOUT["arrow_offset_from_pf_top"],
        style="solid",
    )

    # Neuron block below thermometer (source neurons controlling gain)
    neuron_block_size = 0.18
    neuron_block_y = source_pf_y_base + source_pf_height - neuron_block_size / 2  # Space between thermometer and neuron block
    neuron_block_x = track_x0 - neuron_block_size / 2  # Center on track
    neuron_block_center_x = neuron_block_x + neuron_block_size / 2
    neuron_block_center_y = neuron_block_y + neuron_block_size / 2
    NeuronBlock(
        ax,
        x=neuron_block_x,
        y=neuron_block_y,
        w=neuron_block_size,
        h=neuron_block_size,
        color="source",
    )

    # "g" label below thermometer (similar to x(t) below track)
    g_label_y = thermometer_y_base - LAYOUT["g_label_offset_below_thermometer"]
    ctx.text_center(thermometer_x, g_label_y, r"$g$", size="small")

    # Arrow from neuron block to "g" label
    Arrow(
        ax,
        neuron_block_center_x,  # Center of neuron block (x)
        neuron_block_center_y + neuron_block_size / 2,  # Center of neuron block (y)
        thermometer_x,  # Center of thermometer (x)
        g_label_y - 0.01,  # "g" label position
        style="solid",
    )

    # Labels
    ctx.title("Internal PF + Gain")


def panel_rbf(ax, ctx=None, opt=None):
    """Panel 5.1: High-D PF (RBF decode→encode) - High-dimensional basis functions.

    Equation: φ̂ = D y_src; ŷ = E φ̂
    """
    ctx = ctx or PanelContext(ax)
    ctx.setup_panel()

    # Track with position
    track_y = LAYOUT["track_y"]
    Track(ax, y=track_y)
    track_x0 = LAYOUT["track_x0"]
    track_x1 = LAYOUT["track_x1"]

    # True position with gold dot
    pos_x = 0.4
    gold_point_y = track_y
    x_t_y = gold_point_y - LAYOUT["x_label_offset_below"]
    PositionDot(ax, x=pos_x, y=gold_point_y, color="external")
    ctx.text_center(pos_x, x_t_y, r"$x(t)$", size="small")

    # Calculate bar heights representing probability distribution
    # Main peak at 0.375 (true position), smaller bump at 0.775
    n_bins = 12

    # Define bar heights
    bar_indices = np.arange(n_bins)
    main_peak_bar = bar_indices[4]
    main_peak_height = 0.125
    secondary_peak_bar = bar_indices[10]
    secondary_peak_height = 0.065
    base_peak_height = 0.02

    # Calculate bar heights as probability distribution
    bar_heights = [base_peak_height] * n_bins
    bar_heights[main_peak_bar] = main_peak_height
    bar_heights[main_peak_bar + 1] = secondary_peak_height
    bar_heights[main_peak_bar - 1] = secondary_peak_height
    bar_heights[secondary_peak_bar] = secondary_peak_height

    # Calculate bar center positions (needed for arrows)
    total_width = track_x1 - track_x0
    bar_width = total_width / n_bins
    bar_centers = [track_x0 + (i + 0.5) * bar_width for i in range(n_bins)]

    # High-dimensional barplot emerging from track with probability distribution
    HighDimBarplot(
        ax,
        track_y=track_y,
        n_bins=n_bins,
        max_height=0.15,
        track_x0=track_x0,
        track_x1=track_x1,
        color="stroke",
        bar_heights=bar_heights,
    )

    # Source neurons below track (left side)
    source_neuron_x = 0.5 - LAYOUT["pf_width_highd"] / 2
    source_neuron_y = track_y - LAYOUT["source_pf_spacing_above"] - LAYOUT["pf_width_highd"]

    # Target neurons above track (right side)
    target_neuron_x = 0.5 - LAYOUT["pf_width_highd"] / 2
    target_neuron_y = track_y + LAYOUT["pf_offset_above_track"] + LAYOUT["pf_width_highd"]
    NeuronBlock(ax, x=target_neuron_x, y=target_neuron_y, w=LAYOUT["pf_width_highd"], h=LAYOUT["pf_width_highd"], n_neurons=4, color="target")

    # Draw lines from source neurons to bar centers at track (simple lines, no arrowheads)
    # Calculate actual top of outer circle (center_y + outer_radius)
    # Outer radius = 0.85 * min(w, h) / 2 = 0.85 * width / 2 (since w == h for square)
    # Center_y = y + h / 2
    # Top of circle = center_y + outer_radius = (y + h/2) + (0.85 * h / 2) = y + 0.925 * h
    neuron_block_size = LAYOUT["pf_width_highd"]
    source_neuron_x_center = source_neuron_x + neuron_block_size / 2
    source_neuron_center_y = source_neuron_y + neuron_block_size / 2
    outer_radius = 0.85 * neuron_block_size / 2
    source_neuron_y_top = source_neuron_center_y + outer_radius  # Top of outer circle

    # Number of segments per line for gradient effect
    color = PALETTE.get("source", "gray")
    n_segments = 100
    alpha_exponent = 4.0  # Exponent for alpha decay (higher = faster drop-off)

    # Draw lines from target neurons to bar centers at track with alpha gradient (opposite direction)
    # Calculate actual bottom of outer circle for target neurons
    color = PALETTE.get("target", "gray")
    target_neuron_x_center = target_neuron_x + neuron_block_size / 2
    target_neuron_center_y = target_neuron_y + neuron_block_size / 2
    target_neuron_y_bottom = target_neuron_center_y - outer_radius  # Bottom of outer circle

    for bar_center_x in bar_centers:
        # Create gradient by dividing line into segments with decreasing alpha
        x_start, y_start = target_neuron_x_center, target_neuron_y_bottom
        x_end, y_end = bar_center_x, track_y

        # Generate points along the line
        x_points = np.linspace(x_start, x_end, n_segments + 1)
        y_points = np.linspace(y_start, y_end, n_segments + 1)

        # Draw segments with decreasing alpha (1.0 at neurons, 0.0 at track)
        for i in range(n_segments):
            # Alpha decreases from 1.0 (at neuron) to 0.0 (at track)
            alpha = (1.0 - (i / (n_segments - 1)) if n_segments > 1 else 1.0) ** alpha_exponent
            ax.plot([x_points[i], x_points[i + 1]], [y_points[i], y_points[i + 1]], color=color, linewidth=SIZES["lw"] * 0.3, alpha=alpha, zorder=0)

    # Labels
    ctx.title("High-D PF")


def panel_rbf_internal(ax, ctx=None, opt=None):
    """Panel 5: High-D internal PF (RBF decode→encode) - High-dimensional basis functions.

    Equation: φ̂ = D y_src; ŷ = E φ̂
    """
    ctx = ctx or PanelContext(ax)
    ctx.setup_panel()

    # Track with position
    track_y = LAYOUT["track_y"]
    Track(ax, y=track_y)
    track_x0 = LAYOUT["track_x0"]
    track_x1 = LAYOUT["track_x1"]

    # True position with gold dot
    pos_x = 0.4
    gold_point_y = track_y
    x_t_y = gold_point_y - LAYOUT["x_label_offset_below"]
    PositionDot(ax, x=pos_x, y=gold_point_y, color="external")
    ctx.text_center(pos_x, x_t_y, r"$x(t)$", size="small")

    # Calculate bar heights representing probability distribution
    # Main peak at 0.375 (true position), smaller bump at 0.775
    n_bins = 12

    # Define bar heights
    bar_indices = np.arange(n_bins)
    main_peak_bar = bar_indices[4]
    main_peak_height = 0.125
    secondary_peak_bar = bar_indices[10]
    secondary_peak_height = 0.065
    base_peak_height = 0.02

    # Calculate bar heights as probability distribution
    bar_heights = [base_peak_height] * n_bins
    bar_heights[main_peak_bar] = main_peak_height
    bar_heights[main_peak_bar + 1] = secondary_peak_height
    bar_heights[main_peak_bar - 1] = secondary_peak_height
    bar_heights[secondary_peak_bar] = secondary_peak_height

    # Calculate bar center positions (needed for arrows)
    total_width = track_x1 - track_x0
    bar_width = total_width / n_bins
    bar_centers = [track_x0 + (i + 0.5) * bar_width for i in range(n_bins)]

    # High-dimensional barplot emerging from track with probability distribution
    HighDimBarplot(
        ax,
        track_y=track_y,
        n_bins=n_bins,
        max_height=0.15,
        track_x0=track_x0,
        track_x1=track_x1,
        color="stroke",
        bar_heights=bar_heights,
    )

    # Source neurons below track (left side)
    source_neuron_x = 0.5 - LAYOUT["pf_width_highd"] / 2
    source_neuron_y = track_y - LAYOUT["source_pf_spacing_above"] - LAYOUT["pf_width_highd"]
    NeuronBlock(ax, x=source_neuron_x, y=source_neuron_y, w=LAYOUT["pf_width_highd"], h=LAYOUT["pf_width_highd"], n_neurons=4, color="source")

    # Target neurons above track (right side)
    target_neuron_x = 0.5 - LAYOUT["pf_width_highd"] / 2
    target_neuron_y = track_y + LAYOUT["pf_offset_above_track"] + LAYOUT["pf_width_highd"]
    NeuronBlock(ax, x=target_neuron_x, y=target_neuron_y, w=LAYOUT["pf_width_highd"], h=LAYOUT["pf_width_highd"], n_neurons=4, color="target")

    # Draw lines from source neurons to bar centers at track (simple lines, no arrowheads)
    # Calculate actual top of outer circle (center_y + outer_radius)
    # Outer radius = 0.85 * min(w, h) / 2 = 0.85 * width / 2 (since w == h for square)
    # Center_y = y + h / 2
    # Top of circle = center_y + outer_radius = (y + h/2) + (0.85 * h / 2) = y + 0.925 * h
    neuron_block_size = LAYOUT["pf_width_highd"]
    source_neuron_x_center = source_neuron_x + neuron_block_size / 2
    source_neuron_center_y = source_neuron_y + neuron_block_size / 2
    outer_radius = 0.85 * neuron_block_size / 2
    source_neuron_y_top = source_neuron_center_y + outer_radius  # Top of outer circle

    # Number of segments per line for gradient effect
    color = PALETTE.get("source", "gray")
    n_segments = 100
    alpha_exponent = 4.0  # Exponent for alpha decay (higher = faster drop-off)

    for bar_center_x in bar_centers:
        # Create gradient by dividing line into segments with decreasing alpha
        x_start, y_start = source_neuron_x_center, source_neuron_y_top
        x_end, y_end = bar_center_x, track_y

        # Generate points along the line
        x_points = np.linspace(x_start, x_end, n_segments + 1)
        y_points = np.linspace(y_start, y_end, n_segments + 1)

        # Draw segments with decreasing alpha (1.0 at start, 0.0 at end)
        for i in range(n_segments):
            # Alpha decreases from 1.0 (at neuron) to 0.0 (at track)
            alpha = (1.0 - (i / (n_segments - 1)) if n_segments > 1 else 1.0) ** alpha_exponent
            ax.plot([x_points[i], x_points[i + 1]], [y_points[i], y_points[i + 1]], color=color, linewidth=SIZES["lw"] * 0.3, alpha=alpha, zorder=0)

    # Draw lines from target neurons to bar centers at track with alpha gradient (opposite direction)
    # Calculate actual bottom of outer circle for target neurons
    color = PALETTE.get("target", "gray")
    target_neuron_x_center = target_neuron_x + neuron_block_size / 2
    target_neuron_center_y = target_neuron_y + neuron_block_size / 2
    target_neuron_y_bottom = target_neuron_center_y - outer_radius  # Bottom of outer circle

    for bar_center_x in bar_centers:
        # Create gradient by dividing line into segments with decreasing alpha
        x_start, y_start = target_neuron_x_center, target_neuron_y_bottom
        x_end, y_end = bar_center_x, track_y

        # Generate points along the line
        x_points = np.linspace(x_start, x_end, n_segments + 1)
        y_points = np.linspace(y_start, y_end, n_segments + 1)

        # Draw segments with decreasing alpha (1.0 at neurons, 0.0 at track)
        for i in range(n_segments):
            # Alpha decreases from 1.0 (at neuron) to 0.0 (at track)
            alpha = (1.0 - (i / (n_segments - 1)) if n_segments > 1 else 1.0) ** alpha_exponent
            ax.plot([x_points[i], x_points[i + 1]], [y_points[i], y_points[i + 1]], color=color, linewidth=SIZES["lw"] * 0.3, alpha=alpha, zorder=0)

    # Labels
    ctx.title("High-D Internal PF")


def panel_rrr(ax, ctx=None, opt=None):
    """Panel 6: RRR (non-spatial peer) - Peer prediction; no spatial variables.

    Equation: Ŷ = W_RRR Y_src, rank(W) = r
    """
    ctx = ctx or PanelContext(ax)
    ctx.setup_panel()

    # Use same track_y position as panel 5 for consistency
    track_y = LAYOUT["track_y"]
    track_x0 = LAYOUT["track_x0"]
    track_x1 = LAYOUT["track_x1"]

    # Latent array (replaces track/barplot)
    latent_y = track_y
    latent_w = track_x1 - track_x0
    # Calculate ball radius to match neurons
    neuron_block_size = LAYOUT["pf_width_highd"]
    neuron_outer_radius = 0.85 * neuron_block_size / 2
    neuron_radius = neuron_outer_radius * 0.25  # Same calculation as in NeuronBlock
    # Make height larger to accommodate balls with same radius as neurons
    latent_h = neuron_radius * 4  # Height to comfortably fit balls of neuron radius
    latent_x = track_x0
    latent_array_y = latent_y - latent_h / 2  # Center the rounded rect vertically at track_y

    # Draw latent array and get ball positions
    latent_ball_positions = LatentArray(
        ax, x=latent_x, y=latent_array_y, w=latent_w, h=latent_h, n_latents=12, color="gray", ball_radius=neuron_radius
    )

    # Source neurons below (same position as panel 5)
    source_neuron_x = 0.5 - LAYOUT["pf_width_highd"] / 2
    source_neuron_y = track_y - LAYOUT["source_pf_spacing_above"] - LAYOUT["pf_width_highd"]
    NeuronBlock(ax, x=source_neuron_x, y=source_neuron_y, w=LAYOUT["pf_width_highd"], h=LAYOUT["pf_width_highd"], n_neurons=4, color="source")

    # Target neurons above (same position as panel 5)
    target_neuron_x = 0.5 - LAYOUT["pf_width_highd"] / 2
    target_neuron_y = track_y + LAYOUT["pf_offset_above_track"] + LAYOUT["pf_width_highd"]
    NeuronBlock(ax, x=target_neuron_x, y=target_neuron_y, w=LAYOUT["pf_width_highd"], h=LAYOUT["pf_width_highd"], n_neurons=4, color="target")

    # Draw lines from source neurons to latent array with alpha gradient
    neuron_block_size = LAYOUT["pf_width_highd"]
    source_neuron_x_center = source_neuron_x + neuron_block_size / 2
    source_neuron_center_y = source_neuron_y + neuron_block_size / 2
    outer_radius = 0.85 * neuron_block_size / 2
    source_neuron_y_top = source_neuron_center_y + outer_radius

    color = PALETTE.get("source", "gray")
    n_segments = 100
    alpha_exponent = 4.0

    for latent_x_pos in latent_ball_positions:
        # Create gradient by dividing line into segments with decreasing alpha
        x_start, y_start = source_neuron_x_center, source_neuron_y_top
        x_end, y_end = latent_x_pos, latent_y

        # Generate points along the line
        x_points = np.linspace(x_start, x_end, n_segments + 1)
        y_points = np.linspace(y_start, y_end, n_segments + 1)

        # Draw segments with decreasing alpha (1.0 at start, 0.0 at end)
        for i in range(n_segments):
            alpha = (1.0 - (i / (n_segments - 1)) if n_segments > 1 else 1.0) ** alpha_exponent
            ax.plot([x_points[i], x_points[i + 1]], [y_points[i], y_points[i + 1]], color=color, linewidth=SIZES["lw"] * 0.3, alpha=alpha, zorder=0)

    # Draw lines from target neurons to latent array with alpha gradient
    color = PALETTE.get("target", "gray")
    target_neuron_x_center = target_neuron_x + neuron_block_size / 2
    target_neuron_center_y = target_neuron_y + neuron_block_size / 2
    target_neuron_y_bottom = target_neuron_center_y - outer_radius

    for latent_x_pos in latent_ball_positions:
        # Create gradient by dividing line into segments with decreasing alpha
        x_start, y_start = target_neuron_x_center, target_neuron_y_bottom
        x_end, y_end = latent_x_pos, latent_y

        # Generate points along the line
        x_points = np.linspace(x_start, x_end, n_segments + 1)
        y_points = np.linspace(y_start, y_end, n_segments + 1)

        # Draw segments with decreasing alpha (1.0 at neurons, 0.0 at latents)
        for i in range(n_segments):
            alpha = (1.0 - (i / (n_segments - 1)) if n_segments > 1 else 1.0) ** alpha_exponent
            ax.plot([x_points[i], x_points[i + 1]], [y_points[i], y_points[i + 1]], color=color, linewidth=SIZES["lw"] * 0.3, alpha=alpha, zorder=0)

    # Labels
    ctx.title("RRR (Non-spatial)")


# Panel mapping for easy access
PANEL_FUNCTIONS = {
    1: panel_external_pf,
    2: panel_internal_pf,
    3: panel_external_pf_gain,
    4: panel_internal_pf_gain,
    5: panel_rbf_internal,
    6: panel_rrr,
    7: panel_rbf,
}

PANEL_NAMES = {
    1: "External PF",
    2: "Internal PF",
    3: "External PF + Gain",
    4: "Internal PF + Gain",
    5: "High-D Internal PF",
    6: "RRR (Non-spatial)",
    7: "High-D PF",
}
