"""Primitive drawing components for neuro-model schematics.

Each primitive draws relative to a panel-local coordinate system ([0,1] × [0,1]).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Arc
from config import PALETTE, SIZES, FONTS, ARROW_STYLES


def Track(ax, x0=0.1, x1=0.9, y=0.25, ticks=5, color="stroke", zorder=100):
    """Draw a horizontal track with position ticks."""
    color = PALETTE.get(color, color)

    # Main track line
    ax.plot([x0, x1], [y, y], color=color, linewidth=SIZES["lw"], zorder=zorder)

    # Position ticks
    if ticks > 0:
        tick_positions = np.linspace(x0, x1, ticks)
        for pos in tick_positions:
            ax.plot([pos, pos], [y - SIZES["tick"] / 2, y + SIZES["tick"] / 2], color=color, linewidth=SIZES["lw"], zorder=zorder)


def PositionDot(ax, x, y, color, dashed=False, alpha=1.0):
    """Draw a position dot (measured or decoded)."""
    color = PALETTE.get(color, color)

    if dashed:
        # Dashed outline for decoded position
        circle = Circle((x, y), radius=0.02, facecolor=color, edgecolor="k", alpha=alpha, linestyle="--", linewidth=SIZES["lw"], zorder=101)
    else:
        # Solid for measured position
        circle = Circle((x, y), radius=0.02, facecolor=color, edgecolor="k", alpha=alpha, zorder=101)

    ax.add_patch(circle)


def PF(ax, center, width, height, y_base, color, lw=None):
    """Draw a place field as a bell curve."""
    color = PALETTE.get(color, color)
    lw = lw or SIZES["lw"]

    # Generate bell curve
    x = np.linspace(center - width, center + width, 50)
    y = y_base + height * np.exp(-0.5 * ((x - center) / (width / 3)) ** 2)

    ax.plot(x, y, color=color, linewidth=lw)
    ax.fill_between(x, y_base, y, color=color, alpha=0.3)


def PFsArray(ax, n_pfs, x_pos, width, height, y_base, track_x0=0.1, track_x1=0.9, color="target", lw=None):
    """Draw an array of place fields spread across the track with activity dots.

    Args:
        ax: Matplotlib axis
        n_pfs: Number of place fields to draw
        x_pos: Current position on track (x coordinate) - used to calculate activity dots
        track_x0: Left boundary of track
        track_x1: Right boundary of track
        width: Width of each place field
        height: Height of each place field
        y_base: Base Y position for place fields
        color: Color of place fields
        lw: Line width
    """
    color = PALETTE.get(color, color)
    lw = lw or SIZES["lw"]

    # Calculate PF centers spread across the track
    pf_centers = np.linspace(track_x0 + width, track_x1 - width, n_pfs)

    # Draw each PF
    for center in pf_centers:
        # Generate bell curve
        x = np.linspace(center - width, center + width, 50)
        y = y_base + height * np.exp(-0.5 * ((x - center) / (width / 3)) ** 2)

        ax.plot(x, y, color=color, linewidth=lw)
        ax.fill_between(x, y_base, y, color=color, alpha=0.3)

        # Calculate activity value at x_pos for this PF
        activity = np.exp(-0.5 * ((x_pos - center) / (width / 3)) ** 2)
        activity_y = y_base + height * activity

        # Draw activity dot at x_pos position on this PF
        dot = Circle((x_pos, activity_y), radius=SIZES["pf_dot_radius"], facecolor=color, edgecolor="none", linewidth=lw, zorder=10)
        ax.add_patch(dot)


def NeuronBlock(ax, x, y, w, h, n_neurons=4, color=None, label=None):
    """Draw RNN-style neurons in a circular arrangement with recurrent connections.

    Args:
        ax: Matplotlib axis
        x, y, w, h: Bounding box position and size
        n_neurons: Number of neurons to draw (default 3)
        color: Color of neurons
        label: Optional label below the block
    """
    color = PALETTE.get(color, color) if color else PALETTE["stroke"]

    # Calculate center and radius
    center_x = x + w / 2
    center_y = y + h / 2

    # Outer radius uses 85% of the smaller dimension to leave some padding
    outer_radius = 0.85 * min(w, h) / 2
    inner_radius = 0.5 * outer_radius  # Neurons at 50% of outer radius

    # Draw outer circle
    outer_circle = Circle((center_x, center_y), radius=outer_radius, facecolor="none", edgecolor=PALETTE["stroke"], linewidth=SIZES["lw"], zorder=1)
    ax.add_patch(outer_circle)

    # Calculate neuron positions evenly spaced around the circle
    # Start at top (90 degrees) and go clockwise
    neuron_positions = []
    angles = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False) + np.pi / 2  # Start at top

    for angle in angles:
        neuron_x = center_x + inner_radius * np.cos(angle)
        neuron_y = center_y + inner_radius * np.sin(angle)
        neuron_positions.append((neuron_x, neuron_y))

    # Draw neurons
    neuron_radius = outer_radius * 0.25  # Neuron size relative to outer radius
    for neuron_x, neuron_y in neuron_positions:
        neuron_circle = Circle(
            (neuron_x, neuron_y), radius=neuron_radius, facecolor=color, edgecolor=PALETTE["stroke"], linewidth=1.0, alpha=1, zorder=5
        )
        ax.add_patch(neuron_circle)

    # Draw curved arrows connecting each neuron to the next one (clockwise)
    # Arrows follow the circular path, curving slightly outward for visibility
    if n_neurons > 1:
        for i in range(n_neurons):
            # Current neuron and next neuron (wrapping around)
            current_pos = neuron_positions[i]
            next_pos = neuron_positions[(i + 1) % n_neurons]

            # For clockwise circular connections, use consistent curvature
            # The arrows should follow the circle path, curving outward slightly
            # Positive rad values curve one direction, negative the other
            # For clockwise around a circle, we want consistent positive curvature
            # The curvature should roughly match the circle radius
            arrow_curvature = 0.4  # Positive value for clockwise curvature

            ax.annotate(
                "",
                xy=(next_pos[0], next_pos[1]),
                xytext=(current_pos[0], current_pos[1]),
                arrowprops=dict(
                    arrowstyle="-",
                    connectionstyle=f"arc3,rad={arrow_curvature}",
                    color=PALETTE["stroke"],
                    lw=SIZES["lw"] * 0.8,
                    zorder=1,
                ),
            )

    # Add label
    if label:
        ax.text(x + w / 2, y - 0.05, label, ha="center", va="top", fontsize=FONTS["small"], color=PALETTE["stroke"])


def Box(ax, x, y, w, h, label, color="stroke", fill=False, rounded=True):
    """Draw a labeled box (for matrices/operations)."""
    color = PALETTE.get(color, color)

    if rounded:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", facecolor="white" if fill else "none", edgecolor=color, linewidth=SIZES["lw"])
    else:
        box = patches.Rectangle((x, y), w, h, facecolor="white" if fill else "none", edgecolor=color, linewidth=SIZES["lw"])

    ax.add_patch(box)

    # Add label
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=FONTS["small"], color=PALETTE["stroke"])


def Arrow(ax, x0, y0, x1, y1, style="solid", color="stroke", arrowstyle="->"):
    """Draw an arrow with specified style."""
    color = PALETTE.get(color, color)
    arrow_props = ARROW_STYLES.get(style, ARROW_STYLES["solid"]).copy()
    arrow_props["color"] = color

    if style == "double":
        # Draw double arrow (two-stage process)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle=arrowstyle, **arrow_props))
        # Add second arrow slightly offset
        offset = 0.01
        ax.annotate("", xy=(x1 + offset, y1), xytext=(x0 + offset, y0), arrowprops=dict(arrowstyle=arrowstyle, **arrow_props))
    else:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle=arrowstyle, **arrow_props))


def Legend(ax, items, x=0.02, y=0.98):
    """Draw a legend with colored items."""
    for i, (label, color_key) in enumerate(items):
        color = PALETTE.get(color_key, color_key)
        y_pos = y - i * 0.08

        # Color patch
        circle = Circle((x + 0.02, y_pos), radius=0.015, facecolor=color, edgecolor=PALETTE["stroke"])
        ax.add_patch(circle)

        # Label
        ax.text(x + 0.05, y_pos, label, ha="left", va="center", fontsize=FONTS["small"], color=PALETTE["stroke"])


def Thermometer(ax, x, y_base, total_height, fill_height, width=0.03):
    """Draw a thermometer indicator with parameterizable fill height.

    Args:
        ax: Matplotlib axis
        x: X position (center of thermometer)
        y_base: Base Y position (bottom of thermometer)
        total_height: Total height of thermometer
        fill_height: Height to fill (from bottom)
        width: Width of thermometer body
        color: Color of thermometer outline
        fill_color: Color of thermometer fill
    """
    # Ball (filled black circle) at the bottom
    ball_radius = width * 1.2
    ball_center_y = y_base + ball_radius

    # Body rectangle above the ball
    body_y_base = ball_center_y
    body_height = total_height - ball_radius
    body_width = width

    # Draw filled black circle at bottom
    ball = Circle((x, ball_center_y), radius=ball_radius, facecolor="black", edgecolor="black", linewidth=SIZES["lw"], zorder=2)
    ax.add_patch(ball)

    # Draw outer rounded rectangle (outline)
    outer_rect = patches.FancyBboxPatch(
        (x - body_width / 2, body_y_base),
        body_width,
        body_height,
        boxstyle=f"round,pad=0,rounding_size={body_width/2}",
        facecolor="none",
        edgecolor="black",
        linewidth=SIZES["lw"],
        zorder=2,
    )
    ax.add_patch(outer_rect)

    # Draw inner square rectangle (fill) - smaller and positioned to show fill height
    # fill_height is measured from body_y_base upward
    if fill_height > 0:
        fill_height_clipped = min(fill_height, body_height)
        if fill_height_clipped > 0:
            # Inner rectangle is slightly smaller (padding)
            inner_padding = body_width * 0.15
            inner_width = body_width - 2 * inner_padding
            inner_x = x - inner_width / 2
            inner_y = body_y_base
            inner_height = fill_height_clipped

            # Inner rectangle (square, no rounding)
            inner_rect = patches.Rectangle(
                (inner_x, inner_y),
                inner_width,
                inner_height,
                facecolor="black",
                edgecolor="none",
                zorder=1,
            )
            ax.add_patch(inner_rect)


def PFsArrayAlternateAmplitudes(ax, n_pfs, x_pos, width, main_height, y_base, track_x0=0.1, track_x1=0.9, color="target", lw=None, gain_ratios=None):
    """Draw an array of place fields with alternate gain amplitudes (lines only, no fill).
    Each place field shows multiple gain levels: 2 below main height, 2 above main height.

    Args:
        ax: Matplotlib axis
        n_pfs: Number of place fields to draw
        x_pos: Current position on track (x coordinate) - used to calculate activity dots
        width: Width of each place field
        main_height: Main height of place fields (the reference height)
        y_base: Base Y position for place fields
        track_x0: Left boundary of track
        track_x1: Right boundary of track
        color: Color of place fields
        lw: Line width (should be thinner than normal)
        gain_ratios: List of 4 gain ratios [below1, below2, above1, above2].
    """
    color = PALETTE.get(color, color)
    lw = lw or SIZES["lw"] * 0.5  # Default to thinner line

    # Default gain ratios: 2 below (0.6x, 0.8x) and 2 above (1.2x, 1.4x)
    if gain_ratios is None:
        gain_ratios = [0.85, 0.925, 1.075, 1.15]

    # Calculate PF centers spread across the track
    pf_centers = np.linspace(track_x0 + width, track_x1 - width, n_pfs)

    # Draw each PF with multiple gain lines
    for center in pf_centers:
        # Generate x coordinates for bell curve
        x = np.linspace(center - width, center + width, 50)

        # Draw 2 lines below main height
        # First below line (all PFs have lines at this height)
        below_height1 = main_height * gain_ratios[0]
        y_below1 = y_base + below_height1 * np.exp(-0.5 * ((x - center) / (width / 3)) ** 2)
        ax.plot(x, y_below1, color=color, linewidth=lw, alpha=0.7)

        # Second below line (all PFs have lines at this height)
        below_height2 = main_height * gain_ratios[1]
        y_below2 = y_base + below_height2 * np.exp(-0.5 * ((x - center) / (width / 3)) ** 2)
        ax.plot(x, y_below2, color=color, linewidth=lw, alpha=0.7)

        # Draw 2 lines above main height
        # First above line (all PFs have lines at this height)
        above_height1 = main_height * gain_ratios[2]
        y_above1 = y_base + above_height1 * np.exp(-0.5 * ((x - center) / (width / 3)) ** 2)
        ax.plot(x, y_above1, color=color, linewidth=lw, alpha=0.7)

        # Second above line (all PFs have lines at this height)
        above_height2 = main_height * gain_ratios[3]
        y_above2 = y_base + above_height2 * np.exp(-0.5 * ((x - center) / (width / 3)) ** 2)
        ax.plot(x, y_above2, color=color, linewidth=lw, alpha=0.7)


def HighDimBarplot(ax, track_y, n_bins=12, max_height=0.15, track_x0=0.1, track_x1=0.9, color="basis", bar_heights=None):
    """Draw a high-dimensional barplot representing position estimate.
    Bars are spread across the track width with no spacing and emerge upward from it.
    Outer edges of bars match the outer edges of the track.

    Args:
        ax: Matplotlib axis
        track_x: X position on track (reference point, bars span track width)
        track_y: Y position of track
        n_bins: Number of position bins/components (default 12)
        max_height: Maximum height of bars (default 0.15), used if bar_heights not provided
        track_x0: Left boundary of track
        track_x1: Right boundary of track
        color: Color of bars
        bar_heights: Optional array of bar heights (length n_bins). If None, generates random heights.
    """
    color = PALETTE.get(color, color)

    # Calculate bar width so bars span exactly from track_x0 to track_x1 with no spacing
    total_width = track_x1 - track_x0
    bar_width = total_width / n_bins

    # Generate or use provided bar heights
    if bar_heights is None:
        # Generate random amplitudes (normalized to max_height)
        # Use a fixed seed for reproducible results
        np.random.seed(42)
        amplitudes = np.random.rand(n_bins) * max_height
    else:
        # Use provided bar heights, ensure it matches n_bins
        if len(bar_heights) != n_bins:
            raise ValueError(f"bar_heights length ({len(bar_heights)}) must match n_bins ({n_bins})")
        amplitudes = np.array(bar_heights)

    # Calculate bar positions: bars start at track_x0 and fill to track_x1
    # Each bar starts at track_x0 + i * bar_width
    bar_centers = [None] * n_bins
    for i in range(n_bins):
        bar_x_left = track_x0 + i * bar_width
        bar_bottom = track_y
        amplitude = amplitudes[i]
        bar_centers[i] = bar_x_left + bar_width / 2

        # Draw bar (no spacing between bars)
        bar_rect = patches.Rectangle(
            (bar_x_left, bar_bottom),
            bar_width,
            amplitude,
            facecolor=color,
            edgecolor=PALETTE["stroke"],
            linewidth=SIZES["lw"] * 0.5,
            alpha=0.3,
            zorder=5,
        )
        ax.add_patch(bar_rect)

    return bar_centers


def LatentArray(ax, x, y, w, h, n_latents=12, color="basis", ball_radius=None, pinch=0.01):
    """Draw a linear array of latents as a rounded rectangle with balls inside.

    Args:
        ax: Matplotlib axis
        x, y, w, h: Bounding box position and size
        n_latents: Number of latent units (default 12)
        color: Color of latents
        ball_radius: Optional radius for balls. If None, uses size proportional to height.
    """
    color = PALETTE.get(color, color)

    # Calculate center and dimensions
    center_x = x + w / 2
    center_y = y + h / 2

    # Draw outer rounded rectangle (CSS-like rounded corners)
    # Use FancyBboxPatch with rounded corners - rounding size is half the height for pill shape
    rounding_size = h / 2
    rounded_rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={rounding_size}",
        facecolor="none",
        edgecolor=PALETTE["stroke"],
        linewidth=SIZES["lw"],
        zorder=1,
    )
    ax.add_patch(rounded_rect)

    # Calculate positions for latent balls along the horizontal axis
    # Spread them evenly across the width, positioned at center_y
    ball_positions_x = np.linspace(x + pinch, x + w - pinch, n_latents * 2 + 1)[1::2]

    # Ball size - use provided radius or calculate based on height
    if ball_radius is None:
        # Default to proportional size if not specified
        ball_radius = h * 0.25

    # Draw latent balls (no connections)
    for ball_x in ball_positions_x:
        ball = Circle((ball_x, center_y), radius=ball_radius, facecolor=color, edgecolor="k", linewidth=1.0, alpha=0.8, zorder=3)
        ax.add_patch(ball)

    # Return the ball center positions for potential use in arrows
    return ball_positions_x.tolist()
