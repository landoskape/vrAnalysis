"""Layout helpers and panel context for schematic building."""

import matplotlib.pyplot as plt
import os
from config import SIZES, FONTS, PALETTE, example_path


class PanelContext:
    """Convenience methods for panel-local operations."""

    def __init__(self, ax):
        self.ax = ax

        # Define common anchor points
        self.anchors = {
            "top_left": (0.05, 0.95),
            "top_center": (0.5, 0.95),
            "top_right": (0.95, 0.95),
            "center": (0.5, 0.5),
            "bottom_left": (0.05, 0.05),
            "bottom_center": (0.5, 0.05),
            "bottom_right": (0.95, 0.05),
            "track_left": (0.1, 0.25),
            "track_center": (0.5, 0.25),
            "track_right": (0.9, 0.25),
            "pf_base": (0.5, 0.45),
            "neuron_left": (0.15, 0.65),
            "neuron_right": (0.85, 0.65),
        }

    def anchor(self, name):
        """Get coordinates for a named anchor point."""
        return self.anchors.get(name, (0.5, 0.5))

    def text_center(self, x, y, s, size="small", color="stroke", **kwargs):
        """Add centered text at specified position."""
        fontsize = FONTS.get(size, FONTS["size"])
        color = PALETTE.get(color, color)

        self.ax.text(x, y, s, ha="center", va="center", fontsize=fontsize, color=color, **kwargs)

    def text_left(self, x, y, s, size="small", color="stroke", **kwargs):
        """Add left-aligned text at specified position."""
        fontsize = FONTS.get(size, FONTS["size"])
        color = PALETTE.get(color, color)

        self.ax.text(x, y, s, ha="left", va="center", fontsize=fontsize, color=color, **kwargs)

    def text_right(self, x, y, s, size="small", color="stroke", **kwargs):
        """Add right-aligned text at specified position."""
        fontsize = FONTS.get(size, FONTS["size"])
        color = PALETTE.get(color, color)

        self.ax.text(x, y, s, ha="right", va="center", fontsize=fontsize, color=color, **kwargs)

    def title(self, text, subtitle=None):
        """Add panel title and optional subtitle."""
        self.text_center(0.5, 0.95, text, size="title")
        if subtitle:
            self.text_center(0.5, 0.89, subtitle, size="small")

    def equation(self, text, y=0.07):
        """Add equation at bottom of panel."""
        self.text_center(0.5, y, text, size="small")

    def setup_panel(self):
        """Set up panel coordinate system and appearance."""
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect("equal")
        self.ax.axis("off")


def grid_row(n_panels, panel_size=None, dpi=None):
    """Create a row of panels with consistent sizing."""
    panel_size = panel_size or (SIZES["panel_w"], SIZES["panel_h"])
    dpi = dpi or SIZES["dpi"]

    figsize = (n_panels * panel_size[0], panel_size[1])
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, dpi=dpi, constrained_layout=True)

    # Handle single panel case
    if n_panels == 1:
        axes = [axes]

    # Set up each panel
    for ax in axes:
        ctx = PanelContext(ax)
        ctx.setup_panel()

    return fig, axes


def grid_matrix(n_rows, n_cols, panel_size=None, dpi=None):
    """Create a matrix of panels."""
    panel_size = panel_size or (SIZES["panel_w"], SIZES["panel_h"])
    dpi = dpi or SIZES["dpi"]

    figsize = (n_cols * panel_size[0], n_rows * panel_size[1])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, constrained_layout=True)

    # Handle single panel cases
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    # Set up each panel
    for row in axes:
        for ax in row if isinstance(row, list) else [row]:
            ctx = PanelContext(ax)
            ctx.setup_panel()

    return fig, axes


def add_concept_axis(fig, axes, title="Spatially constrained → Unconstrained"):
    """Add conceptual axis annotation above panels."""
    fig.suptitle(title, y=0.995, fontsize=FONTS["title"] + 1, color=PALETTE["stroke"])

    # Add region labels if we have enough panels
    if len(axes) >= 3:
        y_pos = 1.06

        # External
        axes[0].text(0.08, y_pos, "External", transform=axes[0].transAxes, ha="left", va="bottom", fontsize=FONTS["small"], color=PALETTE["stroke"])

        # Internal (middle)
        mid_idx = len(axes) // 2
        axes[mid_idx].text(
            0.5, y_pos, "Internal", transform=axes[mid_idx].transAxes, ha="center", va="bottom", fontsize=FONTS["small"], color=PALETTE["stroke"]
        )

        # Peer (last)
        axes[-1].text(0.92, y_pos, "Peer", transform=axes[-1].transAxes, ha="right", va="bottom", fontsize=FONTS["small"], color=PALETTE["stroke"])


def save_figure(fig, path, formats=None, use_example_path=True, **kwargs):
    """Save figure in multiple formats with consistent settings."""
    formats = formats or ["pdf"]

    # Use example_path by default unless path is absolute or use_example_path is False
    if use_example_path and not os.path.isabs(path):
        # Create the output directory if it doesn't exist
        os.makedirs(example_path, exist_ok=True)
        base_filename = os.path.basename(path)
        path = os.path.join(example_path, base_filename)

    default_kwargs = {"bbox_inches": "tight", "dpi": SIZES["dpi"], "facecolor": "white", "edgecolor": "none"}
    default_kwargs.update(kwargs)

    saved_files = []
    for fmt in formats:
        if "." not in path:
            filepath = f"{path}.{fmt}"
        else:
            base_path = path.rsplit(".", 1)[0]
            filepath = f"{base_path}.{fmt}"

        fig.savefig(filepath, format=fmt, **default_kwargs)
        saved_files.append(filepath)

    return saved_files
