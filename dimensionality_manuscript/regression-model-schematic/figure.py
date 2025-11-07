"""Figure assembly for neuro-model schematics."""

import matplotlib.pyplot as plt
from config import SIZES, FONTS, PALETTE
from layout import grid_row, add_concept_axis, save_figure, PanelContext
from panels import PANEL_FUNCTIONS, PANEL_NAMES


def build_row(order=(1, 3, 2, 4, 5, 6), figsize=None, dpi=None, annotate_axis=True, **kwargs):
    """Build a row of model panels in specified order.

    Args:
        order: Tuple of panel numbers (1-6) in desired order
        figsize: Figure size override (width, height)
        dpi: DPI override
        annotate_axis: Whether to add conceptual axis annotation
        **kwargs: Additional options passed to panel functions

    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    n = len(order)
    figsize = figsize or (n * SIZES["panel_w"], SIZES["panel_h"])
    dpi = dpi or SIZES["dpi"]

    # Create figure and axes
    fig, axes = plt.subplots(1, n, figsize=figsize, dpi=dpi, constrained_layout=True)

    # Handle single panel case
    if n == 1:
        axes = [axes]

    # Set up each axis
    for ax in axes:
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

    # Draw each panel
    for ax, panel_idx in zip(axes, order):
        if panel_idx in PANEL_FUNCTIONS:
            ctx = PanelContext(ax)
            PANEL_FUNCTIONS[panel_idx](ax, ctx=ctx, opt=kwargs)
        else:
            raise ValueError(f"Unknown panel index: {panel_idx}")

    # Add conceptual axis annotation
    if annotate_axis and n > 1:
        add_concept_axis(fig, axes)

    return fig, axes


def build_single_panel(panel_idx, figsize=None, dpi=None, **kwargs):
    """Build a single model panel.

    Args:
        panel_idx: Panel number (1-6)
        figsize: Figure size override
        dpi: DPI override
        **kwargs: Additional options passed to panel function

    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    figsize = figsize or (SIZES["panel_w"], SIZES["panel_h"])
    dpi = dpi or SIZES["dpi"]

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)

    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    if panel_idx in PANEL_FUNCTIONS:
        ctx = PanelContext(ax)
        PANEL_FUNCTIONS[panel_idx](ax, ctx=ctx, opt=kwargs)
    else:
        raise ValueError(f"Unknown panel index: {panel_idx}")

    return fig, ax


def build_comparison(panel_pairs, figsize=None, dpi=None, titles=None):
    """Build comparison figure with paired panels.

    Args:
        panel_pairs: List of (panel1, panel2) tuples
        figsize: Figure size override
        dpi: DPI override
        titles: Optional list of titles for each pair

    Returns:
        fig, axes: Matplotlib figure and axes array
    """
    n_pairs = len(panel_pairs)
    figsize = figsize or (n_pairs * 2 * SIZES["panel_w"], SIZES["panel_h"])
    dpi = dpi or SIZES["dpi"]

    fig, axes = plt.subplots(1, n_pairs * 2, figsize=figsize, dpi=dpi, constrained_layout=True)

    if n_pairs == 1:
        axes = [axes] if not hasattr(axes, "__len__") else axes

    # Set up axes
    for ax in axes:
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

    # Draw panel pairs
    for i, (panel1, panel2) in enumerate(panel_pairs):
        ax1, ax2 = axes[i * 2], axes[i * 2 + 1]

        ctx1, ctx2 = PanelContext(ax1), PanelContext(ax2)
        PANEL_FUNCTIONS[panel1](ax1, ctx=ctx1, opt={})
        PANEL_FUNCTIONS[panel2](ax2, ctx=ctx2, opt={})

        # Add pair title if provided
        if titles and i < len(titles):
            # Center title between the two panels
            fig.text(
                (ax1.get_position().x0 + ax2.get_position().x1) / 2,
                ax1.get_position().y1 + 0.02,
                titles[i],
                ha="center",
                va="bottom",
                fontsize=FONTS["title"],
            )

    return fig, axes


def export_figure(fig, basename, formats=None, use_example_path=True, **kwargs):
    """Export figure in multiple formats.

    Args:
        fig: Matplotlib figure
        basename: Base filename (without extension)
        formats: List of formats to export ['pdf', 'svg', 'png']
        use_example_path: Whether to save to the example_path directory
        **kwargs: Additional arguments for savefig

    Returns:
        List of saved filenames
    """
    formats = formats or ["pdf"]
    return save_figure(fig, basename, formats=formats, use_example_path=use_example_path, **kwargs)


def create_full_schematic(order=None, export_path=None, formats=None):
    """Create and optionally export the complete schematic.

    Args:
        order: Panel order (default: recommended order)
        export_path: Path to export files (without extension)
        formats: Export formats

    Returns:
        fig, axes, exported_files
    """
    # Use recommended order if not specified
    order = order or (1, 3, 2, 4, 5, 6)

    # Build figure
    fig, axes = build_row(order=order, annotate_axis=True)

    # Export if requested
    exported_files = []
    if export_path:
        formats = formats or ["pdf", "svg"]
        exported_files = export_figure(fig, export_path, formats=formats)

    return fig, axes, exported_files


def create_spatial_progression():
    """Create figure showing spatial constraint progression."""
    # Spatial models only: External PF -> External PF+gain -> Internal PF -> Internal PF+gain
    order = (1, 3, 2, 4)
    fig, axes = build_row(order=order, annotate_axis=True)

    # Custom title for this progression
    fig.suptitle("Spatial Models: External → Internal Progression", y=0.995, fontsize=FONTS["title"] + 1, color=PALETTE["stroke"])

    return fig, axes


def create_method_comparison():
    """Create figure comparing different methodological approaches."""
    # Compare: External PF vs Internal PF vs High-D vs RRR
    order = (1, 2, 5, 6)
    fig, axes = build_row(order=order, annotate_axis=True)

    # Custom title
    fig.suptitle("Methodological Comparison: Spatial vs High-D vs Non-spatial", y=0.995, fontsize=FONTS["title"] + 1, color=PALETTE["stroke"])

    return fig, axes


# Convenience functions for common use cases
def quick_export(panel_or_order, filename, formats=["pdf", "svg"], use_example_path=True):
    """Quick export of a single panel or row."""
    if isinstance(panel_or_order, int):
        # Single panel
        fig, ax = build_single_panel(panel_or_order)
    else:
        # Row of panels
        fig, axes = build_row(order=panel_or_order, annotate_axis=True)

    exported = export_figure(fig, filename, formats=formats, use_example_path=use_example_path)
    plt.close(fig)
    return exported


def show_all_panels():
    """Display all panels individually for inspection."""
    figs = []
    for i in range(1, 7):
        fig, ax = build_single_panel(i)
        fig.suptitle(f"Panel {i}: {PANEL_NAMES[i]}", fontsize=FONTS["title"], color=PALETTE["stroke"])
        figs.append(fig)
    return figs
