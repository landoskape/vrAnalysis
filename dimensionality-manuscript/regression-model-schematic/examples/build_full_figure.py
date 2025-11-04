#!/usr/bin/env python3
"""
Example script to build and export the complete CA1 model schematic.

This script demonstrates how to use the schematic library to create
publication-ready figures of the six CA1 prediction models.
"""

import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figure import create_full_schematic, create_spatial_progression, create_method_comparison, show_all_panels, quick_export


def main():
    """Main function to build and save schematics."""
    print("Building CA1 model schematics...")

    # 1. Create the full schematic with recommended ordering
    print("\n1. Creating full schematic (recommended order: 1→3→2→4→5→6)")
    fig, axes, exported = create_full_schematic(export_path="ca1_models_full", formats=["pdf", "svg", "png"])
    print(f"   Exported: {exported}")

    # 2. Create spatial progression figure
    print("\n2. Creating spatial progression figure")
    fig_spatial, axes_spatial = create_spatial_progression()
    spatial_files = quick_export((1, 3, 2, 4), "ca1_spatial_progression")
    print(f"   Exported: {spatial_files}")
    plt.close(fig_spatial)

    # 3. Create method comparison
    print("\n3. Creating method comparison figure")
    fig_methods, axes_methods = create_method_comparison()
    method_files = quick_export((1, 2, 5, 6), "ca1_method_comparison")
    print(f"   Exported: {method_files}")
    plt.close(fig_methods)

    # 4. Create individual panels for inspection
    print("\n4. Creating individual panels")
    individual_figs = show_all_panels()
    for i, fig in enumerate(individual_figs, 1):
        files = quick_export(i, f"ca1_panel_{i}")
        print(f"   Panel {i} exported: {files}")
        plt.close(fig)

    # 5. Show interactive version
    print("\n5. Displaying interactive version...")
    fig_interactive, axes_interactive, _ = create_full_schematic()
    plt.show()

    print("\nAll schematics created successfully!")


def test_individual_panels():
    """Test function to verify each panel renders correctly."""
    print("Testing individual panels...")

    from panels import PANEL_FUNCTIONS, PANEL_NAMES
    from layout import PanelContext

    for panel_idx in range(1, 7):
        print(f"  Testing Panel {panel_idx}: {PANEL_NAMES[panel_idx]}")

        try:
            fig, ax = plt.subplots(1, 1, figsize=(3, 2.2), dpi=150)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.axis("off")

            ctx = PanelContext(ax)
            PANEL_FUNCTIONS[panel_idx](ax, ctx=ctx, opt={})

            plt.close(fig)
            print(f"    ✓ Panel {panel_idx} rendered successfully")

        except Exception as e:
            print(f"    ✗ Panel {panel_idx} failed: {e}")

    print("Panel testing complete.")


def demo_customization():
    """Demonstrate customization options."""
    print("\nDemonstrating customization options...")

    # Custom ordering
    custom_order = (6, 5, 4, 3, 2, 1)  # Reverse order
    fig, axes = create_full_schematic(order=custom_order)
    fig.suptitle("Custom Order: Unconstrained → Spatially Constrained", y=0.995, fontsize=11)

    files = quick_export(custom_order, "ca1_custom_order")
    print(f"Custom order exported: {files}")
    plt.close(fig)

    # Subset of panels
    subset = (1, 5, 6)  # Just external, high-D, and RRR
    files = quick_export(subset, "ca1_subset")
    print(f"Subset exported: {files}")


if __name__ == "__main__":
    # Test individual panels first
    test_individual_panels()

    # Run main demo
    main()

    # Show customization options
    demo_customization()

    print("\n" + "=" * 50)
    print("Example script completed successfully!")
    print("Check the current directory for exported files.")
    print("=" * 50)
