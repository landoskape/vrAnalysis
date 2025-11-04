"""Build a 2x3 grid of all 6 panels."""

import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SIZES
from layout import PanelContext, save_figure
from panels import PANEL_FUNCTIONS

# Create 2x3 subplot figure with specified size
# Note: Use higher DPI for saving, but lower for display
fig, axes = plt.subplots(2, 3, figsize=(9, 6), dpi=100, constrained_layout=True)

# Flatten axes array for easier iteration
axes_flat = axes.flatten()

# Set up each axis and draw corresponding panel
for idx, ax in enumerate(axes_flat):
    panel_num = idx + 1  # Panel numbers are 1-indexed

    # Set up axis
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    # Draw panel
    if panel_num in PANEL_FUNCTIONS:
        ctx = PanelContext(ax)
        PANEL_FUNCTIONS[panel_num](ax, ctx=ctx, opt={})
    else:
        raise ValueError(f"Unknown panel index: {panel_num}")

# Save the figure to test-figures directory (with high DPI for print quality)
save_figure(fig, "panels_2x3", formats=["pdf"], use_example_path=True)

# Show the figure
# plt.show()
