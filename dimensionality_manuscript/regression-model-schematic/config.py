"""Configuration constants for the neuro-model schematic builder."""

example_path = "C:/Users/Andrew/Documents/GitHub/vrAnalysis/dimensionality_manuscript/regression-model-schematic/test-figures"

# Color palette (color-blind safe)
PALETTE = {
    "external": "#C9A227",  # Gold - true/measured position
    "internal": "#6B7280",  # Grey - decoded/internal position
    "source": "#2CA7A3",  # Teal - source neurons
    "target": "#D1495B",  # Red - target neurons
    "basis": "#7A5CCE",  # Purple - basis/features
    "stroke": "#111827",  # Charcoal - neutral lines
    "gray": "#6B7280",  # Grey - gray lines
}

# Size constants
SIZES = {
    "panel_w": 3.0,  # Panel width in inches
    "panel_h": 2.2,  # Panel height in inches
    "dpi": 300,  # Resolution for output
    "lw": 1.5,  # Line width
    "tick": 0.06,  # Tick size
    "pf_height": 0.25,  # Place field height
    "pf_dot_radius": 0.01,  # Place field dot radius for activity dots
}

# Font settings
FONTS = {
    "family": "DejaVu Sans",
    "size": 9,  # Default font size
    "small": 12,  # Small text
    "title": 10,  # Panel titles
}

# Arrow styles
ARROW_STYLES = {
    "solid": {"linestyle": "-", "alpha": 1.0},
    "dashed": {"linestyle": "--", "alpha": 0.8},
    "double": {"linestyle": "-", "alpha": 1.0, "linewidth": 3.0},
}

# Layout constants
LAYOUT = {
    # Fixed Y positions (consistent across all panels)
    "track_y": 0.42,  # Y position of track (same for external and internal)
    "pf_base_y": 0.45,  # Base Y position for place fields (legacy, may not be used)
    "neuron_y": 0.60,  # Y position for neuron blocks
    "margin": 0.05,  # General margin
    # Position offsets
    "x_label_offset_below": 0.06,  # Offset below track/gold point for x(t) or x̂(t) labels
    "g_label_offset_below_thermometer": 0.03,  # Offset below thermometer for g label
    "pf_offset_above_track": 0.05,  # Distance from track to target PF base
    "yhat_offset_above_pf": 0.1,  # Distance from PF top to ŷ(t) label (standard)
    "yhat_offset_above_pf_gain": 0.15,  # Distance from PF top to g·ŷ(t) label (with gain)
    "arrow_offset_from_pf_top": 0.02,  # Arrow offset from PF top to label
    "arrow_offset_from_label": 0.01,  # Arrow offset from label bottom
    # Source PF region (for internal panels only)
    "source_pf_scale": 0.666,  # Source PF height multiplier
    "source_pf_spacing_above": 0.18,  # Space between source PFs and track
    # Target PF scaling
    "target_pf_scale": 0.8,  # Target PF height multiplier
    "target_pf_scale_gain": 0.8,  # Target PF height for gain panels
    # X positions
    "pos_x_external": 0.575,  # X position for external panels (centered-right)
    "pos_x_internal": 0.375,  # X position for internal panels (centered-left)
    "track_x0": 0.1,  # Left boundary of track
    "track_x1": 0.9,  # Right boundary of track
    # PF dimensions
    "pf_width_1d": 0.15,  # Width of place fields 1-D
    "pf_width_highd": 0.18,  # Width of place fields high-D
    # Thermometer settings
    "thermometer_x_offset": 0.06,  # Offset to left of track
    "thermometer_width": 0.025,  # Width of thermometer body
    "thermometer_arrow_offset": 0.02,  # X offset for thermometer arrow start
    "thermometer_height_multiplier": 1.4,  # Multiplier for thermometer total height
}

# Panel-specific constants
N_PFS = 6  # Number of place fields to display in panels 1-4
