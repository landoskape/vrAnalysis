import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import matplotlib as mpl

import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
from scripts.dimilibi.helpers import figure_folder
from _old_vrAnalysis import helpers


def create_arrow(ax, start, length, head_length=0.4, direction="right", color="gray", width=0.15):
    end = start + length if direction == "right" else start - length
    ax.arrow(start, 0, length, 0, head_width=width, head_length=head_length, fc=color, ec=color, width=width * 0.5, length_includes_head=True)
    return end


def add_text(ax, x, y, text, va="bottom", ha="center", **kwargs):
    ax.text(x, y, text, va=va, ha=ha, **kwargs)


def create_trial_structure_schematic(colors, sizes, positions, yscale=1):
    fig, ax = plt.subplots(figsize=(12, 4))

    fontsize = 28
    # ITI (gray screen) arrow
    end1 = create_arrow(ax, positions["iti_start"], sizes["iti"], color=colors["iti"])
    add_text(ax, positions["iti_start"] + sizes["iti"] / 2, 2 * yscale, "ITI", fontsize=fontsize)
    add_text(ax, positions["iti_start"] + sizes["iti"] / 2, 1 * yscale, "(gray screen)", fontsize=fontsize * 0.75)
    add_text(ax, positions["iti_start"] + sizes["iti"] / 2, -2.5 * yscale, "1-2.5 s", fontsize=fontsize)

    # VR Navigation arrow
    end2 = create_arrow(ax, end1 + positions["gap"], sizes["vr"], color=colors["vr"])
    add_text(ax, end1 + positions["gap"] + sizes["vr"] / 2, 1.5 * yscale, "VR Navigation", fontsize=fontsize)
    add_text(ax, end1 + positions["gap"] + sizes["vr"] / 2, -2.5 * yscale, "< 2 minutes", fontsize=fontsize)

    # ITI with environment change arrow
    create_arrow(ax, end2 + positions["gap"], sizes["iti_change"], color=colors["iti_change"])
    # add_text(ax, end2 + positions['gap'] + sizes['iti_change']/2, 2*yscale, "ITI", fontsize=fontsize)
    add_text(ax, end2 + positions["gap"] + sizes["iti_change"] / 2, 1.5 * yscale, "p(change env) = 0.2", fontsize=fontsize * 0.75)

    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(0, end2 + positions["gap"] + sizes["iti_change"] + 0.5)
    ax.axis("off")

    plt.title("Trial Structure Schematic")
    plt.tight_layout()
    plt.show()


def create_curriculum_visualization(total_sessions, intro_red, intro_blue, blocks_per_session, training_sessions, session_gap=0.2):
    fig, ax = plt.subplots(figsize=(15, 4))

    colors = ["black", "red", "blue"]
    current_x = 0

    for session in range(total_sessions):
        if session < intro_red:
            session_colors = [0] * blocks_per_session
        elif session < intro_blue:
            session_colors = np.random.choice([0, 1], blocks_per_session)
        else:
            session_colors = np.random.choice([0, 1, 2], blocks_per_session)

        for block_color in session_colors:
            ax.barh(0, 1, left=current_x, height=0.5, color=colors[block_color])
            current_x += 1

        if session < total_sessions - 1:
            ax.barh(0, session_gap, left=current_x, height=0.5, color="white")
            current_x += session_gap

    # Add training and imaging bars
    training_width = (blocks_per_session + session_gap) * training_sessions - session_gap
    imaging_width = current_x - training_width

    ax.barh(0.4, training_width, left=0, height=0.06, color="k", alpha=1)
    ax.text(training_width / 2, 0.55, "Training", ha="center", va="bottom", fontsize=28)

    ax.barh(0.4, imaging_width, left=training_width, height=0.06, color="g", alpha=1)
    ax.text(training_width + imaging_width / 2, 0.55, "Imaging", ha="center", va="bottom", fontsize=28)

    ax.set_ylim(-0.5, 1.7)
    ax.set_xlim(0, current_x)
    ax.axis("off")

    # Add legend for environments
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none") for c in colors]
    ax.legend(legend_elements, ["Env 0", "Env 1", "Env 2"], loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    plt.title("Curriculum and Block Structure Visualization")
    plt.tight_layout()
    plt.show()


def create_pca_schematic(
    original_color="lightgray",
    pc_color="lightblue",
    scores_color="lightyellow",
    cov_color="lightgreen",
    width=10,
    height=18,
    scale_factor=1,
    font_size=10,
):

    fig, ax = plt.subplots(figsize=(width * scale_factor, height * scale_factor))

    # Calculate positions based on scale_factor
    unit = scale_factor
    spacing = 0.5 * unit

    # Define rectangle dimensions
    orig_width, orig_height = 3 * unit, 5 * unit
    pc_width, pc_height = 3 * unit, 5 * unit
    scores_width, scores_height = 3 * unit, 3 * unit

    def add_rectangle(x, y, w, h, color, label, is_test=False, **text_props):
        rect = patches.Rectangle((x, y), w, h, facecolor=color, edgecolor="black")
        if is_test:
            rect.set_hatch("/")
            rect._hatch_color = mpl.colors.to_rgba("k", alpha=0.2)
        ax.add_patch(rect)
        default_text_props = dict(ha="center", va="center", fontsize=font_size, wrap=True)
        default_text_props.update(text_props)
        ax.text(x + w / 2, y + h / 2, label, **default_text_props)

    # Training data rectangle
    train_y = orig_height + spacing
    add_rectangle(0, train_y, orig_width, orig_height, original_color, "Train\nPlace\nFields")

    # Equal sign
    ax.text(orig_width + spacing / 2, train_y + orig_height / 2, "=", ha="center", va="center", fontsize=font_size)

    # Principal Components rectangle (Train)
    pc_start = orig_width + spacing
    add_rectangle(pc_start, train_y + (orig_height - pc_height) / 2, pc_width, pc_height, pc_color, "Train\nPCs")

    # Multiplication sign
    ax.text(pc_start + pc_width + spacing / 2, train_y + orig_height / 2, "×", ha="center", va="center", fontsize=font_size)

    # Scores rectangle (Train)
    scores_start = pc_start + pc_width + spacing
    add_rectangle(scores_start, train_y + (orig_height - scores_height) / 2, scores_width, scores_height, scores_color, "Train\nScores")

    # Testing data rectangle
    test_y = 0
    add_rectangle(0, test_y, orig_width, orig_height, original_color, "Test\nPlace\nFields", is_test=True)

    # Multiplication sign
    ax.text(orig_width + spacing / 2, test_y + orig_height / 2, "×", ha="center", va="center", fontsize=font_size)

    # PCs rectangle (Test, same as Train)
    add_rectangle(pc_start, test_y + (orig_height - pc_height) / 2, pc_width, pc_height, pc_color, "Train\nPCs")

    # Equal sign
    ax.text(pc_start + pc_width + spacing / 2, test_y + orig_height / 2, "=", ha="center", va="center", fontsize=font_size)

    # Test Scores rectangle
    add_rectangle(
        scores_start, test_y + (orig_height - scores_height) / 2, scores_width, scores_height, scores_color, "Cross-Val.\nTest\nScores", is_test=True
    )

    # Covariance column
    cov_width = pc_height
    cov_height = unit * 1.6
    cov_start_x = scores_start + scores_width + 2 * spacing - cov_width / 2
    cov_start_y = orig_height + spacing / 2 - cov_height / 2
    add_rectangle(cov_start_x, cov_start_y, cov_width, cov_height, cov_color, "cov(each PC)\n:= eigenspectrum")

    # Arrows from Scores to Covariance
    # arrow_props = dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black', lw=1)
    arrow_props = dict(
        arrowstyle="->", color="black", lw=2, shrinkA=0, shrinkB=0, patchA=None, patchB=None, connectionstyle="arc3,rad=0", mutation_scale=20
    )
    ax.annotate(
        "",
        xy=(cov_start_x + cov_width / 2, cov_start_y + cov_height),
        xytext=(scores_start + scores_width, train_y + orig_height / 2),
        arrowprops=arrow_props,
    )
    ax.annotate(
        "",
        xy=(cov_start_x + cov_width / 2, cov_start_y),
        xytext=(scores_start + scores_width, test_y + orig_height / 2),
        arrowprops=arrow_props,
    )

    # Labels
    ax.text(-spacing / 2, train_y + orig_height / 2, "ROIs", ha="right", va="center", rotation=90, fontsize=font_size)
    ax.text(-spacing / 2, test_y + orig_height / 2, "ROIs", ha="right", va="center", rotation=90, fontsize=font_size)
    ax.text(orig_width / 2, train_y + orig_height + spacing / 2, "position", ha="center", va="bottom", fontsize=font_size)

    # Set limits and turn off axis
    xlimmax = cov_start_x + cov_width + spacing
    ylimmax = train_y + orig_height + spacing
    ax.set_xlim(-spacing, xlimmax)
    ax.set_ylim(-spacing, ylimmax)
    ax.axis("off")

    return fig, ax


def create_neuron_set_schematic(
    width=5,
    height=5,
    box_width_ratio=1,
    box_height_ratio=1,
    borderline_width=20,
    line_width=10,
    arrow_hpad=0.1,
    arrow_vpad=0.1,
    arrowline_vpad=0.3,
    text_size=12,
    colors={"box": "white", "arrow": "lightgray", "text": "black"},
):
    fig, ax = plt.subplots(figsize=(width, height))

    # Calculate positions
    box_width = width * box_width_ratio
    box_height = height * box_height_ratio
    left = (width - box_width) / 2
    bottom = (height - box_height) / 2
    mid_x = left + box_width / 2
    mid_y = bottom + box_height / 2

    # Draw the main box
    ax.add_patch(patches.Rectangle((left, bottom), box_width, box_height, fill=False, edgecolor=colors["text"], linewidth=borderline_width))

    # Draw the internal lines
    ax.plot([mid_x, mid_x], [bottom, bottom + box_height], color=colors["text"], linewidth=line_width)
    ax.plot([left, left + box_width], [mid_y, mid_y], color=colors["text"], linewidth=line_width)

    # Add arrows
    arrow_left = left + arrow_hpad * box_width / 2
    arrow_right = left + box_width - arrow_hpad * box_width / 2
    arrow_height = box_height / 2 * (1 - 2 * arrow_vpad)
    arrow_width = box_width / 4 - arrow_hpad * box_width / 2
    arrow_line_height = box_height / 2 * (1 - 2 * arrowline_vpad)
    arrow_xpos = [
        arrow_left,
        arrow_left + arrow_width,
        arrow_left + arrow_width,
        arrow_right - arrow_width,
        arrow_right - arrow_width,
        arrow_right,
        arrow_right - arrow_width,
        arrow_right - arrow_width,
        arrow_left + arrow_width,
        arrow_left + arrow_width,
    ]
    arrow_ypos = [
        0,
        arrow_height / 2,
        arrow_line_height / 2,
        arrow_line_height / 2,
        arrow_height / 2,
        0,
        -arrow_height / 2,
        -arrow_line_height / 2,
        -arrow_line_height / 2,
        -arrow_height / 2,
    ]
    arrow_xpos = np.array(arrow_xpos)
    arrow_ypos = np.array(arrow_ypos)
    xy_above = np.stack((arrow_xpos, arrow_ypos + mid_y + box_height / 4)).T
    xy_below = np.stack((arrow_xpos, arrow_ypos + mid_y - box_height / 4)).T

    polygon_above = patches.Polygon(xy_above, closed=True, edgecolor=colors["arrow"], facecolor=colors["arrow"], zorder=3)
    polygon_below = patches.Polygon(xy_below, closed=True, edgecolor=colors["arrow"], facecolor=colors["arrow"], zorder=3)
    ax.add_patch(polygon_above)
    ax.add_patch(polygon_below)

    # Add text
    ax.text(width / 2, bottom + box_height + 0.1 * height, "neurons", ha="center", va="bottom", fontsize=text_size)
    ax.text(left - 0.1 * width, mid_y, "timepoints", ha="right", va="center", fontsize=text_size, rotation=90)
    ax.text(mid_x - box_width / 4, bottom + box_height + 0.02 * height, "set 1", ha="center", va="bottom", fontsize=text_size)
    ax.text(mid_x + box_width / 4, bottom + box_height + 0.02 * height, "set 2", ha="center", va="bottom", fontsize=text_size)
    ax.text(left - 0.02 * width, bottom + 3 * box_height / 4, "train", rotation=90, ha="right", va="center", fontsize=text_size)
    ax.text(left - 0.02 * width, bottom + box_height / 4, "test", rotation=90, ha="right", va="center", fontsize=text_size)
    ax.text(mid_x, mid_y + box_height / 4, "measure\ncovariance", ha="center", va="center", fontsize=text_size, color="k")
    ax.text(mid_x, mid_y - box_height / 4, "evaluate\nconsistency", ha="center", va="center", fontsize=text_size, color="k")

    max_dim = max(width, height)
    ax.set_xlim(-0.1, max_dim + 0.1)
    ax.set_ylim(-0.1, max_dim + 0.1)
    ax.axis("off")
    ax.axis("equal")

    return fig, ax


def create_rrr_diagram(sizes=[5, 2, 5], ball_size=0.2, line_width=2, colors=["k", "k", "k"], figsize=(8, 8), dpi=100, xscale=2, fontsize=18):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(-0.5, 2 * xscale + 0.5)
    ax.set_ylim(0, max(sizes) + 1)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Calculate vertical offsets to center the balls
    offsets = [(max(sizes) - size) / 2 for size in sizes]

    # Create balls
    for x, size in enumerate(sizes):
        for j in range(size):
            y = j + 1 + offsets[x]
            circle = plt.Circle((x * xscale, y), ball_size, facecolor=colors[x], edgecolor="black", linewidth=line_width)
            ax.add_patch(circle)

    # Create connections
    for i in range(len(sizes) - 1):
        for j in range(sizes[i]):
            for k in range(sizes[i + 1]):
                xpos = np.array([i, i + 1]) * xscale
                ax.plot(xpos, [j + 1 + offsets[i], k + 1 + offsets[i + 1]], color="k", linewidth=line_width, alpha=1)

    # Add labels
    ax.text(0 * xscale, 0.3, "Input", ha="center", va="center", fontsize=fontsize)
    ax.text(1 * xscale, 0.6, "Latent:", ha="center", va="center", fontsize=fontsize)
    ax.text(1 * xscale, 0.3, "Encode Position", ha="center", va="center", fontsize=fontsize)
    ax.text(1 * xscale, 0.0, "Unconstrained", ha="center", va="center", fontsize=fontsize)

    ax.text(2 * xscale, 0.3, "Output", ha="center", va="center", fontsize=fontsize)

    plt.tight_layout()
    return fig, ax


def create_rrr_diagram_vertical(sizes=[5, 2, 5], ball_size=0.2, line_width=2, colors=["k", "k", "k"], figsize=(8, 8), dpi=100, yscale=2, fontsize=18):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, max(sizes) + 1)
    ax.set_ylim(-0.5, 2 * yscale + 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Calculate horizontal offsets to center the balls
    offsets = [(max(sizes) - size) / 2 for size in sizes]

    # Create connections
    for i in range(len(sizes) - 1):
        for j in range(sizes[i]):
            for k in range(sizes[i + 1]):
                ypos = np.array([i, i + 1]) * yscale
                ax.plot([j + 1 + offsets[i], k + 1 + offsets[i + 1]], ypos, color="k", linewidth=line_width, alpha=1)

    # Create balls
    for y, size in enumerate(sizes):
        for j in range(size):
            x = j + 1 + offsets[y]
            circle = plt.Circle((x, y * yscale), ball_size, facecolor=colors[y], edgecolor="black", linewidth=line_width, zorder=10000)
            ax.add_patch(circle)

    # Add labels
    ax.text(-0.1, 0 * yscale, "Target\nROIs", ha="center", va="center", fontsize=fontsize)
    ax.text(-0.1, 1 * yscale, "Latent:\nRBF(Pos)\nRRR", ha="center", va="center", fontsize=fontsize)
    ax.text(-0.1, 2 * yscale, "Source\nROIs", ha="center", va="center", fontsize=fontsize)

    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    total_sessions = 24
    intro_red = 8  # Session where red (Env 1) is introduced
    intro_blue = 16  # Session where blue (Env 2) is introduced
    blocks_per_session = 10
    training_sessions = 3  # Number of sessions for training phase
    create_curriculum_visualization(total_sessions, intro_red, intro_blue, blocks_per_session, training_sessions, session_gap=0.7)

    colors = {"iti": "gray", "vr": "black", "iti_change": "gray"}
    sizes = {"iti": 1, "vr": 4, "iti_change": 1}
    positions = {"iti_start": 0.5, "gap": 0.05}
    create_trial_structure_schematic(colors, sizes, positions, yscale=0.1)

    pca_colors = dict(
        original_color="lightgray",
        pc_color="lightgray",
        scores_color="lightgray",
        cov_color="lightgray",
    )
    fig, ax = create_pca_schematic(scale_factor=2, font_size=20, width=5, height=4, **pca_colors)
    plt.show()

    fig, ax = create_neuron_set_schematic(
        width=4.5,
        height=4.5,
        box_width_ratio=1,
        box_height_ratio=1,
        borderline_width=12,
        line_width=5,
        arrow_hpad=0.15,
        arrow_vpad=0.15,
        arrowline_vpad=0.275,
        text_size=22,
        colors={"box": "white", "arrow": "lightgray", "text": "black"},
    )
    plt.tight_layout()
    plt.show()

    # save_directory = figure_folder()
    # save_name = "SVCA_Schematic"
    # save_path = save_directory / save_name
    # helpers.save_figure(fig, save_path)

    fig, ax = create_rrr_diagram(ball_size=0.3, line_width=5, xscale=2, fontsize=24)
    fig, ax = create_rrr_diagram_vertical(ball_size=0.3, line_width=5, yscale=2, fontsize=24)
