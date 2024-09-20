import matplotlib.pyplot as plt
import numpy as np


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


# Example usage
total_sessions = 24
intro_red = 8  # Session where red (Env 1) is introduced
intro_blue = 16  # Session where blue (Env 2) is introduced
blocks_per_session = 10
training_sessions = 3  # Number of sessions for training phase

create_curriculum_visualization(total_sessions, intro_red, intro_blue, blocks_per_session, training_sessions, session_gap=0.7)
