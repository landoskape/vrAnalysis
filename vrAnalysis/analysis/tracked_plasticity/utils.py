import numpy as np
from matplotlib import pyplot as plt
from ...helpers import format_spines


def all_combos(idx_ses: list[int], max_session_diff: int, continuous: bool = True):
    """
    Generate all possible combinations of sessions with a maximum session difference
    """
    combos = []
    for ireference in range(len(idx_ses) - 1):
        for icompare in range(ireference + 1, len(idx_ses)):
            if (icompare - ireference) <= max_session_diff:
                if continuous:
                    idx_combo = idx_ses[ireference : icompare + 1]
                else:
                    idx_combo = [idx_ses[ireference], idx_ses[icompare]]
                combos.append(tuple(idx_combo))
    return combos


def generate_combo_display(num_ses: int, max_session_diff: int, continuous: bool = True):
    combos = all_combos(range(num_ses), max_session_diff=max_session_diff, continuous=continuous)
    num_combos = len(combos)
    combo_display = np.zeros((num_combos, num_ses))
    for icombo, combo in enumerate(combos):
        combo_display[icombo, combo] = 1

    fig = plt.figure(figsize=(3, 6), layout="constrained")
    ax = fig.add_subplot(111)
    ax.imshow(combo_display, cmap="gray_r", aspect="auto", interpolation="none")
    for icombo, combo in enumerate(combos):
        ax.text(combo[0], icombo, "PF", ha="center", va="center", fontsize=12, fontweight="bold", color="white")
        ax.text(combo[-1], icombo, "??", ha="center", va="center", fontsize=12, fontweight="bold", color="red")
    format_spines(
        ax,
        x_pos=-0.05,
        y_pos=-0.03,
        xbounds=(0, num_ses - 1),
        xticks=range(num_ses),
        yticks=[],
        spines_visible=["bottom"],
        tick_length=4,
    )
    ax.set_xlabel("Session #")
    ax.set_title("Tracked Session\nCombo Analysis")
    return fig, ax
