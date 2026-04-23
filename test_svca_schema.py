"""
2D sketch of the cross-validated CVSC (SVD on rootA @ rootB).

Shows:
  - Two covariance ellipses (A and B) with shared + private structure
  - rootA and rootB as colored ellipses (the "coloring" operation)
  - The SVD: U, S, V = svd(rootA @ rootB)
  - Left singular vector u1 drawn on A's ellipse
  - Right singular vector v1 drawn on B's ellipse
  - The singular value annotated as the "shared variance" score
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

rng = np.random.default_rng(7)

# ── ground truth geometry ─────────────────────────────────────────────────────
# shared direction: tilted ~35 degrees
theta_shared = np.radians(35)
shared = np.array([np.cos(theta_shared), np.sin(theta_shared)])
perp = np.array([-np.sin(theta_shared), np.cos(theta_shared)])

# Covariance A: large shared variance, small private
lam_s_A, lam_p_A = 3.5, 0.4
cov_A = lam_s_A * np.outer(shared, shared) + lam_p_A * np.outer(perp, perp)

# Covariance B: moderate shared variance, larger private (different shape)
lam_s_B, lam_p_B = 2.2, 1.1
# B's private direction is slightly rotated from A's
theta_B_priv = theta_shared + np.radians(90)
priv_B = np.array([-np.sin(theta_shared + np.radians(15)), np.cos(theta_shared + np.radians(15))])
priv_B /= np.linalg.norm(priv_B)
# orthogonalise priv_B against shared
priv_B = priv_B - (priv_B @ shared) * shared
priv_B /= np.linalg.norm(priv_B)
cov_B = lam_s_B * np.outer(shared, shared) + lam_p_B * np.outer(priv_B, priv_B)


# ── matrix square roots ───────────────────────────────────────────────────────
def mat_sqrt(C):
    vals, vecs = np.linalg.eigh(C)
    return vecs @ np.diag(np.sqrt(np.maximum(vals, 0))) @ vecs.T


rootA = mat_sqrt(cov_A)
rootB = mat_sqrt(cov_B)

# ── SVD of rootA @ rootB ──────────────────────────────────────────────────────
M = rootA @ rootB
U, S, Vt = np.linalg.svd(M)
u1, u2 = U[:, 0], U[:, 1]  # left singular vectors  (live in A's space)
v1, v2 = Vt[0], Vt[1]  # right singular vectors (live in B's space)


# ── helpers ───────────────────────────────────────────────────────────────────
def ellipse_points(cov, n=300, scale=1.0):
    """Points on the 1-sigma ellipse of cov."""
    t = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)])
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(2))
    pts = scale * L @ circle
    return pts[0], pts[1]


def draw_arrow(ax, vec, scale, color, lw=2.2, zorder=5, label=None, label_offset=(0.05, 0.05)):
    ax.annotate("", xy=vec * scale, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14))
    ax.annotate("", xy=-vec * scale, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14))
    if label:
        lp = vec * scale * 1.18 + np.array(label_offset)
        ax.text(lp[0], lp[1], label, color=color, fontsize=11, fontweight="bold", ha="center", va="center")


# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
fig.patch.set_facecolor("#111111")

C_A = "#4A9EDB"  # blue  — dataset A
C_B = "#3DBF8A"  # green — dataset B
C_u = "#F5A623"  # orange — u1 (left SV, lives on A)
C_v = "#E06ADB"  # purple — v1 (right SV, lives on B)
C_shared = "#FFFFFF"  # white — ground truth shared direction
BG = "#111111"

arrow_scale = 1.55

for ax in axes:
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.axhline(0, color="#333", lw=0.5)
    ax.axvline(0, color="#333", lw=0.5)
    ax.tick_params(colors="#555", labelsize=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

# ── Panel 1: the two covariance ellipses ─────────────────────────────────────
ax = axes[0]
ax.set_title("covariance structure", color="#cccccc", fontsize=10, pad=8)

xA, yA = ellipse_points(cov_A, scale=1.0)
xB, yB = ellipse_points(cov_B, scale=1.0)
ax.fill(xA, yA, color=C_A, alpha=0.12)
ax.plot(xA, yA, color=C_A, lw=1.5, label="A (spont 1)")
ax.fill(xB, yB, color=C_B, alpha=0.12)
ax.plot(xB, yB, color=C_B, lw=1.5, label="B (spont 2)")

# ground truth shared direction
for s, label in [(1, "shared"), (-1, "")]:
    ax.annotate(
        "", xy=s * shared * 2.1, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C_shared, lw=1.2, linestyle="dashed", mutation_scale=10)
    )
ax.text(shared[0] * 2.25, shared[1] * 2.25, "true\nshared", color="#888", fontsize=8, ha="center", va="center")

ax.legend(loc="lower right", fontsize=8, framealpha=0.0, labelcolor="white", handlelength=1.5)

# ── Panel 2: root covariances = "coloring" ────────────────────────────────────
ax = axes[1]
ax.set_title("rootA @ rootB  (then SVD)", color="#cccccc", fontsize=10, pad=8)

# Draw rootA and rootB as ellipses (they have same eigenvectors as A, B)
xrA, yrA = ellipse_points(rootA @ rootA.T, scale=1.0)  # = cov_A again, visual only
xrB, yrB = ellipse_points(rootB @ rootB.T, scale=1.0)
ax.fill(xrA, yrA, color=C_A, alpha=0.12)
ax.plot(xrA, yrA, color=C_A, lw=1.5, alpha=0.6, linestyle="--")
ax.fill(xrB, yrB, color=C_B, alpha=0.12)
ax.plot(xrB, yrB, color=C_B, lw=1.5, alpha=0.6, linestyle="--")

# show the product matrix as a filled shape
xM, yM = ellipse_points(M @ M.T, scale=0.45)
ax.fill(xM, yM, color="#888888", alpha=0.18)
ax.plot(xM, yM, color="#888888", lw=1.0, alpha=0.5)
ax.text(0, -0.3, "rootA·rootB", color="#777", fontsize=8, ha="center")

# singular vectors
draw_arrow(ax, u1, arrow_scale, C_u, label="u₁", label_offset=(0.0, 0.12))
draw_arrow(ax, v1, arrow_scale * 0.82, C_v, label="v₁", label_offset=(0.1, -0.15))

# annotate singular value
mid = (u1 + v1) * 0.65
ax.text(
    mid[0] + 0.2,
    mid[1] + 0.5,
    f"σ₁ = {S[0]:.2f}",
    color="#cccccc",
    fontsize=9,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.3", fc="#222", ec="#555", lw=0.8),
)

# ── Panel 3: geometric interpretation ────────────────────────────────────────
ax = axes[2]
ax.set_title("shared mode: max coloured dot product", color="#cccccc", fontsize=10, pad=8)

xA, yA = ellipse_points(cov_A, scale=1.0)
xB, yB = ellipse_points(cov_B, scale=1.0)
ax.fill(xA, yA, color=C_A, alpha=0.12)
ax.plot(xA, yA, color=C_A, lw=1.5)
ax.fill(xB, yB, color=C_B, alpha=0.12)
ax.plot(xB, yB, color=C_B, lw=1.5)

# rootA @ u1 and rootB @ v1 — these are the "coloured" vectors
# their dot product = σ1
Au1 = rootA @ u1
Bv1 = rootB @ v1

scale_Au = 0.85
scale_Bv = 0.85
ax.annotate("", xy=Au1 * scale_Au, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C_u, lw=2.2, mutation_scale=14))
ax.annotate("", xy=Bv1 * scale_Bv, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C_v, lw=2.2, mutation_scale=14))

ax.text(Au1[0] * scale_Au * 1.2, Au1[1] * scale_Au * 1.2, "rootA·u₁", color=C_u, fontsize=9, ha="center", fontweight="bold")
ax.text(Bv1[0] * scale_Bv * 1.22 - 0.1, Bv1[1] * scale_Bv * 1.22, "rootB·v₁", color=C_v, fontsize=9, ha="center", fontweight="bold")

# dot product annotation
dot_val = Au1 @ Bv1
ax.text(
    0,
    -2.7,
    f"(rootA·u₁) · (rootB·v₁)  =  σ₁  =  {dot_val:.2f}",
    color="#aaaaaa",
    fontsize=8.5,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a1a", ec="#444", lw=0.8),
)

ax.text(0, -3.1, "maximised over all unit vectors u, v", color="#666", fontsize=7.5, ha="center")

plt.tight_layout(pad=1.5)
plt.show()
print("done")
