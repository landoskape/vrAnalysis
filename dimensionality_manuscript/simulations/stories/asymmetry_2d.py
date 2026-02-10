import numpy as np
from matplotlib import pyplot as plt

from dimensionality_manuscript.simulations import (
    RotatedEigenbasisGenerator,
    geometric_mean_spd,
    root_sandwich,
    plot_ellipse,
)

plt.rcParams["font.size"] = 14


# Create generator with rotated eigenbasis
offset_ratio = 2
evals = np.array([1, 0.1], dtype=np.float32)
generator = RotatedEigenbasisGenerator(offset_ratio=offset_ratio, evals1=evals)

# Get covariance matrices
C1 = generator.expected_covariance(which=1)
C2 = generator.expected_covariance(which=2)

# Get main directions from the generator's eigenbases
main_dir1 = generator.Q1[:, 0]
main_dir2 = generator.Q2[:, 0]

# Compute cross covariances and geometric means
C1root = root_sandwich(C1, C2)
C2root = root_sandwich(C2, C1)

cross_cov1 = C1root @ C2 @ C1root
cross_cov2 = C2root @ C1 @ C2root
G1 = geometric_mean_spd(C1, C2)
G2 = geometric_mean_spd(C2, C1)

evals_cross1, evecs_cross1 = np.linalg.eigh(cross_cov1)
evals_cross2, evecs_cross2 = np.linalg.eigh(cross_cov2)
evals_G1, evecs_G1 = np.linalg.eigh(G1)
evals_G2, evecs_G2 = np.linalg.eigh(G2)

evals_cross1 = np.sqrt(evals_cross1)
evals_cross2 = np.sqrt(evals_cross2)

# This to make sure they point in the same direction (sign doesn't matter anyway)
evecs_cross1 = evecs_cross1 * np.sign(evecs_cross1[0])
evecs_cross2 = evecs_cross2 * np.sign(evecs_cross2[0])
evecs_G1 = evecs_G1 * np.sign(evecs_G1[0])
evecs_G2 = evecs_G2 * np.sign(evecs_G2[0])

plt.close("all")
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
origin = [0], [0]  # origin point
ax[0].quiver(*origin, main_dir1[0], main_dir1[1], angles="xy", scale_units="xy", scale=1, color="r")
ax[0].quiver(*origin, main_dir2[0], main_dir2[1], angles="xy", scale_units="xy", scale=1, color="b")
ax[0].quiver(*origin, evecs_cross1[0, 1], evecs_cross1[1, 1], angles="xy", scale_units="xy", scale=1, color="darkred")
ax[0].quiver(*origin, evecs_cross2[0, 1], evecs_cross2[1, 1], angles="xy", scale_units="xy", scale=1, color="darkblue")
ax[0].set_xlim(-0.2, 1)
ax[0].set_ylim(-0.2, 1)
ax[0].set_aspect("equal")
ax[0].set_title("Quiver plot of main_dir1")
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")
ax[0].grid(True)

ax[1].plot(evals, marker="o", color="k")
ax[1].plot(0.1 / 3 + np.flipud(evals_cross1), marker="o", color="darkred")
ax[1].plot(0.2 / 3 + np.flipud(evals_cross2), marker="o", color="darkblue")
ax[1].set_xlabel("Component")
ax[1].set_ylabel("Eigenvalue")
ax[1].grid(True)
plt.show()

# Generate sample data
num_samples = 1000
data1 = generator.generate(num_samples, which=1)
data2 = generator.generate(num_samples, which=2)

# Compute covariance matrices from the data
C1_data = np.cov(data1)
C2_data = np.cov(data2)

# Eigendecompositions for plotting ellipses
evals1_data, evecs1_data = np.linalg.eigh(C1_data)
evals2_data, evecs2_data = np.linalg.eigh(C2_data)

# Compute root sandwich operators
C1root = root_sandwich(C1_data, C2_data)
C2root = root_sandwich(C2_data, C1_data)
cross_cov1 = C1root @ C2_data @ C1root
cross_cov2 = C2root @ C1_data @ C2root
evals_cross1, evecs_cross1 = np.linalg.eigh(cross_cov1)
evals_cross2, evecs_cross2 = np.linalg.eigh(cross_cov2)
evals_cross1 = np.sqrt(evals_cross1)
evals_cross2 = np.sqrt(evals_cross2)

# Compute geometric means
G1 = geometric_mean_spd(C1_data, C2_data)
G2 = geometric_mean_spd(C2_data, C1_data)
evals_G1, evecs_G1 = np.linalg.eigh(G1)
evals_G2, evecs_G2 = np.linalg.eigh(G2)

# Create 1x3 subplot layout
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

# Column 1: Both data samples with their ellipses
ax = axes[0]
ax.scatter(data1[0], data1[1], alpha=0.3, s=10, color="r")
ax.scatter(data2[0], data2[1], alpha=0.3, s=10, color="b")
plot_ellipse(ax, evals1_data, evecs1_data, mean=(0, 0), r=2, color="r", linewidth=2, label="C1 Data")
plot_ellipse(ax, evals2_data, evecs2_data, mean=(0, 0), r=2, color="b", linewidth=2, label="C2 Data")
ax.set_title("Data Samples with Ellipses")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend(fontsize=12)
ax.set_aspect("equal")

# Column 2: Root sandwich operators
ax = axes[1]
plot_ellipse(ax, evals1_data, evecs1_data, mean=(0, 0), r=2, color="r", linewidth=2, alpha=0.5, label="C1 Data")
plot_ellipse(ax, evals2_data, evecs2_data, mean=(0, 0), r=2, color="b", linewidth=2, alpha=0.5, label="C2 Data")
plot_ellipse(ax, evals_cross1, evecs_cross1, mean=(0, 0), r=2, color="darkred", linewidth=2, label=r"$C_1^{1/2}C_2C_1^{1/2}$")
plot_ellipse(ax, evals_cross2, evecs_cross2, mean=(0, 0), r=2, color="darkblue", linewidth=2, label=r"$C_2^{1/2}C_1C_2^{1/2}$")
ax.set_title("Root Sandwich Operators")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend(fontsize=12)
ax.set_aspect("equal")

# Column 3: Geometric means
ax = axes[2]
plot_ellipse(ax, evals1_data, evecs1_data, mean=(0, 0), r=2, color="r", linewidth=2, alpha=0.5, label="C1 Data")
plot_ellipse(ax, evals2_data, evecs2_data, mean=(0, 0), r=2, color="b", linewidth=2, alpha=0.5, label="C2 Data")
plot_ellipse(ax, evals_G1, evecs_G1, mean=(0, 0), r=2, color="green", linewidth=2, label=r"$C_1 \# C_2$")
plot_ellipse(ax, evals_G2, evecs_G2, mean=(0, 0), r=2, color="darkgreen", linewidth=2, label=r"$C_2 \# C_1$")
ax.set_title("Geometric Means")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend(fontsize=12)
ax.set_aspect("equal")

plt.tight_layout()
plt.show()
