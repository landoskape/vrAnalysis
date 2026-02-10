# Covariance Generator Transforms

This document describes all available transform methods for `CovarianceGenerator.variant()`.

## Spectrum-Only Transforms

These transforms modify eigenvalues while keeping eigenvectors fixed.

### `scale_spectrum`

Multiply all eigenvalues by a constant factor.

**Parameters:**
- `scale` (float, required): Scaling factor to apply to all eigenvalues

**Example:**
```python
gen2 = gen.variant("scale_spectrum", scale=2.0)
```

---

### `band_scale_spectrum`

Scale head, middle, and tail eigenvalues differently.

**Parameters:**
- `k_head` (int, required): Number of eigenvalues in the head (largest)
- `k_tail` (int, required): Number of eigenvalues in the tail (smallest)
- `scale_head` (float, required): Scaling factor for head eigenvalues
- `scale_mid` (float, required): Scaling factor for middle eigenvalues
- `scale_tail` (float, required): Scaling factor for tail eigenvalues

**Example:**
```python
gen2 = gen.variant("band_scale_spectrum", k_head=5, k_tail=10, 
                   scale_head=2.0, scale_mid=1.0, scale_tail=0.5)
```

---

### `powerlaw_resample`

Replace eigenvalues with a new power-law spectrum.

**Parameters:**
- `alpha_new` (float, required): New power-law exponent
- `normalize_total_variance` (bool, default=False): If True, scale eigenvalues to preserve total variance

**Example:**
```python
gen2 = gen.variant("powerlaw_resample", alpha_new=1.5, normalize_total_variance=True)
```

---

### `permute_eigenvalues`

Shuffle eigenvalues while keeping eigenvectors fixed.

**Parameters:**
- `mode` (str, default="full"): Permutation mode
  - `"full"`: Permute all eigenvalues
  - `"partial"`: Permute only a subset of eigenvalues
- `k` (int, optional): For "partial" mode, number of eigenvalues to permute
- `frac` (float, optional): Alternative to k: fraction of eigenvalues to permute (0-1)
- `where` (str, default="head"): For "partial" mode, where to select eigenvalues
  - `"head"`: Select from the head (largest)
  - `"tail"`: Select from the tail (smallest)
  - `"random"`: Select randomly

**Examples:**
```python
# Permute all eigenvalues
gen2 = gen.variant("permute_eigenvalues", mode="full")

# Permute top 10 eigenvalues
gen2 = gen.variant("permute_eigenvalues", mode="partial", k=10, where="head")

# Permute random 20% of eigenvalues
gen2 = gen.variant("permute_eigenvalues", mode="partial", frac=0.2, where="random")
```

---

## Eigenvector-Only Transforms

These transforms modify eigenvectors while keeping eigenvalues fixed.

### `randomize_eigenvectors`

Replace all eigenvectors with a new random orthonormal basis.

**Parameters:** None

**Example:**
```python
gen2 = gen.variant("randomize_eigenvectors")
```

---

### `rotate_plane`

Rotate eigenvectors in a selected 2-D subspace by angle θ.

**Parameters:**
- `i` (int, required): Index of first eigenvector (0-indexed)
- `j` (int, required): Index of second eigenvector (0-indexed)
- `theta` (float, required): Rotation angle in radians

**Example:**
```python
gen2 = gen.variant("rotate_plane", i=0, j=1, theta=np.pi/4)
```

---

### `rotate_block`

Apply a random or structured rotation within k eigenvectors.

**Parameters:**
- `k` (int, required): Number of eigenvectors to rotate (must be ≥ 2)
- `where` (str, default="top"): Which eigenvectors to rotate
  - `"top"`: Rotate top k eigenvectors (largest eigenvalues)
  - `"tail"`: Rotate tail k eigenvectors (smallest eigenvalues)
- `strength` (float, optional): Strength of rotation (0-1). If provided, uses structured rotation
- `theta` (float, optional): Explicit rotation angle. If provided, applies same rotation to all pairs

**Examples:**
```python
# Rotate top 5 eigenvectors
gen2 = gen.variant("rotate_block", k=5, strength=0.5)

# Rotate tail 3 eigenvectors
gen2 = gen.variant("rotate_block", k=3, where="tail", strength=0.8)
```

---

### `perturb_eigenvectors`

Add small random perturbations to eigenvectors and re-orthonormalize.

**Parameters:**
- `epsilon` (float, default=0.1): Scale of the random noise to add
- `where` (str, default="all"): Which eigenvectors to perturb
  - `"all"`: Perturb all eigenvectors
  - `"topk"`: Perturb only top k eigenvectors (requires `k` parameter)

**Example:**
```python
gen2 = gen.variant("perturb_eigenvectors", epsilon=0.05, where="topk", k=10)
```

---

## Mixed Transforms

These transforms modify both eigenvectors and eigenvalues.

### `shared_subspace_rotated`

Share span of top-k eigenvectors but rotate within subspace.

**Parameters:**
- `k` (int, required): Number of top eigenvectors to preserve span of
- `rotation_strength` (float, required): Strength of rotation within subspace (0-1)

**Example:**
```python
gen2 = gen.variant("shared_subspace_rotated", k=5, rotation_strength=0.3)
```

---

### `shared_subset_full_match`

Preserve selected eigenpairs (eigenvectors + eigenvalues); randomize others.

**Parameters:**
- `indices` (array-like, optional): Explicit indices of eigenpairs to preserve
- `k` (int, optional): Number of eigenpairs to preserve (used with `where`)
- `where` (str, default="head"): Where to preserve (used with `k`)
  - `"head"`: Preserve first k eigenpairs
  - `"tail"`: Preserve last k eigenpairs
  - `"random"`: Preserve k random eigenpairs

**Examples:**
```python
# Preserve first 5 eigenpairs
gen2 = gen.variant("shared_subset_full_match", k=5, where="head")

# Preserve specific eigenpairs
gen2 = gen.variant("shared_subset_full_match", indices=[0, 2, 4])
```

---

## Additive Transforms

These transforms add structure to the covariance matrix and recompute the eigendecomposition.

### `add_isotropic_noise`

Add σ²I to covariance (noise floor change).

**Parameters:**
- `sigma2` (float, required): Variance of isotropic noise to add

**Example:**
```python
gen2 = gen.variant("add_isotropic_noise", sigma2=0.1)
```

---

### `add_diagonal_noise`

Add neuron-specific variance (heteroskedastic noise).

**Parameters:**
- `scale` (float, required): Scale of diagonal noise
- `distribution` (str, default="uniform"): Distribution for noise
  - `"uniform"`: Uniform distribution
  - `"exponential"`: Exponential distribution
  - `"gamma"`: Gamma distribution

**Example:**
```python
gen2 = gen.variant("add_diagonal_noise", scale=0.1, distribution="exponential")
```

---

### `add_rank1_mode`

Add a new low-rank covariance component vvᵀ.

**Parameters:**
- `strength` (float, required): Strength of the rank-1 component
- `alignment` (str, default="random"): Alignment mode
  - `"aligned"`: Align with first eigenvector
  - `"orthogonal"`: Orthogonal to first eigenvector
  - `"random"`: Random direction

**Example:**
```python
gen2 = gen.variant("add_rank1_mode", strength=1.0, alignment="orthogonal")
```

---

## Summary

**Core transforms (12 total):**

**Spectrum-only (4):**
- `scale_spectrum` - uniform scaling
- `band_scale_spectrum` - band-specific scaling
- `powerlaw_resample` - replace with new power-law
- `permute_eigenvalues` - shuffle (full or partial)

**Eigenvector-only (4):**
- `randomize_eigenvectors` - replace all
- `rotate_plane` - 2D rotation between two indices
- `rotate_block` - rotation within top-k
- `perturb_eigenvectors` - small random perturbations

**Mixed (2):**
- `shared_subspace_rotated` - preserve span, rotate within
- `shared_subset_full_match` - preserve selected eigenpairs

**Additive (3):**
- `add_isotropic_noise` - uniform noise floor
- `add_diagonal_noise` - heteroskedastic noise
- `add_rank1_mode` - add rank-1 component
