# Low-Rank Gain Modulation as an Analogue of Residual SVD

## Existing Additive Model

Suppose we have:

\[
z_{ct}
\]

observed activity for neuron \(c\) at time \(t\), and a place-field prediction

\[
f_{ct} = f_c(x_t).
\]

The current model decomposes activity into:

\[
z_{ct}
=
f_{ct}
+
\sum_{k=1}^{r} u_{ck}s_{kt}.
\]

In matrix notation:

\[
Z = F + US^\top.
\]

where:

- \(F\) is the place-field prediction matrix
- \(U \in \mathbb{R}^{N \times r}\) contains neuron loadings
- \(S \in \mathbb{R}^{T \times r}\) contains latent shared fluctuations

### Training

Compute residuals:

\[
R = Z - F.
\]

Then perform SVD:

\[
R \approx U_r \Sigma_r V_r^\top.
\]

This gives the optimal rank-\(r\) approximation.

### Testing

Given source neurons:

\[
r_t^{(src)}
=
z_t^{(src)}
-
f_t^{(src)}
\]

infer latent state

\[
s_t
=
U_{src}^{\top} r_t^{(src)}
\]

(or least-squares variant).

Predict targets:

\[
\hat z_t^{(tgt)}
=
f_t^{(tgt)}
+
U_{tgt}s_t.
\]

This is elegant because everything reduces to standard SVD.

---

# Rank-1 Gain Modulation

A simple gain model is

\[
z_{ct}
=
g_t f_{ct}.
\]

Equivalently,

\[
z_{ct}
=
f_{ct}
+
(g_t-1)f_{ct}.
\]

The residual scales with place-field strength.

When a neuron is inactive according to the place field, gain modulation has little effect.

---

# Multi-Rank Gain Modulation

The natural generalization is

\[
z_{ct}
=
f_{ct}
\left(
1+\sum_{k=1}^{r}u_{ck}s_{kt}
\right).
\]

In vector form:

\[
z_t
=
f_t
\odot
(1 + Us_t).
\]

where \(\odot\) denotes elementwise multiplication.

---

# The Key diag() Observation

Using

\[
a\odot b
=
\operatorname{diag}(a)b,
\]

we can rewrite:

\[
z_t
=
f_t
+
\operatorname{diag}(f_t)Us_t.
\]

Therefore residuals become

\[
r_t
=
z_t-f_t
=
\operatorname{diag}(f_t)Us_t.
\]

This is exactly the gain model expressed as an additive residual model.

---

# Interpretation

The additive model assumes:

\[
r_t = Us_t.
\]

The gain model assumes:

\[
r_t
=
\operatorname{diag}(f_t)Us_t.
\]

The latent state remains low-dimensional.

The neuron loadings remain low-dimensional.

However the effective loading matrix becomes:

\[
U_{\mathrm{eff}}(t)
=
\operatorname{diag}(f_t)U.
\]

Thus:

- Same latent dimensions everywhere
- Same neuron loadings everywhere
- Effect size scales according to current place-field drive

This is attractive biologically because neurons only express gain modulation when their place field is active.

---

# Why Ordinary SVD No Longer Works

The additive model gives

\[
R = US^\top,
\]

which is low-rank.

The gain model gives

\[
R
=
F \odot (US^\top).
\]

Elementwise multiplication by \(F\) destroys the standard low-rank structure.

Therefore ordinary SVD is no longer the optimal solution.

---

# Approach 1: SVD in Relative-Gain Space

Observe that

\[
r_{ct}
=
f_{ct}
\sum_k u_{ck}s_{kt}.
\]

Dividing by the place-field prediction gives

\[
\tilde R_{ct}
=
\frac{r_{ct}}{f_{ct}}
=
\sum_k u_{ck}s_{kt}.
\]

Now

\[
\tilde R
=
US^\top.
\]

This is again a standard low-rank factorization.

Procedure:

1. Compute residuals

\[
R = Z-F.
\]

2. Compute relative residuals

\[
\tilde R
=
\frac{R}{F}.
\]

(elementwise division)

3. Perform SVD

\[
\tilde R
\approx
US^\top.
\]

4. Reconstruct

\[
\hat Z
=
F + F\odot(US^\top).
\]

### Numerical Issue

When

\[
f_{ct}\approx 0,
\]

the division becomes unstable.

This is particularly problematic in hippocampus because most neurons are outside their place fields most of the time.

---

# Stabilized Relative-Gain Factorization

A simple fix is

\[
\tilde R_{ct}
=
\frac{R_{ct}}
     {f_{ct}+\lambda}.
\]

Then

\[
R_{ct}
\approx
(f_{ct}+\lambda)
\sum_k u_{ck}s_{kt}.
\]

For small \(\lambda\), this behaves similarly to the gain model while remaining numerically stable.

---

# Approach 2: Direct Low-Rank Gain Factorization

Rather than dividing, fit the model directly:

\[
R
\approx
F\odot(US^\top).
\]

Minimize:

\[
L
=
\sum_{ct}
\left(
R_{ct}
-
F_{ct}(US^\top)_{ct}
\right)^2.
\]

This is a weighted matrix factorization problem.

A simple algorithm:

1. Initialize \(U,S\) using the stabilized division trick.
2. Fix \(U\), solve for \(S\).
3. Fix \(S\), solve for \(U\).
4. Repeat until convergence.

This is essentially alternating least squares.

---

# Decoding Latent State from Source Neurons

For additive SVD:

\[
r_t
=
Us_t.
\]

and decoding is

\[
s_t
=
(U^\top U)^{-1}U^\top r_t.
\]

For gain modulation:

\[
r_t
=
\operatorname{diag}(f_t)Us_t.
\]

Define

\[
D_t
=
\operatorname{diag}(f_t).
\]

Then

\[
r_t
=
D_tUs_t.
\]

The least-squares estimate is

\[
s_t
=
(U^\top D_t^2 U)^{-1}
U^\top D_t r_t.
\]

Importantly, the decoder changes with position.

Neurons with stronger predicted place-field activity contribute more strongly to inference of the latent state.

---

# Conceptual Summary

The additive model is:

\[
z_t
=
f_t
+
Us_t.
\]

The gain model is:

\[
z_t
=
f_t
+
\operatorname{diag}(f_t)Us_t.
\]

The only change is replacing the constant loading matrix \(U\) with a position-dependent loading matrix:

\[
U
\rightarrow
\operatorname{diag}(f_t)U.
\]

This yields a true multidimensional gain-modulation model:

- shared latent states \(s_t\)
- neuron-specific gain loadings \(U\)
- modulation strength proportional to place-field drive

while remaining very close in spirit to the original residual-SVD framework.