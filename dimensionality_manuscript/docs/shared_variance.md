# Shared Variance Exposition

Suppose we have a population of neurons responding to some stimulus of interest while also being influenced by stimulus-independent "nuisance" signals and independent noise. We want to measure how much the *stimulus-dependent* covariance structure contributes to the *overall* covariance structure of population activity. To formalize this goal, let's define a generative model for neural activity. 

Let: 
$$x_t = g(s_t) + h(n_t) + \varepsilon_t \quad \text{where} \; x_t \in \mathbb{R}^{N}$$

$g(s_t)$ is a stimulus-dependent population vector, $h(n_t)$ is a stimulus-independent population vector representing the response to nuisance signals, and $\varepsilon_t$ is independent neural noise with diagonal covariance. Assume that $s_t$ and $n_t$ are independent random variables, and that $g(\cdot)$ and $h(\cdot)$ are deterministic functions.

From this model, we can define the following objects:

$$\begin{aligned}
\Sigma_{\text{stim}} &= \operatorname{cov}_s(g(s_t)) \\
\Sigma_{\text{nuisance}} &= \operatorname{cov}_n(h(n_t)) \\
\Sigma_{\text{noise}} &= \operatorname{cov}(\varepsilon_t) \\
\Sigma_{\text{full}} &= \operatorname{cov}_t(x_t) \\
\Sigma_{\text{full}} &= \Sigma_{\text{stim}} + \Sigma_{\text{nuisance}} + \Sigma_{\text{noise}} \\
\end{aligned}$$

We are curious how much the geometry of $\Sigma_{\text{stim}}$ contributes to the covariance structure of $\Sigma_{\text{full}}$. In particular, we want a metric that fulfills the properties defined in the next section.

## Measurement goal

Let $A$ denote a candidate explanatory covariance matrix, such as $\Sigma_{\mathrm{stim}}$, and let $B$ denote a target covariance matrix, such as $\Sigma_{\mathrm{full}}$. The metric should satisfy three properties.

1. **Joint geometric awareness.**
    The metric should depend on the geometry of both $A$ and $B$ and be *symmetric* to the ordering of $A$ and $B$.
2. **Reliability awareness.**
    The measure should focus on reproducible covariance structure in both the candidate and target matrices. In addition, it should admit a cross-validated form for unbiased estimation of shared geometry and unbiased estimation of stimulus covariance.
3. **Cross-condition comparability.**
    The measure should remain meaningful even if $A$ and $B$ are covariances from different conditions, where neither matrix is necessarily Loewner-bounded by the other (see below for explanation of Loewner ordering). When $A=\Sigma_{\text{stim}}$ and $B=\Sigma_{\text{full}}$, then $A$ represents a subset of variance in $B$. However, we may be interested in how much the variance of one stimulus type is shared with the variance of another, or how much the stimulus-evoked covariance is shared with spontaneous covariance.

---

## Shared variance overlap: $\kappa(A, B)$
To arrive at our metric, we must first define an operator that evaluates the overlap between two covariance matrices. For two positive semi-definite covariance matrices $A, B$, define:

$$\kappa(A, B) \triangleq \| A^{1/2} B^{1/2} \|_*$$

where $\|\cdot\|_*$ is the nuclear norm. 

We can express this in various equivalent ways, in terms of singular values ($\sigma_i$), eigenvalues ($\lambda_i$), or the trace:

$$\| A^{1/2} B^{1/2} \|_* = \sum_i \sigma_i(A^{1/2}B^{1/2}) = \sum_i \sqrt{\lambda_i(A^{1/2}BA^{1/2})} = \operatorname{tr} \sqrt{A^{1/2}BA^{1/2}}$$

To understand what $\kappa(A, B)$ represents intuitively, consider the geometric meaning of each mode of overlap in the singular value decomposition of $A^{1/2}B^{1/2}$. 

Let $\mathcal{E}_A \triangleq \{ A^{1/2}u : \|u\| = 1 \}$ and $\mathcal{E}_B \triangleq \{ B^{1/2}v : \|v\| = 1 \}$ be the ellipsoids spanned by the covariance structure of $A$ and $B$. 

Then the first mode, denoted $\kappa_1(A, B)$, solves the following optimization problem:

$$\kappa_1(A, B) = \max_{\|u\|=\|v\|=1} \langle A^{1/2}u,\, B^{1/2}v \rangle$$
and each subsequent $\kappa_i(A, B)$, $i > 1$, solves the same problem subject to orthogonality constraints on singular vectors $u, v$. This formulation demonstrates that $\kappa(A, B)$ can be viewed as an attempt to find the pairs of vectors on the ellipsoids spanned by $A$ and $B$ that have maximal inner product, subject to orthogonality constraints on the singular vectors $u$ and $v$. 


---


## Shared Variance Ratio

Equipped with the shared variance overlap operator $\kappa$, we can now define a metric for measuring the contribution of stimulus-dependent covariance to the overall covariance structure of population activity.

First, note that although we are interested in the geometry of $\Sigma_{\text{stim}}$, we don't have direct access to it directly because we only observe noisy samples of neural activity. We can define the empirical covariance of the average stimulus-evoked response as:

$$\tilde{\Sigma}_{\text{stim}} = \operatorname{cov}_s(\bar{x}_s), \quad \text{ where } \; \bar{x}_s = \frac{1}{m_s} \sum_{t: s_t = s} x_t$$

Then, assuming equal and large sample size for each stimulus such that the nuisance variables are balanced:
$$\begin{aligned}
\bar{x}_s &= g(s) + \frac{1}{m_s} \sum_{t: s_t = s} h(n_t) + \frac{1}{m_s} \sum_{t: s_t = s} \varepsilon_t \\
\mathbb{E}\left[\tilde{\Sigma}_{\text{stim}}\right] &\approx \Sigma_{\text{stim}} + \frac{1}{m_s} \Sigma_{\text{nuisance}} + \frac{1}{m_s} \Sigma_{\text{noise}}
\end{aligned}$$

Where $\mathbb{E}[\tilde{\Sigma}_{\text{stim}}]$ is the expected value of $\tilde{\Sigma}_{\text{stim}}$ across many repeats of stimuli and nuisance signals, assuming balanced sampling of stimuli and nuisance variables.

We partition the trials of neural activity into two disjoint sets denoted $\mathcal{F}_{\text{train}}$ and $\mathcal{F}_{\text{test}}$, then measure the following quantities:

$$\begin{aligned}
\bar{x}_s^{\text{train}} &= \mathbb{E}_{t \in \mathcal{F}_{\text{train}}}[x_t \mid s_t = s] \\
\tilde{\Sigma}_{\text{stim}}^{\text{train}} &= \operatorname{cov}_s\!\left(\bar{x}_s^{\text{train}}\right) \\
\tilde{\Sigma}_{\text{full}}^{\text{train}} &= \operatorname{cov}_t(x_t),\quad t \in \mathcal{F}_{\text{train}} \\
\tilde{\Sigma}_{\text{full}}^{\text{test}} &= \operatorname{cov}_t(x_t),\quad t \in \mathcal{F}_{\text{test}} \\
\end{aligned}$$

Then, we use the following formula to define the ***shared variance ratio***, denoted $\text{SVR}$, which measures the ratio of reproducible structure in $\Sigma_{\text{full}}$ that is shared with $\Sigma_{\text{stim}}$:

$$\text{SVR} = \frac{\kappa\!\left(\tilde{\Sigma}_{\text{stim}}^{\text{train}},\, \tilde{\Sigma}_{\text{full}}^{\text{test}}\right)}{\kappa\!\left(\tilde{\Sigma}_{\text{full}}^{\text{train}},\, \tilde{\Sigma}_{\text{full}}^{\text{test}}\right)}$$

The numerator measures how much the geometry of stimulus-dependent covariance in the training set is shared with the geometry of the full covariance in the test set. The denominator measures how much the geometry of the full covariance is shared across the training and test sets, which serves as a normalization factor that accounts for the reliability of the structure in $\Sigma_{\text{full}}$.

Therefore, the full ratio measures the ratio of reliable structure in $\Sigma_{\text{full}}$ that is shared with $\Sigma_{\text{stim}}$.

##### Comparison to Centered Kernel Alignment (CKA)
Note how this differs from CKA, which uses the following formula:

$$CKA(A, B) = \frac{\operatorname{tr}(A B)}{\sqrt{\operatorname{tr}(A^2) \operatorname{tr}(B^2)}}$$

In CKA, the denominator is a normalization factor that accounts for the total variance in *each* matrix, rather than accounting for repeatability in the structure of the target matrix. Additionally, CKA measures alignment independent of scale (it acts like the cosine similarity between the two matrices), while $\text{SVR}$ measures the ratio of reproducible structure in $B$ shared with $A$. 

#### The Stimulus-Space Representation
In our desiderata, we wanted a measure that is amenable to cross-validation. However, as presented, the current $\text{SVR}$ formula and $\kappa$ operator require the input variables to be PSD covariance matrices, which means we can't use a cross-validated operator for $\Sigma_{\text{stim}}$. (We need to be able to compute $A^{1/2}$ and $B^{1/2}$, why are not guaranteed to be real for non-symmetric matrices). Although the metric still identifies reliable variance by comparing across training and test sets, it is of interest to define a form amenable to cross-validation.  


Let $G_A \in \mathbb{R}^{N \times S}$ be a centered and scaled data matrix such that $G_A G_A^T = A$. For example, if $G_A$ is the (neurons x stimuli) matrix of average responses by stimulus (subject to centering and scaling by $1/\sqrt{S}$), then $G_A G_A^T = \tilde{\Sigma}_{\text{stim}}$. Note how $G_A$ acts like the square root of $A$ in the sense that when multiplied by itself (subject to a transpose) it produces $A$.

Define the stimulus-space kernel:

$$
K_B(A) \triangleq G_A^T B G_A
$$

This kernel has the same nonzero eigenvalues as $A^{1/2}BA^{1/2}$, which means we can measure $\kappa(A, B)$ by measuring the eigenvalues of $K_B(A)$ instead. To see this, let $X = B^{1/2} G_A$. Then:

$$
X^T X = G_A^T B G_A = K_B(A)
$$

and

$$
\begin{aligned}
X X^T &= B^{1/2} G_A G_A^T B^{1/2} \\
X X^T &= B^{1/2} A B^{1/2}
\end{aligned}
$$

Since $X^T X$ and $X X^T$ have the same nonzero eigenvalues, and $B^{1/2} A B^{1/2}$ and $A^{1/2} B A^{1/2}$ have the same eigenvalues, we have:

$$\lambda_i(K_B(A)) = \lambda_i(A^{1/2} B A^{1/2})$$

for all nonzero modes.

Therefore, 

$$
\begin{aligned}
\kappa_i(A, B) &= \sqrt{\lambda_i(K_B(A))} \\
\kappa(A, B) &= \operatorname{tr}\sqrt{K_B(A)} \\
\end{aligned}
$$


This formulation is especially useful for cross-validation: independent foldwise estimates of $G_A$ yield unbiased two-view estimators of the underlying stimulus-space kernel, even when the resulting finite-sample estimator is no longer symmetric.

#### Estimating $\kappa(\Sigma_{\text{stim}}, \Sigma_{\text{full}})$ without bias
In practice, we don't have direct access to $\Sigma_{\text{stim}}$, so we need to estimate it from data. As described above, although $\bar{x}_s$ is an unbiased estimator of $g(s)$, the empirical covariance $\tilde{\Sigma}_{\text{stim}} = \operatorname{cov}_s(\bar{x}_s)$ is a biased estimator of $\Sigma_{\text{stim}}$ because it includes noise from finite trial sampling. 

To mitigate this bias, we can use a cross-validated stimulus space estimator of $\kappa(\Sigma_{\text{stim}}, \Sigma_{\text{full}})$ that uses three independent splits of the data, $\mathcal{F}_1$, $\mathcal{F}_2$, and $\mathcal{F}_3$, to compute three independent estimates of the stimulus-evoked mean response for each stimulus. The first estimate will be used to compute directions, and the remaining estimates will be used for cross-validated measurement of amplitudes. We will use the stimulus space form as follows:

Divide the training trials into three repeats: $\mathcal{F}_{tr1}$, $\mathcal{F}_{tr2}$, and $\mathcal{F}_{tr3}$. Then, compute the stimulus-evoked mean response for each stimulus in each repeat:

$$
\bar{x}_s^{(k)} = \mathbb{E}_{t \in \mathcal{F}_{k}}[x_t \mid s_t = s] \\
\bar{X}^{(k)} = \begin{bmatrix} \bar{x}_1^{(k)} & \cdots & \bar{x}_S^{(k)} \end{bmatrix}
$$

Next, from each repeat, we can compute the "pre-covariance" matrix by centering and scaling the stimulus-evoked mean responses. Here, $\mathbf{1}_S$ is a vector of ones of length $S$, and the centering term $\frac{1}{S} \bar{X}^{(k)} \mathbf{1}_S \mathbf{1}_S^T$ removes the mean across stimuli from each neuron's response:

$$
G_{\text{stim}}^{(k)} = \frac{1}{\sqrt{S - 1}} \left( \bar{X}^{(k)} - \frac{1}{S} \bar{X}^{(k)} \mathbf{1}_S \mathbf{1}_S^T \right) \in \mathbb{R}^{N \times S}
$$

With this, we have $G_{\text{stim}}^{(k)} (G_{\text{stim}}^{(k)})^T = \tilde{\Sigma}_{\text{stim}}^{(k)}$, which is the empirical stimulus covariance computed from repeat $k$.

To estimate directions, we first compute the eigenvectors of $K_{\text{full}}(G_{\text{stim}}^{(1)}) = (G_{\text{stim}}^{(1)})^T \tilde{\Sigma}_{\text{full}}^{\text{test}} G_{\text{stim}}^{(1)}$ using only the first repeat. Then, we measure the covariance along those directions of using the second and third repeats. With $u_i$ denoting the $i$th eigenvector of $K_{\text{full}}(G_{\text{stim}}^{(1)})$, we can compute the $i$th mode of shared variance as follows:

$$
w_i^{cv} = (G_{\text{stim}}^{(2)} u_i)^T \tilde{\Sigma}_{\text{full}}^{\text{test}} (G_{\text{stim}}^{(3)} u_i)
$$

This is a valid, unbiased estimator of the covariance along the $i$th mode of overlap because $G_{\text{stim}}^{(2)}$ and $G_{\text{stim}}^{(3)}$ are independent estimates of the stimulus space, and $\tilde{\Sigma}_{\text{full}}^{\text{test}}$ is yet another independent estimate of the full covariance. However, it has the wrong scale as $\kappa$. Whereas $\kappa_i(A, B) = \sqrt{\lambda_i(K_B(A))}$, we have $\mathbb{E}[w_i^{cv}] = \lambda_i(K_B(A))$, but we can't take the square root of $w_i^{cv}$ because it can be negative. 

We can do one of the following: 

1. Instead of using the "amplitude" scale of $\kappa$, which estimates the singular values rather than eigenvalues, we can turn to this "energy" scale, which matches the bilinear form of our cross-validated estimator. In that case, the numerator of $\text{SVR}$ would be $\sum_i w_i^{cv}$, which is an unbiased estimator of $\operatorname{tr}(K_B(A)) = \operatorname{tr}(A B)$. The denominator of $\text{SVR}$ would be $\operatorname{tr}(\tilde{\Sigma}_{\text{full}}^{\text{train}} \tilde{\Sigma}_{\text{full}}^{\text{test}})$, which is an unbiased estimator of $\operatorname{tr}(\Sigma_{\text{full}}^2)$. This is closer in relation to CKA than the original definition of $\text{SVR}$. It measures the ratio of shared energy rather than the ratio of shared variance, and emphasize high-variance modes more than full distributions. 

2. Alternatively, we can make the argument that negative values of $w_i^{cv}$ are likely to be noise, and therefore we can threshold $w_i^{cv}$ at 0 then take the square root. Although this is biased, it permits the cross-validated form to estimate the same quantity as the original $\text{SVR}$. This form looks like this:
$$\text{SVR} = \frac{\sum_i \sqrt{\max(w_i^{cv}, 0)}}{\kappa\!\left(\tilde{\Sigma}_{\text{full}}^{\text{train}},\, \tilde{\Sigma}_{\text{full}}^{\text{test}}\right)}$$


#### Summary of 3 Key metrics
1. Shared Variance Ratio

$$\text{SVR}(A, B) = \frac{\kappa\!\left(A_{\text{train}}, B_{\text{test}}\right)}{\kappa\!\left(B_{\text{train}}, B_{\text{test}}\right)}$$

$$\kappa(A, B) = \| A^{1/2} B^{1/2} \|_*$$

- The shared variance ratio measures how much variance overlap there is between $A$ and $B$ relative to the reliability of the structure in $B$. 
- It compares $A$ and $B$ in train samples to $B$ in test samples, which makes it cross-validated, although it is not unbiased because the nuclear norm $\|\cdot\|_*$ is nonnegative.
- When $A=B$, then $\kappa(A, B) = \operatorname{tr}(A) = \operatorname{tr}(B)$ and $\text{SVR}(A, B) = 1$.
- $\text{SVR}(A, B)$ emphasizes the full dimensionality because it uses the ***variance scale*** of $A$ and $B$ by using the square root of the eigenvalues of a covariance matrix product (compare to the two metrics below).


2. Cross-validated Shared Energy Ratio

$$\text{cvSER}(A, B) = \frac{\operatorname{tr}(G_{12})}{\operatorname{tr}(B_{12} B_3)}$$

$$G_{ij} \triangleq f(A_i)^T B_3 f(A_j)$$

$$f(A) = \frac{1}{\sqrt{S - 1}} \left( A - \frac{1}{S} A \mathbf{1}_S \mathbf{1}_S^T \right)$$

- The cross-validated shared energy ratio measures how much variance overlap there is between $A$ and $B$ relative to the reliability of the structure in $B$.
- The numerator is a cross-validated estimator of $\operatorname{tr}(A B)$, which means we can use repeats of a stimulus presentation (or environment traversal) to reduce the contribution of nuisance signals to the estimation of stimulus-specific covariance.
- The $\text{cvSER}$ emphasizes high-variance modes because it uses the ***variance-squared scale*** of $A$ and $B$ by using the eigenvalues of a covariance matrix product. This means if several high-variance modes are shared, the $\text{cvSER}$ will be high, even if there are many lower-variance modes that disagree.

3. Centered Kernel Alignment (CKA)

$$CKA(A, B) = \frac{\operatorname{tr}(A_{\text{train}} B_{\text{test}})}{\sqrt{\operatorname{tr}(A_{\text{train}}^2) \operatorname{tr}(B_{\text{test}}^2)}}$$

- CKA measures how much variance overlap there is between $A$ and $B$ relative to the total variance in each matrix, without accounting for reliability of structure in either matrix.
- It compares $A$ in a train sample to $B$ in a test sample, which makes it cross-validated (in the numerator, not the denominator!), although it is not unbiased because the trace of covariance matrices is nonnegative.
- When $A=B$, then population $CKA(A, B) = 1$ (sample CKA will approach 1 as sample size increases).
- Like $\text{cvSER}$, CKA emphasizes high-variance modes because it uses the variance-squared scale of $A$ and $B$ by using the eigenvalues of a covariance matrix product without a square root. 
- CKA acts like a matrix inner product (it measures the cosine similarity between $A$ and $B$, with $\langle A, B \rangle = \operatorname{tr}(A B)$). Due to this property, CKA is always bounded above by $1$ and measures alignment (weighted by variance) independent of scale. 



#### Amplitude vs. energy perspectives
Above, we measure $\kappa(A, B) = \sum_i \kappa_i(A, B)$ to quantify the total shared variance across all overlap modes, in which we use the "amplitude" scale of $AB$ by taking the square root of the eigenvalues. In this case, when $A=B$, then $\kappa_i(A, A) = \lambda_i(A)$ which feels appropriate.

However, the cross-validated bilinear form is more amenable to the "energy" scale that does not use a square root. If instead we used $\omega(A, B) = \sum_i \lambda_i(K_B(A)) = \operatorname{tr}(AB)$, then we could use the cross-validated estimator without modification. In this case, when $A=B$, then $\kappa_i(A, A) = \lambda_i^2(A)$.

The amplitude perspective is more sensitive to the full distribution of variance across modes, whereas the energy perspective is more sensitive to any highly overlapping high-variance modes. Both perspectives are interesting.


#### The SVR estimates a true fraction for within condition comparisons
We have defined the shared variance ratio as follows:

$$
\text{SVR} = \frac{\kappa(A, C)}{\kappa(B, C)}
$$

In the case where matrix $A$ comes from a subset of the variance in $B$, we can establish that the population $\text{SVR}$ is bounded between 0 and 1, making it a true *fraction* of shared variance.

To prove this claim, we first must define the Loewner order on positive semi-definite (PSD) matrices. For two Hermitian matrices $A$ and $B$, we say that $A \preceq B$ if $B - A$ is PSD, which implies that $v^T A v \leq v^T B v$ for all vectors $v$, and therefore $\lambda_i(A) \leq \lambda_i(B)$ for all $i$, where $\lambda_i(\cdot)$ denotes the eigenvalues of the matrix. All covariance matrices are Hermitian. We use Loewner order notation to indicate that if a matrix $A$ is PSD, then $A \succeq 0$. 

Matrix congruence preserves Loewner order. If $A \preceq B$ and $C$ is any matrix, then $C^T A C \preceq C^T B C$. This is because $C^T B C - C^T A C = C^T (B - A) C$, and $C^T (B - A) C$ is PSD if $B - A$ is PSD. Let's assume that $A \preceq B$, $C \succeq 0$, and $C=C^T$. We can use the properties of Loewner order to show that $\text{SVR}$ is bounded between 0 and 1.

We can express $\kappa(A, C)$ in terms of the eigenvalues of $A^{1/2}CA^{1/2}$ or $C^{1/2}AC^{1/2}$, which are the same:

$$\begin{aligned}
\kappa(A, C) &= \sum_i \sqrt{\lambda_i(A^{1/2}CA^{1/2})} = \sum_i \sqrt{\lambda_i(C^{1/2}AC^{1/2})} \\
\kappa(B, C) &= \sum_i \sqrt{\lambda_i(B^{1/2}CB^{1/2})} = \sum_i \sqrt{\lambda_i(C^{1/2}BC^{1/2})}
\end{aligned}$$

Then, using the rightmost notation, where $C$ wraps $A$ or $B$, we have:

$$\begin{aligned}
A &\preceq B \\
\implies C^{1/2} A C^{1/2} &\preceq C^{1/2} B C^{1/2} \\
\implies \lambda_i(C^{1/2} A C^{1/2}) &\leq \lambda_i(C^{1/2} B C^{1/2}) \quad \forall i \\
\implies \kappa(A, C) &\leq \kappa(B, C)
\end{aligned}$$

In our specific case, we have:

$$\begin{aligned}
A &= \Sigma_{\text{stim}}^{\text{train}} \\
B &= \Sigma_{\text{full}}^{\text{train}} \\
C &= \Sigma_{\text{full}}^{\text{test}}
\end{aligned}$$

We know from our generative model that $\Sigma_{\text{stim}} \preceq \Sigma_{\text{full}}$ because $\Sigma_{\text{full}} = \Sigma_{\text{stim}} + \Sigma_{\text{nuisance}} + \Sigma_{\text{noise}}$ and $\Sigma_{\text{nuisance}} + \Sigma_{\text{noise}} \succeq 0$. Therefore, 

$$0 \leq \frac{\kappa\!\left(\Sigma_{\text{stim}}^{\text{train}},\, \Sigma_{\text{full}}^{\text{test}}\right)}{\kappa\!\left(\Sigma_{\text{full}}^{\text{train}},\, \Sigma_{\text{full}}^{\text{test}}\right)} \leq 1.$$


#### The empirical SVR is an upward-biased proxy for the latent SVR
***NOTE THIS SECTION IS WEAK - PROBABLY NEEDS JENSEN INEQUALITY ANALYSIS FOR THE FINITE-SAMPLE CASE***
Note that although we are interested in $\Sigma_{\text{stim}}$, we only have access to the empirical estimate $\tilde{\Sigma}_{\text{stim}}$. Using the definition of $\mathbb{E}[\tilde{\Sigma}_{\text{stim}}]$ from above, we have:

$$
\Sigma_{\text{stim}} \preceq \mathbb{E}[\tilde{\Sigma}_{\text{stim}}]
$$

Which means:

$$0 \leq \frac{\kappa\!\left(\Sigma_{\text{stim}}^{\text{train}},\, \Sigma_{\text{full}}^{\text{test}}\right)}{\kappa\!\left(\Sigma_{\text{full}}^{\text{train}},\, \Sigma_{\text{full}}^{\text{test}}\right)} \leq \frac{\kappa\!\left(\mathbb{E}[\tilde{\Sigma}_{\text{stim}}]^{\text{train}},\, \Sigma_{\text{full}}^{\text{test}}\right)}{\kappa\!\left(\Sigma_{\text{full}}^{\text{train}},\, \Sigma_{\text{full}}^{\text{test}}\right)} \leq 1.$$

This means that the expected value of $\text{SVR}$, which uses $\tilde{\Sigma}_{\text{stim}}$ in the numerator, is an upper bound of the latent $\text{SVR}$. The real case, where $\tilde{\Sigma}_{\text{stim}}$ is a finite-sample, rather than it's expected value is more complicated, but follows a similar trend. Using the law of total covariance for the empirical covariance matrix, we have:

$$\operatorname{cov}_t(x_t)=\operatorname{cov}_s(\bar{x}_s) + \mathbb{E}_s[\operatorname{cov}_t(x_t \mid s_t = s)]$$

which implies that $\tilde{\Sigma}_{\text{stim}} \preceq \tilde{\Sigma}_{\text{full}}$.

Therefore, the empirically measured $\text{SVR}$ also has an upper bound of 1, but it may be above or below the latent $\text{SVR}$ depending on the structure of the data and the sample size. 
