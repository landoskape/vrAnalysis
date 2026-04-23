# Shared Variance Exposition

Let $x_t = g(s_t) + h_t + \varepsilon_t$, where $x_t \in \mathbb{R}^{N}$.

$g(s_t)$ is a stimulus-dependent deterministic population vector, $h_t$ is a stimulus-*independent* population vector sample, and $\varepsilon_t$ is independent neural noise.

Define $\bar{g}_s = \mathbb{E}_t[g(s_t) \mid s_t = s]$ as the mean stimulus-evoked response for stimulus $s$. Then:

$$
\begin{aligned}
\Sigma_{\text{stim}}  &= \operatorname{Cov}_s(\bar{g}_s) = \mathbb{E}_s\!\left[(\bar{g}_s - \mu)(\bar{g}_s - \mu)^T\right], \quad \mu = \mathbb{E}_s[\bar{g}_s] \\
\Sigma_{\text{full}}  &= \Sigma_{\text{stim}} + \Sigma_{\text{non-stim}} + \Sigma_{\text{noise}} \\
\Sigma_{\text{spont}} &= \Sigma_{\text{non-stim}} + \Sigma_{\text{noise}}
\end{aligned}
$$

We are curious how much the geometry of $\Sigma_{\text{stim}}$ contributes to the covariance structure of $\Sigma_{\text{full}}$ or $\Sigma_{\text{spont}}$. 

---

## Simple starting point: PCA
Our assumptions lead to a natural PSD matrix ordering, with: $\Sigma_{\text{stim}} \preceq \Sigma_{\text{full}}$ (this is because $\Sigma_{\text{full}} = \Sigma_{\text{stim}} + \Sigma_{\text{rest}}$, with $\Sigma_{\text{rest}} \succeq 0$, such that $\lambda_{\text{full}}^{\{i\}} \geq \lambda_{\text{stim}}^{\{i\}} \forall i$. Therefore, we could simply measure the eigenvalues of $\Sigma_{\text{stim}}$ and $\Sigma_{\text{full}}$, then compare them either modewise or by summing across the full spectrum. 

However, this has several drawbacks. 

1. We are interested in how much the ***geometry*** of the full population covariance lies in directions where the stimulus covariance ***also*** has covariance. PCA only compares variance magnitudes but ignores with high-variance directions in $\Sigma_{\text{full}}$ align with those of $\Sigma_{\text{stim}}$.
2. It does not account for reliability. We simply record the eigenvalues of a sample of the stimulus average matrix and compare with the full data matrix without consideration of whether the variance structure is ***repeatable***. 
3. It fails to be meaningful in cross-condition contexts - although direct variance comparisons work when $\Sigma_{\text{test}} \preceq \Sigma_{\text{ref}}$, it doesn't work when we want to compare across conditions, like $\Sigma_{\text{stim}}$ vs $\Sigma_{\text{full}}$. 

---

## An alternative: shared variance overlap
For $A, B \succeq 0$, define:
$$\kappa_i(A, B) \triangleq \sqrt{\lambda_i(A^{1/2}BA^{1/2})} = \sigma_i(A^{1/2}B^{1/2})$$

In words, $\kappa_i$ measures how much variance is shared between $A$ and $B$ along their $i$-th overlap mode. The singular value version $\kappa_i(\cdot) = \sigma_i(\cdot)$ has the most intuitive geometric meaning: 

Let $\mathcal{E}_A \triangleq \{ A^{1/2}u : \|u\| = 1 \}$ and $\mathcal{E}_B \triangleq \{ B^{1/2}v : \|v\| = 1 \}$ be the ellipsoids spanned by the covariance structure of $A$ and $B$. 

Then $\kappa_1(A, B)$ solves:
$$\kappa_1(A, B) = \max_{\|u\|=\|v\|=1} \langle A^{1/2}u,\, B^{1/2}v \rangle$$
and each subsequent $\kappa_i(A, B)$, $i > 1$, solves the same problem subject to orthogonality constraints on singular vectors $u, v$.

To compare full variance overlap, we simply take the sum of $\kappa_i$ across each mode:

$$\kappa(A, B) = \sum_i \kappa_i(A, B)$$

Which is related to the trace of $AB$ as follows:

$$\sum_i \kappa_i^2(A, B) = \sum_i \lambda_i(AB) = \operatorname{tr}(AB)$$


#### The Stimulus-Space Representation
The overlap measure $\kappa(A, B)$ is only defined for PSD matrices and depends on the spectrum of $A^{1/2}BA^{1/2}$. However, when either $A$ or $B$ arises from a stimulus-evoked pattern, we can formulate $\kappa$ in a stimulus-space representation that is amenable to cross-validation.

Let $G_A \in \mathbb{R}^{N \times S}$ be a centered and scaled data matrix such that $G_A G_A^T = A$. For example, if $G_A$ is the (neurons x stimuli) matrix of average responses by stimulus (subject to centering and scaling by $1/\sqrt{S}$), then $G_A G_A^T = \Sigma_{\text{stim}}$. Note how $G_A$ acts like the square root of $A$ in the sense that when multiplied by itself (subject to a transpose) it produces $A$.

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

#### Why shared variance overlap? 
1. $\kappa$ explicitly measures the how the ***geometry*** of one matrix relates to another. 
2. $\kappa(A, B)$ can be scaled by $\kappa(B, B')$ to account for ***repeatability*** of the structure in $B$. 
3. $B$ and $A$ need not maintain Loewner Order ($B \succeq $A), so we can compare across conditions, like $A=\Sigma_{\text{stim}}$ vs $B=\Sigma_{\text{spont}}$ where $\Sigma_{\text{spont}}$ is measured during a period without stimulus presentation (or it could be recorded for a different kind of stimulus, or behavior state, etc.).

#### Case 1: Shared Variance *Fraction*
Suppose we want to measure how much the geometry of stimulus-dependent covariance contributes to the full covariance structure of a population. In this case, we want to know how much $\Sigma_{\text{stim}}$ contributes to the structure of $\Sigma_{\text{full}}$, where $\Sigma_{\text{full}} = \Sigma_{\text{stim}} + \Sigma_{\text{rest}}$.

We partition trials into three disjoint sets $\mathcal{F}_1, \mathcal{F}_2, \mathcal{F}_3$, and define per-fold quantities:

$$\begin{aligned}
\bar{g}_s^{(k)} &= \mathbb{E}_{t \in \mathcal{F}_k}[x_t \mid s_t = s] \\
\Sigma_{\text{full}}^{(k)} &= \operatorname{Cov}_t(x_t),\quad t \in \mathcal{F}_k \\
\Sigma_{\text{full}}^{(1,2)} &= \operatorname{Cov}_t(x_t),\quad t \in \mathcal{F}_1 \cup \mathcal{F}_2 \\
\tilde{\Sigma}_{\text{stim}}^{(1,2)} &= \operatorname{Cov}_s\!\left(\bar{g}_s^{(1,2)}\right), \quad \bar{g}_s^{(1,2)} = \mathbb{E}_{t \in \mathcal{F}_1 \cup \mathcal{F}_2}[x_t \mid s_t = s]
\end{aligned}$$

We use the following formula for the shared variance fraction:

$$\text{SVF} = \frac{\kappa\!\left(\tilde{\Sigma}_{\text{stim}}^{(1,2)},\, \Sigma_{\text{full}}^{(3)}\right)}{\kappa\!\left(\Sigma_{\text{full}}^{(1,2)},\, \Sigma_{\text{full}}^{(3)}\right)}$$

##### Lemma (Bound on Shared Variance Fraction):

We can show that the $\text{SVF}$ is bounded between 0 and 1, making it a true fraction, such that: 
$$0 \leq \text{SVF} \leq 1$$

We know from our assumptions that $\tilde{\Sigma}_{\text{stim}}^{(1,2)} \preceq \Sigma_{\text{full}}^{(1,2)}$ because $\Sigma_{\text{full}} = \Sigma_{\text{stim}} + \Sigma_{\text{rest}}$, with $\Sigma_{\text{rest}} \succeq 0$. 

Matrix congruence preserves Loewner order. If $A \preceq B$ and $C$ is any matrix, then $C^T A C \preceq C^T B C$.

$$\kappa(A, B) = \sqrt{\lambda(A^{1/2}BA^{1/2})} = \sqrt{\lambda(B^{1/2}AB^{1/2})}$$

We have: 
$$\begin{aligned}
&\left(\Sigma_{\text{full}}^{(3)}\right)^{1/2} \tilde{\Sigma}_{\text{stim}}^{(1,2)} \left(\Sigma_{\text{full}}^{(3)}\right)^{1/2} \preceq \left(\Sigma_{\text{full}}^{(3)}\right)^{1/2} \Sigma_{\text{full}}^{(1,2)} \left(\Sigma_{\text{full}}^{(3)}\right)^{1/2} \\
\implies &\lambda_i\!\left(\left(\Sigma_{\text{full}}^{(3)}\right)^{1/2} \tilde{\Sigma}_{\text{stim}}^{(1,2)} \left(\Sigma_{\text{full}}^{(3)}\right)^{1/2}\right) \leq \lambda_i\!\left(\left(\Sigma_{\text{full}}^{(3)}\right)^{1/2} \Sigma_{\text{full}}^{(1,2)} \left(\Sigma_{\text{full}}^{(3)}\right)^{1/2}\right) \\
\implies &\kappa_i\!\left(\tilde{\Sigma}_{\text{stim}}^{(1,2)},\, \Sigma_{\text{full}}^{(3)}\right) \leq \kappa_i\!\left(\Sigma_{\text{full}}^{(1,2)},\, \Sigma_{\text{full}}^{(3)}\right) \quad \forall i \\
\end{aligned}$$

Therefore, 

$$0 \leq \frac{\sum_i \kappa_i\!\left(\tilde{\Sigma}_{\text{stim}}^{(1,2)},\, \Sigma_{\text{full}}^{(3)}\right)}{\sum_i \kappa_i\!\left(\Sigma_{\text{full}}^{(1,2)},\, \Sigma_{\text{full}}^{(3)}\right)} \leq 1.$$

##### Comparison to Centered Kernel Alignment (CKA)
Note how this differs from CKA, which uses the following formula:

$$CKA(\Sigma_{\text{stim}}, \Sigma_{\text{full}}) = \frac{\operatorname{tr}(\Sigma_{\text{stim}}\, \Sigma_{\text{full}})}{\sqrt{\operatorname{tr}(\Sigma_{\text{stim}}^2) \operatorname{tr}(\Sigma_{\text{full}}^2)}}$$

In CKA, the denominator is a normalization factor that accounts for the total variance in *each* matrix, rather than accounting for repeatability in the structure of the target matrix. CKA measures alignment independent of scale (it acts like the cosine similarity between the two matrices), while $\text{SVF}$ measures the fraction of reproducible structure in $B$ shared with $A$. 

#### Case 2: Shared Variance *Ratio*
Instead, suppose we want to compare how much the geometry of stimulus-dependent covariance contributes to the covariance of a population when the stimulus *isn't* present, for example during a spontaneous activity period (or during presentation of a different kind of stimulus). In this case, we want to know how much $\Sigma_{\text{stim}}$ contributes to the structure of $\Sigma_{\text{spont}}$, where $\Sigma_{\text{spont}} = \Sigma_{\text{non-stim}} + \Sigma_{\text{noise}}$.

We use the same three-fold partition applied independently to the spontaneous data, defining $\Sigma_{\text{spont}}^{(k)}$ and $\Sigma_{\text{spont}}^{(1,2)}$ analogously to the full covariance above. We use the following formula for the shared variance ratio:

$$\text{SVR} = \frac{\kappa\!\left(\tilde{\Sigma}_{\text{stim}}^{(1,2)},\, \Sigma_{\text{spont}}^{(3)}\right)}{\kappa\!\left(\Sigma_{\text{spont}}^{(1,2)},\, \Sigma_{\text{spont}}^{(3)}\right)}$$

There is no Loewner order between $\Sigma_{\text{stim}}$ and $\Sigma_{\text{spont}}$, so we cannot guarantee that $\text{SVR}$ is bounded between 0 and 1. However, we can still interpret $\text{SVR}$ as a ***ratio*** of shared variance. 
- If $\text{SVR} < 1$, we can interpret this as an indication that the spontaneous covariance contains partial overlap with stimulus-dependent covariance modes. 
- If $\text{SVR} \geq 1$, the stimulus covariance is shared with spontaneous as much as two samples of spontaneous is with itself - indicating that the spontaneous covariance is primarily determined by stimulus-dependent structure and is potentially quite noisy. 


#### Estimating $\Sigma_{\text{stim}}$
In practice, we don't have direct access to $\Sigma_{\text{stim}}$, so we need to estimate it from data. Although the per-fold stimulus mean $\bar{g}_s^{(k)}$ is an unbiased estimate of $\bar{g}_s$, it contains noise due to finite trial sampling. Therefore $\tilde{\Sigma}_{\text{stim}}^{(1,2)} = \operatorname{Cov}_s(\bar{g}_s^{(1,2)})$ is a biased estimator of $\Sigma_{\text{stim}}$: the sample mean noise inflates all eigenvalues.

To mitigate this bias, we can use a cross-validated estimator that uses $\mathcal{F}_1$ and $\mathcal{F}_2$ as independent half-splits:

$$\hat{\Sigma}_{\text{stim}} = \operatorname{Cov}_s\!\left(\bar{g}_s^{(1)},\, \bar{g}_s^{(2)}\right) = \mathbb{E}_s\!\left[\left(\bar{g}_s^{(1)} - \mu^{(1)}\right)\left(\bar{g}_s^{(2)} - \mu^{(2)}\right)^T\right],$$

where $\mu^{(k)} = \mathbb{E}_s[\bar{g}_s^{(k)}]$. Because $\mathcal{F}_1$ and $\mathcal{F}_2$ are disjoint, the noise in $\bar{g}_s^{(1)}$ and $\bar{g}_s^{(2)}$ is independent and does not contribute to their cross-covariance in expectation, making $\hat{\Sigma}_{\text{stim}}$ unbiased.

Unfortunately, $\hat{\Sigma}_{\text{stim}}$ is not guaranteed to be PSD — it is an asymmetric cross-covariance matrix and can have negative eigenvalues in finite samples. This means the Loewner order argument in the Lemma does not apply to $\hat{\Sigma}_{\text{stim}}$, and $\text{SVF}$ computed with $\hat{\Sigma}_{\text{stim}}$ in the numerator need not be bounded in $[0, 1]$ in finite samples.

#### The energy perspective
Above, we measure $\kappa(A, B) = \sum_i \kappa_i(A, B)$ to quantify the total shared variance across all overlap modes. This measures aligned directions in the covariance structure, and is more sensitive to distributed structure. 

An alternative is to use the a quadratic form of $\kappa$ where we measure $\kappa(A, B) = \sum_i \kappa_i^2(A, B)$ = $\operatorname{tr}(AB)$. This quantity measure the total *energy* shared, which can be dominated by high-variance modes. 

## Getting Ready for my "Stimulus Perspective" Transition
kappa(SigA, SigB) = sigma (SigA^{1/2} SigB^{1/2}))

But we don't need to look at SigA ^ {1/2}, we can look at the gram matrix of the data directly - which is almost the same. So noow instead of A^{1/2}BA^{1/2}, we can look at G^TBG where G is the centered, scaled data matrix that produces SigA in Gram form (G^TG = SigA). Here, we're looking at the stimulus covariance in the basis of B covariance. 

In this setting, we can use G0 and G1 as independent splits to make it cross-validated. It has the right expectation, but is unbiased! The denominator can be the same. We lose the fraction of variance, but of course we do get an expected value for fraction of variance :)

I want to simulate this with a better simulator for stimulus within full variance. 