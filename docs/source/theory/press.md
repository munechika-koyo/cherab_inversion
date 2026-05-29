# PRESS criterion

## Definition

The Predicted Residual Error Sum of Squares (PRESS), called also the ordinary cross-validation, is based on the basic leave-one-out cross-validation, which is proposed by D. M. Allen, 1974 {footcite}`Allen1974-bu`.<br>
Let $\mathbf{x}_\lambda^{(l)}$ be the solution in which the $l$-th observation is omitted.
The PRESS criterion's argument is that if $\lambda$ is a good choice, then the $l$-th component $\left(\mathbf{T}\mathbf{x}_\lambda^{(l)}\right)_l$ should be a good predictor of $b_l$.
Therefore, the PRESS criterion leads to choosing $\lambda$ as the minimizer of the PRESS function $\mathcal{P}(\lambda)$, defined by

$$
\mathcal{P}(\lambda)
\equiv
\sum_{l=1}^{M}
\left[
    \left(
        \mathbf{T}\mathbf{x}_\lambda^{(l)}
    \right)_l
    -
    b_l
\right]^2.
$$

Using the [Generalized Tikhonov regularization solution](inversion.md#generalized-tikhonov-regularization), let

$$
\hat{\mathbf{T}} \equiv \mathbf{B}\mathbf{T},\quad
\hat{\mathbf{b}} \equiv \mathbf{B}\mathbf{b},\quad
\mathbf{Q} = \mathbf{B}^\mathsf{T}\mathbf{B}.
$$

Then the PRESS criterion can be rewritten as

# $$\mathcal{P}(\lambda)

\left\|
\mathbf{D}*\lambda
\left(
\mathbf{I} - \mathbf{A}*\lambda
\right)
\hat{\mathbf{b}}
\right\|_2^2,
$$ (PRESS_SM)

where

# $$\begin{align*}\mathbf{A}_\lambda&\equiv\hat{\mathbf{T}}\left(\hat{\mathbf{T}}^\mathsf{T}\hat{\mathbf{T}} + \lambda\mathbf{H}\right)^{-1}\hat{\mathbf{T}}^\mathsf{T}

\mathbf{B}\mathbf{T}\left(\mathbf{T}^\mathsf{T}\mathbf{Q}\mathbf{T} + \lambda\mathbf{H}\right)^{-1}\mathbf{T}^\mathsf{T}\mathbf{B}^\mathsf{T},\\
\mathbf{D}*\lambda
&\equiv
\mathrm{diag}
\left(
\cdots,\frac{1}{1 - a*{\lambda, ii}},\cdots
\right),\\
a_{\lambda, ii}
&\equiv
\left(\mathbf{A}*\lambda\right)*{ii}.
\end{align*}
$$

Using [series-expansion form of the solution](inversion.md#series-expansion-of-the-solution), $\mathcal{P}(\lambda)$ can be written as

# $$\mathcal{P}(\lambda)

\left\|
\left[
\mathrm{Diag}\left(
\mathbf{I} - \mathbf{U}\mathbf{F}*\lambda\mathbf{U}^\mathsf{T}
\right)
\right]^{-1}
\left(
\mathbf{I} - \mathbf{U}\mathbf{F}*\lambda\mathbf{U}^\mathsf{T}
\right)\hat{\mathbf{b}}
\right\|_2^2,
$$ (PRESS_series)

where $\mathrm{Diag}(\mathbf{M})$ denotes the diagonal matrix formed from the diagonal entries of $\mathbf{M}$.

Equivalently, expanding element-wise with $u_{li}\equiv(\mathbf{U})_{li}$:

# $$\mathcal{P}(\lambda)

\sum_{l=1}^{M}
\left[
\frac{
\hat{b}*l -
\sum*{i=1}^{r}
f_{\lambda,i}\,u_{li}\,(\mathbf{u}*i^\mathsf{T}\hat{\mathbf{b}})
}{
1 -
\sum*{i=1}^{r} f_{\lambda,i}\,u_{li}^2
}
\right]^2.
$$ (PRESS_series_comp)

### Derivation of {eq}`PRESS_SM`

Let $\hat{\mathbf{t}}_l\in\mathbb{R}^N$ be the $l$-th row of $\hat{\mathbf{T}}$ treated as a column vector, and define

$$
\mathbf{K}_\lambda \equiv \hat{\mathbf{T}}^\mathsf{T}\hat{\mathbf{T}} + \lambda\mathbf{H}
= \mathbf{T}^\mathsf{T}\mathbf{Q}\mathbf{T} + \lambda\mathbf{H}.
$$

Then the solution of the transformed generalized Tikhonov problem is

$$
\mathbf{x}_\lambda = \mathbf{K}_\lambda^{-1}\hat{\mathbf{T}}^\mathsf{T}\hat{\mathbf{b}}.
$$

Removing the $l$-th observation from the transformed system changes the normal equations as

$$
\begin{cases}
\hat{\mathbf{T}}^{(l)\mathsf{T}}\hat{\mathbf{T}}^{(l)}
= \hat{\mathbf{T}}^\mathsf{T}\hat{\mathbf{T}} - \hat{\mathbf{t}}_l\hat{\mathbf{t}}_l^\mathsf{T},\\[4pt]
\hat{\mathbf{T}}^{(l)\mathsf{T}}\hat{\mathbf{b}}^{(l)}
= \hat{\mathbf{T}}^\mathsf{T}\hat{\mathbf{b}} - \hat{b}_l\hat{\mathbf{t}}_l,
\end{cases}
$$

so the $l$-th leave-one-out solution $\mathbf{x}_\lambda^{(l)}$ is given by

# $$\mathbf{x}_\lambda^{(l)}

\bigl(\mathbf{K}_\lambda - \hat{\mathbf{t}}_l\hat{\mathbf{t}}_l^\mathsf{T}\bigr)^{-1}
\bigl(\hat{\mathbf{T}}^\mathsf{T}\hat{\mathbf{b}} - \hat{b}_l\hat{\mathbf{t}}_l\bigr).
$$

Applying the [Sherman-Morrison formula](wiki:Sherman–Morrison_formula) to the rank-1 perturbation with $\hat{\mathbf{c}}_l\equiv\mathbf{K}_\lambda^{-1}\hat{\mathbf{t}}_l$ gives

# $$\bigl(\mathbf{K}_\lambda - \hat{\mathbf{t}}_l\hat{\mathbf{t}}_l^\mathsf{T}\bigr)^{-1}

\mathbf{K}*\lambda^{-1}
+
\frac{\hat{\mathbf{c}}*l\hat{\mathbf{c}}*l^\mathsf{T}}{1 - a*{\lambda,ll}},
\quad
a*{\lambda,ll}
= \hat{\mathbf{t}}*l^\mathsf{T}\hat{\mathbf{c}}*l
= \bigl(\hat{\mathbf{T}}\mathbf{K}*\lambda^{-1}\hat{\mathbf{T}}^\mathsf{T}\bigr)*{ll}
= \bigl(\mathbf{A}*\lambda\bigr)_{ll}.
$$

Substituting and using
$\hat{\mathbf{c}}_l^\mathsf{T}\hat{\mathbf{T}}^\mathsf{T}\hat{\mathbf{b}}
=\hat{\mathbf{t}}_l^\mathsf{T}\mathbf{x}_\lambda
=(\mathbf{A}_\lambda\hat{\mathbf{b}})_l$
and
$\hat{\mathbf{c}}_l^\mathsf{T}\hat{\mathbf{t}}_l=a_{\lambda,ll}$:

$$
\begin{align*}
\mathbf{x}_\lambda^{(l)}
&=
\mathbf{x}_\lambda - \hat{b}_l\hat{\mathbf{c}}_l
+
\frac{\hat{\mathbf{c}}_l\bigl[(\mathbf{A}_\lambda\hat{\mathbf{b}})_l - \hat{b}_l a_{\lambda,ll}\bigr]}
     {1 - a_{\lambda,ll}}\\
&=
\mathbf{x}_\lambda
+
\hat{\mathbf{c}}_l\cdot
\frac{(\mathbf{A}_\lambda\hat{\mathbf{b}})_l - \hat{b}_l}{1 - a_{\lambda,ll}}.
\end{align*}
$$

Hence the $l$-th leave-one-out prediction error in the transformed space is

$$
\begin{align*}
\bigl(\hat{\mathbf{T}}\mathbf{x}_\lambda^{(l)}\bigr)_l - \hat{b}_l
&=
\hat{\mathbf{t}}*l^\mathsf{T}\mathbf{x}*\lambda

- a_{\lambda,ll}\cdot
  \frac{(\mathbf{A}_\lambda\hat{\mathbf{b}})_l - \hat{b}*l}{1 - a*{\lambda,ll}}

* # \hat{b}*l\\&=\bigl[(\mathbf{A}*\lambda\hat{\mathbf{b}})*l - \hat{b}*l\bigr]\left(1 + \frac{a*{\lambda,ll}}{1-a*{\lambda,ll}}\right)\\&=\frac{(\mathbf{A}_\lambda\hat{\mathbf{b}})*l - \hat{b}*l}{1-a*{\lambda,ll}}
  -\frac{\bigl[(\mathbf{I}-\mathbf{A}*\lambda)\hat{\mathbf{b}}\bigr]*l}{1-a*{\lambda,ll}}.
  \end{align*}
  $$

Summing the squares over all $l$ yields {eq}`PRESS_SM`:

# $$\mathcal{P}(\lambda)

# \sum_{l=1}^{M}\left[\frac{\bigl[(\mathbf{I}-\mathbf{A}_\lambda)\hat{\mathbf{b}}\bigr]*l}{1-a*{\lambda,ll}}\right]^2

\bigl\|\mathbf{D}*\lambda(\mathbf{I}-\mathbf{A}*\lambda)\hat{\mathbf{b}}\bigr\|_2^2.
$$

### Derivation of {eq}`PRESS_series`

Using the decomposed solution form, we first identify $\mathbf{A}_\lambda$:

$$
\begin{align*}
\mathbf{A}_\lambda\hat{\mathbf{b}}
&=
\hat{\mathbf{T}}\mathbf{x}_\lambda\\
&=
\hat{\mathbf{T}}\tilde{\mathbf{V}}\mathbf{F}_\lambda\mathbf{S}^{-1}\mathbf{U}^\mathsf{T}\hat{\mathbf{b}}\\
&=
\mathbf{U}\mathbf{S}\mathbf{F}_\lambda\mathbf{S}^{-1}\mathbf{U}^\mathsf{T}\hat{\mathbf{b}}\quad(\because \hat{\mathbf{T}}\tilde{\mathbf{V}} = \mathbf{U}\mathbf{S})\\
&=
\mathbf{U}\mathbf{F}_\lambda\mathbf{U}^\mathsf{T}\hat{\mathbf{b}},\\
\therefore\quad
\mathbf{A}_\lambda &= \mathbf{U}\mathbf{F}_\lambda\mathbf{U}^\mathsf{T}.
\end{align*}
$$

Substituting the above directly into {eq}`PRESS_SM` gives {eq}`PRESS_series`, since $\mathbf{D}_\lambda = [\mathrm{Diag}(\mathbf{I}-\mathbf{A}_\lambda)]^{-1}$ by definition.

To obtain the component form {eq}`PRESS_series_comp`, note that the $l$-th diagonal entry of $\mathbf{A}_\lambda$ is

# $$a_{\lambda,ll}

# \left(\mathbf{U}\mathbf{F}*\lambda\mathbf{U}^\mathsf{T}\right)*{ll}

\sum_{i=1}^{r} f_{\lambda,i}\,u_{li}^2,
$$

and the $l$-th component of $(\mathbf{I}-\mathbf{A}_\lambda)\hat{\mathbf{b}}$ is

# $$\hat{b}*l - \left(\mathbf{U}\mathbf{F}*\lambda\mathbf{U}^\mathsf{T}\hat{\mathbf{b}}\right)_l

\hat{b}*l - \sum*{i=1}^{r} f_{\lambda,i}\,u_{li}\,(\mathbf{u}_i^\mathsf{T}\hat{\mathbf{b}}).
$$

Dividing the $l$-th residual component by $(1-a_{\lambda,ll})$ and summing the squares over $l$ yields {eq}`PRESS_series_comp`.

```{footbibliography} ../references.bib
```
