# L-curve criterion

## Definition

The L-curve criterion is proposed by Hansen {footcite}`Hansen1992-pf, Hansen2000-zf`.
Let us consider the same ill-posed linear inverse problem introduced in the [](inversion.md):

$$
\mathbf{x}_\lambda := \arg\min_{\mathbf{x}}
\left[
    \| \mathbf{T} \mathbf{x} - \mathbf{b} \|_\mathbf{Q}^2 + \lambda\|\mathbf{x}\|_\mathbf{H}^2
\right].
$$

The L-curve is precisely following points curve:

$$
\left(\| \mathbf{T}\mathbf{x}_\lambda - \mathbf{b} \|_\mathbf{Q},\;\|\mathbf{x}_\lambda\|_\mathbf{H}\right)
=
\left(\sqrt{\rho}, \sqrt{\eta}\right).
$$

Here $\rho$ and $\eta$ are defined in the [inversion theory section](inversion.md#expression-of-the-squared-norm-using-the-series-expansion).

This curve is monotonically decreasing varying $\lambda$ from $0$ to $\infty$.

The L-curve criterion gives a way to choose the optimal regularization parameter $\lambda$ by finding the corner of the L-curve plotted in the log-log scale in figure below.
The reason way the corner of the L-curve is optimal is discussed in the [below section](#miscellaneous).

```{figure} ../_static/images/l_curve.svg
---
align: center
alt: L-curve
---
The schematic diagram of the L-curve. The dot on the curve represents the corner of the L-curve, which is the point where the curvature is maximal.
```

## Derivation of the curvature of the L-curve

To mathematically determine the L-curve's corner, its curvature is derived, and the corner is defined as the point where the curvature is maximal.

Recall the definition of the [series-expansion form of the solution](inversion.md#series-expansion-of-the-solution), and let

$$
\hat{\rho} \equiv \log \rho,
\quad
\hat{\eta} \equiv \log \eta
$$

such that the L-curve is a plot of $(\hat{\rho}/2,\; \hat{\eta}/2)$.

Then the curvature $\kappa(\lambda)$ of the L-curve is defined as follows:

$$
\begin{align}
\kappa(\lambda)
&\equiv
\frac{
    \left(\hat{\rho}/2\right)''\left(\hat{\eta}/2\right)'
    - \left(\hat{\rho}/2\right)'\left(\hat{\eta}/2\right)''
    }{
    \left[
        \left((\hat{\rho}/2)'\right)^2
        + \left((\hat{\eta}/2)'\right)^2
    \right]^{3/2}
    }\\
&=
2\frac{
    \hat{\rho}''\hat{\eta}'
    - \hat{\rho}'\hat{\eta}''
    }{
    \left[
        (\hat{\rho}')^2 + (\hat{\eta}')^2
    \right]^{3/2}
    },
\end{align}
$$ (curvature-original)

where the prime denotes the derivative with respect to $\lambda$.
If $\kappa(\lambda) > 0$, the L-curve is convex at $\lambda$, and if $\kappa(\lambda) < 0$, the L-curve is concave at $\lambda$.

Before expressing $\hat{\rho}'$, $\hat{\eta}'$, ... etc., the following calculation is useful:

$$
\begin{align}
f_{\lambda, i} - 1
&=
\frac{\sigma_i^2}{\sigma_i^2 + \lambda}\ - 1\\
&=
-\frac{\lambda}{\sigma_i^2 + \lambda}\\
&=
-\frac{\lambda}{\sigma_i^2}f_{\lambda, i},
\end{align}
$$ (flambra-1)

$$
\begin{align*}
\frac{\partial}{\partial \lambda}f_{\lambda, i}^2
&=
2f_{\lambda, i}f_{\lambda, i}',
\end{align*}
$$

$$
\begin{align*}
\frac{\partial}{\partial \lambda}(f_{\lambda, i} - 1)^2
&=
2(f_{\lambda, i} - 1)f_{\lambda, i}'\\
&=
-2 \frac{\lambda}{\sigma_i^2}f_{\lambda, i}f_{\lambda, i}'\\
&=
-\lambda\frac{\partial}{\partial \lambda}\frac{f_{\lambda, i}^2}{\sigma_i^2}.
\end{align*}
$$

Therefore, the following relation is obtained by calculating the derivative of $\rho$ and $\eta$:

$$
\begin{align}
\rho'
&=
\sum_{i=1}^r \frac{\partial}{\partial \lambda}(f_{\lambda, i} - 1)^2 (\mathbf{u}_i^\mathsf{T}\hat{\mathbf{b}})^2\\
&=
-\lambda\sum_{i=1}^r \frac{\partial}{\partial \lambda}\frac{f_{\lambda, i}^2}{\sigma_i^2} (\mathbf{u}_i^\mathsf{T}\hat{\mathbf{b}})^2\\
&=
-\lambda \eta'.
\end{align}
$$ (rho'-eta')

Now let us represent $\hat{\rho}', \hat{\rho}'', \hat{\eta}', \hat{\eta}''$ in terms of $\rho, \eta, \eta', \eta''$:

$$
\begin{align*}
&\hat{\rho}' = \frac{\rho'}{\rho} = -\lambda\frac{\eta'}{\rho},\\
&\hat{\eta}' = \frac{\eta'}{\eta},\\
&\hat{\rho}'' = -\frac{\eta'}{\rho} - \lambda\frac{\eta''}{\rho} - \lambda^2\frac{(\eta')^2}{\rho^2},\\
&\hat{\eta}'' = \frac{\eta''}{\eta} - \frac{(\eta')^2}{\eta^2}.
\end{align*}
$$

Substituting these into the curvature $\kappa(\lambda)$ {eq}`curvature-original`, we obtain the following:

$$
\begin{align*}\text{numerator of } \frac{\kappa(\lambda)}{2}&=\hat{\rho}''\hat{\eta}' - \hat{\rho}'\hat{\eta}''\\&=\left(-\frac{\eta'}{\rho} - \lambda\frac{\eta''}{\rho} - \lambda^2\frac{(\eta')^2}{\rho^2}\right)\left(\frac{\eta'}{\eta}\right)
=
\left(
-\lambda\frac{\eta'}{\rho}
\right)
\left(
\frac{\eta''}{\eta} - \frac{(\eta')^2}{\eta^2}
\right)
\\
&=
-\lambda\frac{(\eta')^3}{\rho\eta^2} - \frac{(\eta')^2}{\rho\eta} - \lambda^2\frac{(\eta')^3}{\rho^2\eta}\\
&=
-\frac{(\eta')^3}{\rho^2\eta^2}
\left(
\lambda^2\eta + \lambda\rho + \rho\eta/\eta'
\right).\\
\end{align*}
$$

$$
\begin{align*}
\text{denominator of } \kappa(\lambda)
&=
\left[
    \left(\hat{\rho}'\right)^2
    + \left(\hat{\eta}'\right)^2
\right]^{3/2}\\
&=
\left[
    \left(
        -\lambda\frac{\eta'}{\rho}
    \right)^2
    + \left(
        \frac{\eta'}{\eta}
    \right)^2
\right]^{3/2}\\
&=
\left[
    \left(
        \frac{\eta'}{\rho\eta}
    \right)^2
    \left(
        \lambda^2\eta^2 + \rho^2
    \right)
\right]^{3/2}\\
&=
\frac{(\eta')^3}{\rho^3\eta^3}
\left(
    \lambda^2\eta^2 + \rho^2
\right)^{3/2}.
\end{align*}
$$

$$
\therefore\kappa(\lambda)
=
-2\rho\eta\frac{\lambda^2\eta + \lambda\rho + \rho\eta/\eta'}{(\lambda^2\eta^2 + \rho^2)^{3/2}}.
$$ (curvature)

### Express $\eta'$ with series expansion components

Let us express $\eta'$ with series expansion components $\mathbf{S}$, $\mathbf{U}$, $\mathbf{V}$, etc.

Firstly the derivative of $f_{\lambda, i}$ with respect to $\lambda$ can be expressed using the relation {eq}`flambra-1` as follows:

$$
\begin{align*}
f_{\lambda, i}'
&=
\frac{\partial}{\partial \lambda}\left(\frac{\sigma_i^2}{\sigma_i^2 + \lambda}\right)\\
&=
-\frac{\sigma_i^2}{(\sigma_i^2 + \lambda)^2}\\
&=
\frac{1}{\lambda}\cdot -\frac{\lambda}{\sigma_i^2}f_{\lambda, i} \cdot f_{\lambda, i}\\
&=
\frac{1}{\lambda}(f_{\lambda, i} - 1)f_{\lambda, i}.
\end{align*}
$$

Therefore, $\eta'$ is expressed as follows:

$$
\begin{align}
\eta'
&=
\frac{\partial}{\partial \lambda}\eta\\
&=
\frac{\partial}{\partial \lambda} \sum_{i=1}^r \frac{f_{\lambda, i}^2}{\sigma_i^2} (\mathbf{u}_i^\mathsf{T}\hat{\mathbf{b}})^2\\
&=
\sum_{i=1}^r 2f_{\lambda, i}f_{\lambda, i}'\frac{1}{\sigma_i^2} (\mathbf{u}_i^\mathsf{T}\hat{\mathbf{b}})^2\\
&=
\frac{2}{\lambda} \sum_{i=1}^r (f_{\lambda, i} - 1)f_{\lambda, i}^2 \frac{1}{\sigma_i^2} (\mathbf{u}_i^\mathsf{T}\hat{\mathbf{b}})^2\\
&=
\frac{2}{\lambda}
(\mathbf{U}^\mathsf{T}\hat{\mathbf{b}})^\mathsf{T}
(\mathbf{F}_\lambda - \mathbf{I}_r)\mathbf{F}_\lambda^2\mathbf{S}^{-2}\ \mathbf{U}^\mathsf{T}\hat{\mathbf{b}}.
\end{align}
$$

## Miscellaneous

### Theorem 1.

The L-curve is monotonically decreasing varying $\lambda$ from $0$ to $\infty$.

```{admonition} Proof

Let us calculate the derivative of $\sqrt{\eta}$ as a function of $\sqrt{\rho}$ using the relation {eq}`rho'-eta'`:

$$
\begin{align*}
\frac{\partial \sqrt{\eta}}{\partial \sqrt{\rho}}
&=
\frac{\partial \sqrt{\eta} / \partial \lambda}{\partial \sqrt{\rho} / \partial \lambda}\\
&=
\frac{\eta'}{\rho'}\frac{\rho}{\eta}\\
&=
-\frac{\rho}{\lambda\eta}\\
&< 0. \qquad(\because \rho, \eta > 0 \text{ and } \lambda \in (0, \infty))
\end{align*}
$$

```

### Theorem 2.

The following asymptotic behavior of the L-curve is obtained:

$$
\lim_{\lambda \to 0} \left(\sqrt{\rho},\; \sqrt{\eta} \right)
=
\left(0,\; \|\mathbf{x}_0\|_\mathbf{H}\right),
\quad
\lim_{\lambda \to \infty} \left(\sqrt{\rho},\; \sqrt{\eta}\right) = \left(\|\hat{\mathbf{b}}\|,\; 0 \right).
$$

where $\mathbf{x}_0 = (\mathbf{T}^\mathsf{T}\mathbf{Q}\mathbf{T})^{-1}\mathbf{T}^\mathsf{T}\mathbf{Q}\mathbf{b}$, which is the least-squares solution.

```{admonition} Proof

The filter factor $f_{\lambda, i}$ is expressed as follows:

$$
f_{\lambda, i} = \frac{1}{1 + \lambda/\sigma_i^2}
\to
\begin{cases}
1 \quad (\lambda \to 0)\\
0 \quad (\lambda \to \infty)
\end{cases}.
$$

$$
\therefore
\mathbf{F}_\lambda
\to
\begin{cases}
\mathbf{I}_r \quad (\lambda \to 0)\\
\mathbf{0} \quad (\lambda \to \infty)
\end{cases}.
$$

According to the [Expression of the squared norm using the series expansion](inversion.md#expression-of-the-squared-norm-using-the-series-expansion), $\rho$ and $\eta$ are asymptotically expressed as follows:

$$
\begin{align*}
\rho
&=
\|
    \mathbf{U}(\mathbf{F}_\lambda - \mathbf{I}_r)\mathbf{U}^\mathsf{T}\hat{\mathbf{b}}
\|^2
\to
\begin{cases}
0 \quad &(\lambda \to 0)\\
\|\hat{\mathbf{b}}\|^2 \quad &(\lambda \to \infty)
\end{cases},\\
\eta
&=\|\mathbf{x}_\lambda\|_\mathbf{H}^2
\to
\begin{cases}
\|\mathbf{x}_0\|_\mathbf{H}^2 \quad &(\lambda \to 0)\\
0 \quad &(\lambda \to \infty)
\end{cases}.
\end{align*}
$$

```

### Characteristics of the L-curve

The given data is often noisy, and the data $\mathbf{b}$ can be written as

$$
\mathbf{b} = \bar{\mathbf{b}} + \mathbf{e},\qquad \bar{\mathbf{b}} = \mathbf{T}\bar{\mathbf{x}},
$$

where $\bar{\mathbf{b}}$ represents the exact unperturbed data,
$\bar{\mathbf{x}}$ represents the exact solution,
and $\mathbf{e}$ represents the errors in the data.

```{important}
---
title: Assumptions
---
Assuming the following conditions:

1. $|\mathbf{u}_i^\mathsf{T}\mathbf{B}\bar{\mathbf{b}}|$ decay faster than $\sigma_i$. (Discrete Picard condition (**DPC**))
1. $\mathbf{e}$ is the white noise.
1. Sufficient SNR (Signal-to-Noise Ratio) is given, i.e. $\|\bar{\mathbf{b}}\|/\|\mathbf{e}\| \gg 1$.

Then the L-curve has the corner where the residual norm $\|\mathbf{T}\mathbf{x}_\lambda - \mathbf{b}\|_\mathbf{Q}$ is approximated to be equal to $\|\mathbf{e}\|_\mathbf{Q}$.

```

**Description**<br>
The $\eta$ is written as

$$
\eta
=
\sum_{i=1}^r
\left(
f_{\lambda, i} \frac{\mathbf{u}_i^\mathsf{T}\mathbf{B}\bar{\mathbf{b}}}{\sigma_i}
-
f_{\lambda, i} \frac{\mathbf{u}_i^\mathsf{T}\mathbf{B}\mathbf{e}}{\sigma_i}
\right)^2
$$

According to the first condition, $\frac{\mathbf{u}_i^\mathsf{T}\mathbf{B}\bar{\mathbf{b}}}{\sigma_i}$ does not become large as $i$ increases, while $\frac{\mathbf{u}_i^\mathsf{T}\mathbf{B}\mathbf{e}}{\sigma_i}$ becomes large because it does not satisfy the DPC. So, the $\eta$ is dominated by the second term in $\lambda \ll 1$.
Increasing $\lambda$, the $\eta$ decreases because the high-frequency components of the second term are suppressed by the $f_{\lambda, i}$, then the $\eta$ is dominated by the first term where the L-curve is horizontal.
Somewhere in between, there is a range of $\lambda$-values that correspond to a transition between the two domination L-curves.

When we find the L-curve corner numerically, it is important to set the range of $\lambda$.
Regińska proved that

```{quote} Theorem
The log-log L-curve is always strictly concave for

$$ \sigma_r^2\leq \lambda\leq\sigma_1^2, $$

where $\sigma_1$ and $\sigma_r$ are the largest and smallest singular values, respectively {footcite}`Reginska2012-dh`.
```

Hansen also presented the reason using the curvature expression {eq}`curvature` and modeling $|\mathbf{u}_i^\mathsf{T}\hat{\mathbf{b}}|$ as a power-law function of $\sigma_i$ at Section 6 in {footcite}`Hansen2000-zf`.

## Example

The example script shows in [a notebook](../notebooks/non-iterative/L_curve).

```{footbibliography} ../references.bib
```
