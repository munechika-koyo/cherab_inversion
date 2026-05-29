# GCV criterion

## Definition

The Generalized Cross Validation (GCV) criterion is a very similar method to the [PRESS method](press).
GCV is a rotation-invariant form of the PRESS method.
The deriviation of the GCV from PRESS is shown in Golub, et al. 1979 {footcite}`Golub1979-gf`.

GCV leads to choosing $\lambda$ as the minimizer of the GCV function $\mathcal{G}(\lambda)$, defined by

$$
\mathcal{G}(\lambda)
\equiv
\frac{ \| (\mathbf{I} - \mathbf{A}_\lambda)\mathbf{B}\mathbf{b} \|^2 }
     {\mathrm{tr}(\mathbf{I} - \mathbf{A}_\lambda)^2},
$$

where $\mathbf{A}_\lambda \equiv \mathbf{B}\mathbf{T}(\mathbf{T}^\mathsf{T}\mathbf{Q}\mathbf{T} + \lambda\mathbf{H})^{-1}\mathbf{T}^\mathsf{T}\mathbf{B}^\mathsf{T}$, $\mathrm{tr}(\cdot)$ is the trace of a matrix, and $\mathbf{Q}=\mathbf{B}^\mathsf{T}\mathbf{B}$.

Using [series-expansion form of the solution](inversion.md#series-expansion-of-the-solution), $\mathcal{G}(\lambda)$ can be written as

# $$\mathcal{G}(\lambda)

\frac{\rho}
{\left(r - \sum_{i=1}^r f_{\lambda,i} \right)^2}.
$$ (gcv_series)

### Deriviation of {eq}`gcv_series`

Recalling the [Generalized Tikhonov regularized](./inversion.md#generalized-tikhonov-regularization) solution form and the [series expansion](./inversion.md#series-expansion-of-the-solution), we obtain the following:

$$
\begin{align*}
\mathbf{A}_\lambda \mathbf{B}\mathbf{b}
&=
\mathbf{B}\mathbf{T} \mathbf{x}_\lambda\\
&=
\mathbf{B}\mathbf{T}\tilde{\mathbf{V}}\mathbf{F}_\lambda\mathbf{S}^{-1}\mathbf{U}^\mathsf{T}\hat{\mathbf{b}}\\
&=
\mathbf{U}\mathbf{S}\mathbf{F}_\lambda\mathbf{S}^{-1}\mathbf{U}^\mathsf{T}\mathbf{B}\mathbf{b}\quad(\because \mathbf{B}\mathbf{T}\tilde{\mathbf{V}} = \mathbf{U}\mathbf{S})\\
&=
\mathbf{U}\mathbf{F}_\lambda\mathbf{U}^\mathsf{T}\mathbf{B}\mathbf{b}.\\
\therefore
\mathbf{A}_\lambda &= \mathbf{U}\mathbf{F}_\lambda\mathbf{U}^\mathsf{T}.
\end{align*}
$$

Then we have

$$
\begin{align*}
\text{numerator of } \mathcal{G}(\lambda)
&=
\| (\mathbf{I} - \mathbf{A}_\lambda)\mathbf{B}\mathbf{b} \|^2 \\
&=
\| \mathbf{B}\mathbf{b} - \mathbf{B}\mathbf{T}\mathbf{x}_\lambda \|^2\\
&=
\| \mathbf{b} - \mathbf{T}\mathbf{x}_\lambda\|_\mathbf{Q}^2 = \rho.\\\\
\text{denominator of } \mathcal{G}(\lambda)
&=
\mathrm{tr}(\mathbf{I} - \mathbf{A}_\lambda)^2\\
&=
\left(
    \mathrm{tr}(\mathbf{I}) - \mathrm{tr}(\mathbf{A}_\lambda)
\right)^2\\
&=
\left(
    \mathrm{tr}(\mathbf{I})  - \mathrm{tr}(\mathbf{U}\mathbf{F}_\lambda\mathbf{U}^\mathsf{T})
\right)^2\\
&=
\left(
    \mathrm{tr}(\mathbf{I})  - \mathrm{tr}(\mathbf{F}_\lambda)
\right)^2
\quad\left(
    \because \mathrm{tr}(\mathbf{U}\mathbf{F}_\lambda\mathbf{U}^\mathsf{T}) = \mathrm{tr}(\mathbf{U}^\mathsf{T}\mathbf{U}\mathbf{F}_\lambda) = \mathrm{tr}(\mathbf{F}_\lambda)
\right)\\
&=
\left(
    \sum_{i=1}^r 1 - \sum_{i=1}^r f_{\lambda,i}
\right)^2\\
&=
\left(
    r - \sum_{i=1}^r f_{\lambda,i}
\right)^2.
\end{align*}
$$

## Example

The example is shown in [a notebook](../notebooks/non-iterative/gcv).

## Limitation

GCV is a good method when the noise is unknown and the noise is assumed to be white noise, however, it often fails to give a satisfactory result when the error is highly correlated.<br>
See the [example case](../notebooks/non-iterative/lcurve_vs_gcv) for the limitation of GCV.

```{footbibliography} ../references.bib
```
