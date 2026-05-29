# Maximum Likelihood Expectation Maximization

## Definition

Maximum Likelihood Expectation Maximization (MLEM) {footcite}`Shepp1982-ky` is an iterative method for solving the inverse problem

$$
\mathbf{T}\mathbf{x} = \mathbf{b},
$$

especially when the measured data are count-like and are modeled by Poisson statistics.
In that setting, MLEM seeks a non-negative solution by maximizing the likelihood of observing $\mathbf{b}\in\mathbb{R}^M$ given the forward model $\mathbf{T}\in\mathbb{R}^{M\times N}$.

For a current estimate $\mathbf{x}^{(k)}$, the update used in this package is

$$
\mathbf{x}^{(k+1)}
=
\mathbf{x}^{(k)}
\odot
\left[
\mathbf{T}^\mathsf{T}
\left(
\mathbf{b} \oslash \mathbf{T}\mathbf{x}^{(k)}
\right)
\oslash
\mathbf{T}^\mathsf{T}\mathbf{1}_M
\right],
$$ (mlem_update)

where $\odot$ and $\oslash$ denote element-wise product and division ([Hadamard product](<wiki:Hadamard_product_(matrices)>)).
This multiplicative update keeps non-negative iterates non-negative when the initial guess is non-negative.

## Derivation (outline)

Assume independent Poisson observations $b_m \sim \mathrm{Poisson}((\mathbf{T}\mathbf{x})_m)$ for $m=1,\dots,M$.
Ignoring constants independent of $\mathbf{x}$, the log-likelihood is

$$
\log \mathcal{L}(\mathbf{x})
=
\sum_{m=1}^{M}
\left[
b_m \log (\mathbf{T}\mathbf{x})_m - (\mathbf{T}\mathbf{x})_m
\right].
$$

Introducing latent contributions and applying the EM procedure yields the fixed-point map in {eq}`mlem_update`.
The factor $(\mathbf{T}^\mathsf{T}\mathbf{1}_M)^{-1}$ acts as a sensitivity normalization for each unknown component.

## Implementation

The implementation of `cherab.inversion.statistical.MLEM` follows the update in {eq}`mlem_update` with the following procedure:

1. Set an initial guess $\mathbf{x}^{(0)}$ (default is ones).
1. Compute forward projection $\mathbf{T}\mathbf{x}^{(k)}$.
1. Form the ratio $\mathbf{b} \oslash (\mathbf{T}\mathbf{x}^{(k)})$.
1. Back-project and normalize by $(\mathbf{T}^\mathsf{T}\mathbf{1}_M)$.
1. Update $\mathbf{x}^{(k+1)}$ multiplicatively.
1. Stop when

$$

\max_i\left|x_i^{(k+1)} - x_i^{(k)}\right|
<
\mathrm{tol}\cdot\max_i\left|x_i^{(k)}\right|

$$

or the maximum iteration count is reached.

The solver supports both single-vector data and multi-column data (multiple time slices) with the same element-wise formula.

## Notes

- MLEM typically converges stably but can be slow near convergence.
- The method does not require an explicit regularization parameter, unlike L-curve, PRESS, or GCV.
- In practice, early stopping often plays the role of implicit regularization.

## Example

The example is shown in [a notebook](../notebooks/iterative/03-melm).

```{footbibliography} ../references.bib
```
