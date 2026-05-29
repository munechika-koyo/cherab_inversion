# Theory of inversion problem

## Definition of the ill-posed linear equation

The inversion problem is described as a linear equation:

$$
\mathbf{T} \mathbf{x} = \mathbf{b},
$$ (linear_equation)

where $\mathbf{T}\in\mathbb{R}^{M\times N}$ is a linear operator, $\mathbf{x}\in\mathbb{R}^N$ is the a solution vector of the inversion problem, and $\mathbf{b}\in\mathbb{R}^M$ is the given data vector.

Frequently, the above equation cannot be solved directly because of the following reasons:

1. The number of data $M$ is less than the number of unknowns $N$.
1. The data $\mathbf{b}$ is contaminated by noise.
1. The operator $\mathbf{T}$ (or $\mathbf{T}^\mathsf{T}\mathbf{T}$) is not full rank or not invertible.

The above equation is called an ill-posed linear equation.

## Generalized Tikhonov Regularization

In order to solve the ill-posed linear equation {eq}`linear_equation`, we need to introduce a objective (or penalty) functional: $O(\mathbf{x})$ and minimize the following functional {footcite}`Ingesson2008-ve`:

$$
\|\mathbf{T} \mathbf{x} - \mathbf{b} \|_\mathbf{Q}^2 + \lambda \cdot O(\mathbf{x}),
$$ (minimize-functional)

where $\|\mathbf{x}\|_\mathbf{Q}^2$ stands for the weighted norm squared $\mathbf{x}^\mathsf{T} \mathbf{Q} \mathbf{x}$ (compare with the [Mahalanobis distance](wiki:Mahalanobis_distance)) with a symmetric positive semi-definite matrix $\mathbf{Q}$.
$\lambda$ is a regularization parameter that controls the trade-off between the data
misfit and the objective functional.

The objective functional is often defined as a quadratic form:
$O(\mathbf{x}) = \mathbf{x}^\mathsf{T} \mathbf{H} \mathbf{x} = \|\mathbf{x}\|_\mathbf{H}^2$, where $\mathbf{H}$ is called the regularization matrix.
Thus, the functional {eq}`minimize-functional` can be rewritten as:

$$
\|\mathbf{T}\mathbf{x} - \mathbf{b}\|_\mathbf{Q}^2 + \lambda\|\mathbf{x}\|_\mathbf{H}^2.
$$ (generalized-tikhonov)

This regularization scheme is called the **Generalized Tikhonov Regularization**.
The conventional Tikhonov regularization is a special case for $\mathbf{Q}$ and $\mathbf{H}$ being identity matrices.
Additionally, the first and second terms are called the residual and regularization terms, respectively.

The solution $\mathbf{x}$ can be obtained by differentiating the
functional {eq}`generalized-tikhonov` with respect to $\mathbf{x}$ and setting it to zero as follows:

$$
\begin{align*}
\frac{\partial}{\partial \mathbf{x}}
\left[
    \|
        \mathbf{T}\mathbf{x} - \mathbf{b}\|_\mathbf{Q}^2
        +
        \lambda\|\mathbf{x}
    \|_\mathbf{H}^2
\right]
&=
\frac{\partial}{\partial \mathbf{x}}
\left[
    (\mathbf{T}\mathbf{x} - \mathbf{b})^\mathsf{T} \mathbf{Q} (\mathbf{T}\mathbf{x} - \mathbf{b})
    +
    \lambda\mathbf{x}^\mathsf{T} \mathbf{H} \mathbf{x}
\right]\\
&=
\frac{\partial}{\partial \mathbf{x}}
\left[
    \mathbf{x}^\mathsf{T} \left(\mathbf{T}^\mathsf{T} \mathbf{Q} \mathbf{T} + \lambda \mathbf{H}\right) \mathbf{x}
    -
    2\mathbf{b}^\mathsf{T} \mathbf{Q} \mathbf{T} \mathbf{x}
    +
    \mathbf{b}^\mathsf{T} \mathbf{Q} \mathbf{b}
\right]
\quad\left(
\because
\mathbf{b}^\mathsf{T} \mathbf{Q} \mathbf{T} \mathbf{x}
=
\mathbf{x}^\mathsf{T} \mathbf{T}^\mathsf{T} \mathbf{Q} \mathbf{b}\in \mathbb{R}^1
\right)\\
&=
2\left(
    \mathbf{T}^\mathsf{T} \mathbf{Q} \mathbf{T} + \lambda \mathbf{H}
\right) \mathbf{x}
- 2\mathbf{T}^\mathsf{T} \mathbf{Q} \mathbf{b}
\end{align*}
$$

Therefore, the solution $\mathbf{x}$ is given by:

$$
\mathbf{x} = \left(\mathbf{T}^\mathsf{T} \mathbf{Q} \mathbf{T} + \lambda \mathbf{H}\right)^{-1} \mathbf{T}^\mathsf{T} \mathbf{Q} \mathbf{b}.
$$ (solution)

## Series expansion of the solution

Although a direct inverse calculation for {eq}`solution` is possible, it often needs a lot of computational resources.
Additionally, to comprehend the solution, the cholesky decomposition and the singular value decomposition are often used {footcite}`Odstrcil2016-va`.

### 1. Cholesky decomposition

Let $\mathbf{Q}$ and $\mathbf{H}$ be factorized as follows:

$$
\begin{cases}
    \mathbf{P}_\mathbf{Q}\mathbf{Q}\mathbf{P}_\mathbf{Q}^\mathsf{T} = \mathbf{L}_\mathbf{Q}\mathbf{L}_\mathbf{Q}^\mathsf{T},\\
    \mathbf{P}_\mathbf{H}\mathbf{H}\mathbf{P}_\mathbf{H}^\mathsf{T} = \mathbf{L}_\mathbf{H}\mathbf{L}_\mathbf{H}^\mathsf{T},
\end{cases}
$$ (cholesky)

where $\mathbf{P}_\mathbf{Q}, \mathbf{P}_\mathbf{H}$ are fill-reducing permutation matrices and $\mathbf{L}_\mathbf{Q}, \mathbf{L}_\mathbf{H}$ are lower triangular matrices.
Let {eq}`cholesky` be simple as follows:

$$
\begin{cases}
    \mathbf{Q} = \mathbf{B}^\mathsf{T}\mathbf{B},\quad(\mathbf{B} \equiv \mathbf{L}_\mathbf{Q}^\mathsf{T}\mathbf{P}_\mathbf{Q}),\\
    \mathbf{H} = \mathbf{C}^\mathsf{T}\mathbf{C},\quad(\mathbf{C} \equiv \mathbf{L}_\mathbf{H}^\mathsf{T}\mathbf{P}_\mathbf{H}).
\end{cases}
$$ (cholesky-simple)

### 2. Singular Value Decomposition

Let us substitute the result of the cholesky decomposition {eq}`cholesky-simple` into {eq}`solution`:

$$
\begin{align*}
\mathbf{x}
&=
\left(
    \mathbf{T}^\mathsf{T} \mathbf{Q} \mathbf{T} + \lambda \mathbf{H}
\right)^{-1}\mathbf{T}^\mathsf{T} \mathbf{Q} \mathbf{b} \\
&=
\left(
    \mathbf{T}^\mathsf{T} \mathbf{B}\mathbf{B}^\mathsf{T} \mathbf{T}
    + \lambda \mathbf{C}\mathbf{C}^\mathsf{T}
\right)^{-1} \mathbf{T}^\mathsf{T} \mathbf{b}\\
&=
\left[
    \mathbf{C}^\mathsf{T}
    \left(
        \mathbf{C}^\mathsf{-T} \mathbf{T}^\mathsf{T} \mathbf{B}^\mathsf{T} \mathbf{B} \mathbf{T} \mathbf{C}^{-1}
        + \lambda \mathbf{I}_N
    \right)
    \mathbf{C}
\right]^{-1}
\mathbf{T}^\mathsf{T}\mathbf{B}^\mathsf{T}\mathbf{B} \mathbf{b} \qquad(\mathbf{I}_N\in\mathbb{R}^{N\times N}: \text{identity matrix})\\
&=
\mathbf{C}^{-1}
\left(
    \mathbf{A}^\mathsf{T}\mathbf{A} + \lambda \mathbf{I}_N
\right)^{-1}
\mathbf{A}^\mathsf{T} \hat{\mathbf{b}},
\end{align*}
$$

where $\mathbf{A} \equiv \mathbf{B} \mathbf{T} \mathbf{C}^{-1}$ and $\hat{\mathbf{b}} \equiv \mathbf{B} \mathbf{b}$.

Let us perform the singular value decomposition to $\mathbf{A}$:

$$
\mathbf{A} = \mathbf{U}\mathbf{S}\mathbf{V}^\mathsf{T},
$$ (svd)

where $\mathbf{U}\in\mathbb{R}^{M\times r}$ and $\mathbf{V}\in\mathbb{R}^{N\times r}$ are the left and right singular vectors, respectively, and $\mathbf{S}\in\mathbb{R}^{r\times r}$ is a diagonal matrix with the singular values $\sigma_i$. Here, $r$ is the rank of $\mathbf{A}$ and $r\leq\min(M,N)$.

Hence, the solution $\mathbf{x}$ can be written as:

$$
\begin{align}
\mathbf{x}
&=
\mathbf{C}^{-1}
\left(
    \mathbf{VS U}^\mathsf{T} \mathbf{US V}^\mathsf{T} + \lambda \mathbf{I}_N
\right)^{-1}
\mathbf{VS}^\mathsf{T} \mathbf{U}^\mathsf{T} \hat{\mathbf{b}}\\
&=
\mathbf{C}^{-1}\mathbf{V}^{-\mathsf{T}}
\left(\mathbf{S}^2 + \lambda \mathbf{I}_r\right)^{-1}
\mathbf{V}^{-1}\mathbf{V S U}^\mathsf{T} \hat{\mathbf{b}} \qquad(\because \mathbf{S}^\mathsf{T} = \mathbf{S})\\
&=
\tilde{\mathbf{V}}
\left(\mathbf{I}_r + \lambda \mathbf{S}^{-2}\right)^{-1}
\mathbf{S}^{-1} \mathbf{U}^\mathsf{T} \hat{\mathbf{b}}
\qquad(
    \because \tilde{\mathbf{V}}\equiv \mathbf{C}^{-1}\mathbf{V},
    \quad\mathbf{V}^{-\mathsf{T}} = \mathbf{V}
)\\
&=
\tilde{\mathbf{V}}\mathbf{F}_\lambda\mathbf{S}^{-1}\mathbf{U}^\mathsf{T} \hat{\mathbf{b}}
\qquad\left(\because \mathbf{F}_\lambda\equiv \left(\mathbf{I}_r + \lambda \mathbf{S}^{-2}\right)^{-1}\right),
\end{align}
$$ (sol-matrix)

where $\tilde{\mathbf{V}}\in\mathbb{R}^{N\times r}$ is called the inverted solution basis and $\mathbf{F}_\lambda\in\mathbb{R}^{r\times r}$ is a diagonal matrix, the element $f_{\lambda, i}$ of which plays the role of a filter that suppresses the small singular values.
The diagonal elements of $\mathbf{F}_\lambda$ are given by:

$$
f_{\lambda, i} = \left(1 + \frac{\lambda}{\sigma_i^2}\right)^{-1},
$$ (filter)

where $\sigma_i$ is the $i$-th diagonal element of $\mathbf{S}$.

If matrices have the following forms:

$$
\begin{align*}
&\tilde{\mathbf{V}} =
    \begin{bmatrix}
        \tilde{\mathbf{v}}_1 & \tilde{\mathbf{v}}_2 & \cdots & \tilde{\mathbf{v}}_r
    \end{bmatrix},\\
&\mathbf{F}_\lambda =
    \begin{bmatrix}
        f_{\lambda, 1} & & & \\
        & f_{\lambda, 2} & & \\
        & & \ddots & \\
        & & & f_{\lambda, r}
    \end{bmatrix},\\
&\mathbf{S} =
    \begin{bmatrix}
        \sigma_1 & & & \\
        & \sigma_2 & & \\
        & & \ddots & \\
        & & & \sigma_r
    \end{bmatrix},\\
&\mathbf{U} =
    \begin{bmatrix}
        \mathbf{u}_1 & \mathbf{u}_2 & \cdots & \mathbf{u}_r
    \end{bmatrix},\\
\end{align*}
$$

then the {eq}`sol-matrix` can be calculated as follows:

$$
\begin{align}
\mathbf{x} &=
\begin{bmatrix}
    \tilde{\mathbf{v}}_1 & \tilde{\mathbf{v}}_2 & \cdots & \tilde{\mathbf{v}}_r
\end{bmatrix}
\begin{bmatrix}
    f_{\lambda, 1}/\sigma_1 & & & \\
    & f_{\lambda, 2}/\sigma_2 & & \\
    & & \ddots & \\
    & & & f_{\lambda, r}/\sigma_r
\end{bmatrix}
\begin{bmatrix}
    \mathbf{u}_1^\mathsf{T} \hat{\mathbf{b}} \\
    \mathbf{u}_2^\mathsf{T} \hat{\mathbf{b}} \\
    \vdots \\
    \mathbf{u}_r^\mathsf{T} \hat{\mathbf{b}}
\end{bmatrix}\\
&=
\begin{bmatrix}
    \tilde{\mathbf{v}}_1 & \tilde{\mathbf{v}}_2 & \cdots & \tilde{\mathbf{v}}_r
\end{bmatrix}
\begin{bmatrix}
    f_{\lambda, 1} \mathbf{u}_1^\mathsf{T} \hat{\mathbf{b}} / \sigma_1 \\
    f_{\lambda, 2} \mathbf{u}_2^\mathsf{T} \hat{\mathbf{b}} / \sigma_2 \\
    \vdots \\
    f_{\lambda, r} \mathbf{u}_r^\mathsf{T} \hat{\mathbf{b}} / \sigma_r
\end{bmatrix}\\
&=
\sum_{i=1}^r f_{\lambda, i} \frac{\mathbf{u}_i^\mathsf{T} \hat{\mathbf{b}}}{\sigma_i} \tilde{\mathbf{v}}_i.
\end{align}
$$ (sol-expansion)

The solution of the ill-posed linear equation {eq}`solution` can be expressed as a linear combination of the inverted solution basis vectors $\tilde{\mathbf{v}}_i$.
The weight of the $i$-th inverted solution basis vector is determined by the $f_{\lambda, i}$.
The larger the index $i$ is, the smaller the singular value $\sigma_i$ is and the much smaller the $f_{\lambda, i}$ is if $\lambda$ is sufficiently large.
Therefore, the noisy components of the solution are suppressed by the regularization parameter $\lambda$.

## Expression of the squared norm using the series expansion

For the sake of further discussion, let us derive the expression of the squared residual norm and the squared regularization norm using the series expansion {eq}`sol-expansion`.

Let each squared norm be $\rho$ and $\eta$ respectively:

$$
\rho \equiv \| \mathbf{T}\mathbf{x}_\lambda - \mathbf{b} \|_\mathbf{Q}^2,\quad
\eta \equiv \| \mathbf{x}_\lambda \|_\mathbf{H}^2,
$$

where $\mathbf{Q} = \mathbf{B}^\mathsf{T}\mathbf{B}$ and $\mathbf{H} = \mathbf{C}^\mathsf{T}\mathbf{C}$.

Firstly we transform $\mathbf{B}\mathbf{T}\tilde{\mathbf{V}}$ into the following form:

$$
\begin{split}
\mathbf{B}\mathbf{T}\tilde{\mathbf{V}}
&=
\mathbf{B}\mathbf{T}\mathbf{C}^{-1}\mathbf{V}
\qquad(\because \tilde{\mathbf{V}} = \mathbf{C}^{-1}\mathbf{V})\\
&=
\mathbf{U}\mathbf{S}\mathbf{V}^\mathsf{T}\mathbf{V}
\qquad(\because \mathbf{A} = \mathbf{B}\mathbf{T}\mathbf{C}^{-1} = \mathbf{U}\mathbf{S}\mathbf{V}^\mathsf{T})\\
&=\mathbf{U}\mathbf{S}.
\end{split}
$$ (BTV)

Additionally, using $\|\mathbf{a}\|_\mathbf{Q}^2 = \|\mathbf{B}\mathbf{a}\|^2$ ($\|\cdot\|$ is a Euclidean norm), the $\rho$ is expressed as follows:

$$
\begin{align}
\rho
&=
\| \mathbf{T}\mathbf{x}_\lambda - \mathbf{b} \|_\mathbf{Q}^2\\
&=
\| \mathbf{B}\mathbf{T}\tilde{\mathbf{V}}\mathbf{F}_\lambda\mathbf{S}^{-1}\mathbf{U}^\mathsf{T}\mathbf{b}
- \mathbf{B}\mathbf{b} \|^2\\
&=
\| \mathbf{U}\mathbf{S}\mathbf{F}_\lambda\mathbf{S}^{-1}\mathbf{U}^\mathsf{T}\hat{\mathbf{b}} - \hat{\mathbf{b}} \|^2
\qquad(\because \mathbf{B}\mathbf{T}\tilde{\mathbf{V}} = \mathbf{U}\mathbf{S})\\
&=
\| \mathbf{U}\mathbf{F}_\lambda\mathbf{U}^\mathsf{T}\hat{\mathbf{b}} - \hat{\mathbf{b}} \|^2
\qquad(\because \mathbf{S}\mathbf{F}_\lambda = \mathbf{F}_\lambda\mathbf{S})\\
&=
\| \mathbf{U}(\mathbf{F}_\lambda - \mathbf{I}_r)\mathbf{U}^\mathsf{T}\hat{\mathbf{b}} \|^2\\
&=
\| (\mathbf{F}_\lambda - \mathbf{I}_r)\mathbf{U}^\mathsf{T}\hat{\mathbf{b}} \|^2
\qquad(\because \| \mathbf{Uy} \|^2_2 = \mathbf{y}^\mathsf{T}\mathbf{U}^\mathsf{T}\mathbf{U}\mathbf{y} = \| \mathbf{y} \|^2,\; \text{where } \forall\mathbf{y}\in\mathbb{R}^r)\\
&=
\sum_{i=1}^r (f_{\lambda, i} - 1)^2 (\mathbf{u}_i^\mathsf{T}\hat{\mathbf{b}})^2.
\end{align}
$$

Also the $\eta$ is expressed as follows:

$$
\begin{align}
\eta
&=
\|\mathbf{x}_\lambda\|_\mathbf{H}^2 = \|\mathbf{C}\mathbf{x}_\lambda\|^2
\qquad (\because\mathbf{H} = \mathbf{C}^\mathsf{T}\mathbf{C})\\
&=
\| \mathbf{C}\tilde{\mathbf{V}}\mathbf{F}_\lambda\mathbf{S}^{-1}\mathbf{U}^\mathsf{T}\hat{\mathbf{b}} \|^2\\
&=
\| \mathbf{V}\mathbf{F}_\lambda \mathbf{S}^{-1}\mathbf{U}^\mathsf{T}\hat{\mathbf{b}} \|^2
\qquad (\because \tilde{\mathbf{V}} = \mathbf{C}^{-1}\mathbf{V})\\
&=
\| \mathbf{F}_\lambda \mathbf{S}^{-1}\mathbf{U}^\mathsf{T}\hat{\mathbf{b}} \|^2
\qquad(\because
\| \mathbf{Vy} \|^2
= \mathbf{y}^\mathsf{T}\mathbf{V}^\mathsf{T}\mathbf{V}\mathbf{y}
= \| \mathbf{y} \|^2,\; \text{where } \forall\mathbf{y}\in\mathbb{R}^r
)\\
&=
\sum_{i=1}^r \frac{f_{\lambda, i}^2}{\sigma_i^2} (\mathbf{u}_i^\mathsf{T}\hat{\mathbf{b}})^2.
\end{align}
$$

Lastly, we derive the series-expansion form of the vector: $\mathbf{T}\mathbf{x}_\lambda - \mathbf{b}$ as follows:

$$
\begin{align}
\mathbf{T}\mathbf{x}_\lambda - \mathbf{b}
&=
\mathbf{U}(\mathbf{F}_\lambda - \mathbf{I}_r)\mathbf{U}^\mathsf{T}\hat{\mathbf{b}}\\
&=
\sum_{i=1}^r (f_{\lambda, i} - 1) (\mathbf{u}_i^\mathsf{T}\hat{\mathbf{b}}) \mathbf{u}_i.
\end{align}
$$

```{footbibliography} ../references.bib
```
