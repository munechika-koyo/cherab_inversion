import numpy as np
import ultraplot as uplt

from cherab.inversion import Derivative

# X-Y grids
x = np.linspace(-1, 1, 60)
y = np.linspace(-1, 1, 60)

# Create voxels center coordinates in (x, y, z)
xx, yy = np.meshgrid(x, y)
centers = np.zeros((x.size, y.size, 1, 3))
centers[:, :, 0, 0] = xx.T
centers[:, :, 0, 1] = yy.T
centers[:, :, 0, 2] = 0

# Create a 2-D profile
indices = np.ndindex(centers.shape[:3])
profile = np.zeros(centers.shape[:3])

CENTER = 0.6
WIDTH = 0.05
PHASE = -np.pi / 4

profile = np.zeros(centers.shape[:3])
radius = np.hypot(xx, yy).T
angles = np.arctan2(yy, xx).T
profile[..., 0] = (1 + np.cos(angles * 4 + PHASE)) * np.exp(-((radius - CENTER) ** 2) / WIDTH)


def scalar_func(x, y):
    """Scalar function to be used in the derivative calculation."""
    return np.hypot(x, y)


# Operate derivative
deriv = Derivative(centers)
dmat_para, dmat_perp = deriv.matrix_gradient(scalar_func)
profile_para = dmat_para @ profile[deriv.grid_map > -1]
profile_perp = dmat_perp @ profile[deriv.grid_map > -1]
profile_para = profile_para.reshape(deriv.grid_map.shape)
profile_perp = profile_perp.reshape(deriv.grid_map.shape)

# Calculate Scalar function values and its gradient
fvals = scalar_func(xx, yy)
grads = np.gradient(fvals, centers[:, 0, 0, 0], centers[0, :, 0, 1])

# Calculate the gradient of the profile
grad_profile = np.gradient(profile[..., 0], centers[:, 0, 0, 0], centers[0, :, 0, 1])
grad_profile = np.hypot(grad_profile[0], grad_profile[1])[:, :, None]

# Plot the profiles
fig, axes = uplt.subplots(nrows=2, ncols=3)
for ax, f, title in zip(
    axes,
    [
        fvals[..., np.newaxis],
        [grads[0], grads[1]],
        [-grads[1], grads[0]],
        profile,
        profile_para,
        profile_perp,
    ],
    [
        "$f(x, y)$",
        "$\\nabla f(x, y)$",
        "Perp of $\\nabla f(x, y)$",
        "Profile",
        "$\\nabla_\\parallel$-derivative",
        "$\\nabla_\\perp$-derivative",
    ],
    strict=False,
):
    if "\\nabla f(x, y)" not in title:
        ax.pcolormesh(
            centers[:, 0, 0, 0],
            centers[0, :, 0, 1],
            f[..., 0].T,
            discrete=False,
            vmax=np.max(np.abs(f)),
            vmin=-np.max(np.abs(f)),
        )
    else:
        ax.quiver(
            centers[:, 0, 0, 0][::8],
            centers[0, :, 0, 1][::8],
            f[0][::8, ::8].T,
            f[1][::8, ::8].T,
            scale=10,
        )
    ax.format(title=title)

axes.format(
    aspect="equal",
    xlabel="X",
    ylabel="Y",
    tickdir="in",
)

fig.show()
