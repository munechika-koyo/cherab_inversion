import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from cherab.inversion import Derivative

# R-Phi grids
radius = np.linspace(0.1, 1, 30)
angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)

# Create voxels center coordinates in (x, y, z)
rr, pp = np.meshgrid(radius, angles)
centers = np.zeros((30, 180, 1, 3))
centers[:, :, 0, 0] = (rr * np.cos(pp)).T
centers[:, :, 0, 1] = (rr * np.sin(pp)).T
centers[:, :, 0, 2] = 0

# Create a 2-D profile
indices = np.ndindex(centers.shape[:3])
profile = np.zeros(centers.shape[:3])

CENTER = 0.6
WIDTH = 0.05
PHASE = -np.pi / 4
for ir, ip, iz in indices:
    profile[ir, ip, iz] = (1 + np.cos(angles[ip] * 4 + PHASE)) * np.exp(
        -((radius[ir] - CENTER) ** 2) / WIDTH
    )


# Operate derivative
deriv = Derivative(
    centers,
    np.arange(np.prod(centers.shape[:3]), dtype=np.int32).reshape(centers.shape[:3]),
)
dmat_r = deriv.matrix_along_axis(0, diff_type="forward", boundary="dirichlet")
dmat_p = deriv.matrix_along_axis(1, diff_type="forward", boundary="periodic")

profile_dr = dmat_r @ profile.ravel()
profile_dp = dmat_p @ profile.ravel()
profile_dr = profile_dr.reshape(centers.shape[:3])
profile_dp = profile_dp.reshape(centers.shape[:3])

# Plot the profiles
fig = plt.figure(layout="constrained")
gs = GridSpec(1, 3, figure=fig)
for i, (f, title) in enumerate(
    zip(
        [profile, profile_dr, profile_dp],
        ["Original", "$R$-derivative", "$\\phi$-derivative"],
        strict=True,
    )
):
    ax = fig.add_subplot(gs[i], projection="polar")
    ax.grid(False)
    ax.pcolormesh(
        angles,
        radius,
        f[..., 0],
        cmap="RdBu_r",
        vmax=np.amax(np.abs(f)),
        vmin=-np.amax(np.abs(f)),
    )
    ax.set_rticks([])
    ax.set_thetagrids([])
    ax.set_title(title)

plt.show()
