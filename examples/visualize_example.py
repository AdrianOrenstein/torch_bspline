# 3rd Party
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch_bspline.src.torch_bspline.bspline import BSpline
from torch_bspline.src.torch_bspline.functions import BSplineFunctions
from torch_bspline.src.torch_bspline.tensor_basis import TensorBasis

# Local
from torch_bspline.src.torch_bspline.tensor_grid import TensorGrid

params = {"mathtext.default": "regular"}
plt.rcParams.update(params)


def visualize_basis(
    *,
    cmap: str = "bwr",
    dtype: torch.dtype = torch.float64,
    fontsize: int = 40,
    num_x_bases: int = 3,
    num_y_bases: int = 3,
    num_contour_levels: int = 15,
    poly_order: int = 0,
    title="Func Value",
) -> None:
    x_basis = BSpline.uniform(lims=(0, 1), n_segments=num_x_bases - poly_order, degree=poly_order, dtype=dtype)
    y_basis = BSpline.uniform(lims=(0, 1), n_segments=num_y_bases - poly_order, degree=poly_order, dtype=dtype)

    xy_basis = TensorBasis(x_basis, y_basis)

    xy_grid = TensorGrid(
        xs=torch.linspace(0, 1, 50, dtype=dtype), ys=torch.linspace(0, 1, 100, dtype=dtype), x_varies_first=True
    )
    X, Y = np.meshgrid(xy_grid.xs.numpy(), xy_grid.ys.numpy())

    fig = plt.figure()
    fig.show()
    for i in range(num_x_bases):
        for j in range(num_y_bases):
            title = f"$N_\u007b {i}, {j} \u007d$"

            weights = torch.zeros((num_x_bases * num_y_bases, 1), dtype=dtype)
            weights[i * num_y_bases + j] = 1.0

            f = BSplineFunctions(xy_basis, weights)
            Z = f(xy_grid).reshape(X.shape)
            ax = fig.add_subplot(111)

            cs = ax.contourf(X, Y, Z, levels=num_contour_levels, cmap=cmap)

            ax.set_title(title, fontsize=fontsize)
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(cs, cax=cbar_ax)
            fig.canvas.draw()
            plt.pause(1.0)
            fig.clf()

    return


if __name__ == "__main__":
    visualize_basis()
