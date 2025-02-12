
# Standard
from typing import Tuple
# 3rd Party
import torch
import numpy as np
import matplotlib.pyplot as plt

# Local
from torch_bspline import TensorGrid
from torch_bspline import BSpline
from torch_bspline import TensorBasis
from torch_bspline import BSplineFunctions

params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

def visualize_function(*,
    num_x_bases:int,
    num_y_bases:int,
    weights:torch.Tensor,
    cmap:str='bwr',
    dtype:torch.dtype=torch.float64,
    fontsize:int=40,
    num_contour_levels:int=15,
    poly_order:int=3,
    title='Func Value'
) -> Tuple[BSplineFunctions, TensorGrid]:

    assert weights.shape == (num_x_bases*num_y_bases,)

    x_basis = BSpline.uniform(
        lims=(0,1),
        n_segments=num_x_bases - poly_order,
        degree=poly_order,
        dtype=dtype
    )

    y_basis = BSpline.uniform(
        lims=(0,1),
        n_segments=num_y_bases - poly_order,
        degree=poly_order,
        dtype=dtype
    )

    xy_basis = TensorBasis(x_basis, y_basis)

    xy_grid = TensorGrid(
        xs = torch.linspace(0,1,50, dtype=dtype),
        ys = torch.linspace(0,1,100, dtype=dtype),
        x_varies_first=True
    )
    X, Y  = np.meshgrid(xy_grid.xs.numpy(), xy_grid.ys.numpy())

    fig = plt.figure()
    fig.show()

    title = f'Function Value'

    f = BSplineFunctions(xy_basis, weights)
    Z = f(xy_grid).reshape(X.shape)
    ax = fig.add_subplot(111)

    cs = ax.contourf(X, Y, Z, levels=num_contour_levels, cmap=cmap)

    ax.set_title(title, fontsize=fontsize)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(cs, cax=cbar_ax)
    fig.canvas.draw()
    plt.show()

    return f, xy_grid

if __name__ == '__main__':

    f, xy_grid = visualize_function(
        weights=torch.ones((25,), dtype=torch.float64),
        num_x_bases=5,
        num_y_bases=5
    )

    print("Laplacian should be everywhere zero...")
    print(f"Laplacian(f).max() = {f.laplacian(xy_grid).max()}")
    print(f"Laplacian(f).min() = {f.laplacian(xy_grid).min()}")

    f, xy_grid = visualize_function(
        weights=torch.randn((25,), dtype=torch.float64),
        num_x_bases=5,
        num_y_bases=5
    )

    print("Laplacian should NOT be everywhere zero...")
    print(f"Laplacian(f).max() = {f.laplacian(xy_grid).max()}")
    print(f"Laplacian(f).min() = {f.laplacian(xy_grid).min()}")

    xy = torch.rand((100,2), dtype=torch.float64)

    print("Laplacian should NOT be everywhere zero...")
    print(f"Laplacian(f).max() = {f.laplacian(xy).max()}")
    print(f"Laplacian(f).min() = {f.laplacian(xy).min()}")
