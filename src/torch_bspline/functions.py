# Copyright 2025, Hugo Melchers, Eindhoven University of Technology
from __future__ import annotations

# 3rd Party
import torch
from torch import nn

# Local
from torch_bspline.tensor_grid import TensorGrid
from torch_bspline.tensor_basis import TensorBasis


class BSplineFunctions(nn.Module):

    def __init__(self, basis, weights, offset=None):
        super().__init__()
        """
        Construct a set of functions using different weights in the same basis.
        If `basis` consists of n functions, then `weights` should be a `n×m`
        array and the resulting `Functions` object will represent `m` functions.
        Note that the weights are not trainable parameters.
        """
        self.basis = basis
        self.register_buffer("weights", weights)
        self.offset = offset
        self.dim = self.basis.dim


    def is_singleton(self):
        return self.weights.ndim == 1


    def get_weights_2d(self):
        if self.is_singleton():
            return self.weights[:, None]
        else:
            return self.weights


    def maybe_flatten(self, output):
        if self.is_singleton():
            return output.squeeze(1)
        else:
            return output


    def forward(self, points):
        weights = self.get_weights_2d()

        if isinstance(points, TensorGrid) and isinstance(self.basis, TensorBasis):
            # Special path: for tensor basis on tensor grid, we do not need to first evaluate the entire tensor product
            # Instead, we evaluate the components, and then do tensor product + linear combinations in one step
            f_x = self.basis.f_x
            f_y = self.basis.f_y if self.basis.f_y is not None else f_x

            p_x = points.xs
            p_y = points.ys if points.ys is not None else p_x
            
            W = weights.reshape(len(f_x), len(f_y), -1)

            fxs = f_x(p_x)
            fys = fxs if f_y is f_x and p_y is p_x else f_y(p_y)

            output = torch.einsum("xf,fgF,yg->xyF", fxs, W, fys)
            
            if points.x_varies_first:
                output = output.permute(1, 0, 2)
            output = output.reshape(-1, output.shape[2])
        else:
            F = self.basis(points)
            output = torch.einsum("xb,bf->xf", F, weights)
        output = self.maybe_flatten(output)
        if self.offset:
            return output + self.offset
        else:
            return output


    def derivative(self, points, k=1):
        # constant term drops out in the derivatives
        F = self.basis.derivative(points, k)
        return self.maybe_flatten(torch.einsum("xb,bf->xf", F, self.get_weights_2d()))


    def gradient(self, points):
        weights = self.get_weights_2d()
        if isinstance(points, TensorGrid) and isinstance(self.basis, TensorBasis):
            # Special path: for tensor basis on tensor grid, we do not need to first evaluate the entire tensor product
            # Instead, we evaluate the components, and then do tensor product + linear combinations in one step
            f_x = self.basis.f_x
            f_y = self.basis.f_y if self.basis.f_y is not None else f_x
            p_x = points.xs
            p_y = points.ys if points.ys is not None else p_x
            W = weights.reshape(len(f_x), len(f_y), -1)
            fxs = f_x(p_x)
            fys = fxs if f_y is f_x and p_y is p_x else f_y(p_y)
            d_fxs = f_x.derivative(p_x)
            d_fys = d_fxs if f_y is f_x and p_y is p_x else f_y.derivative(p_y)
            output_x = torch.einsum("xf,fgF,yg->xyF", d_fxs, W, fys)
            output_y = torch.einsum("xf,fgF,yg->xyF", fxs, W, d_fys)
            if points.x_varies_first:
                output_x = output_x.permute(1, 0, 2)
                output_y = output_y.permute(1, 0, 2)
            output_x = output_x.reshape(-1, output_x.shape[2])
            output_y = output_y.reshape(-1, output_y.shape[2])
            return self.maybe_flatten(torch.stack((output_x, output_y), dim=-1))
        else:
            F = self.basis.gradient(points)
            return self.maybe_flatten(torch.einsum("xbi,bf->xfi", F, weights))


    def laplacian(self, points):
        weights = self.get_weights_2d()
        if isinstance(points, TensorGrid) and isinstance(self.basis, TensorBasis):
            # Special path: for tensor basis on tensor grid, we do not need to first evaluate the entire tensor product
            # Instead, we evaluate the components, and then do tensor product + linear combinations in one step
            f_x = self.basis.f_x
            f_y = self.basis.f_y if self.basis.f_y is not None else f_x

            p_x = points.xs
            p_y = points.ys if points.ys is not None else p_x

            W = weights.reshape(len(f_x), len(f_y), -1)

            fxs = f_x(p_x)
            fys = fxs if f_y is f_x and p_y is p_x else f_y(p_y)

            d2_fxs = f_x.derivative(p_x, 2)
            d2_fys = d2_fxs if f_y is f_x and p_y is p_x else f_y.derivative(p_y, 2)

            output_xx = torch.einsum("xf,fgF,yg->xyF", d2_fxs, W, fys)
            output_yy = torch.einsum("xf,fgF,yg->xyF", fxs, W, d2_fys)
            
            if points.x_varies_first:
                output_xx = output_xx.permute(1, 0, 2)
                output_yy = output_yy.permute(1, 0, 2)

            output_xx = output_xx.reshape(-1, output_xx.shape[2])
            output_yy = output_yy.reshape(-1, output_yy.shape[2])

            return (
                    output_xx
                +   output_yy
            )
        else:
            F = self.basis.laplacian(points=points)
            return self.maybe_flatten(torch.einsum("xb,bf->xf", F, weights))



    def hessian(self, points):
        weights = self.get_weights_2d()
        if isinstance(points, TensorGrid) and isinstance(self.basis, TensorBasis):
            # Special path: for tensor basis on tensor grid, we do not need to first evaluate the entire tensor product
            # Instead, we evaluate the components, and then do tensor product + linear combinations in one step
            f_x = self.basis.f_x
            f_y = self.basis.f_y if self.basis.f_y is not None else f_x
            p_x = points.xs
            p_y = points.ys if points.ys is not None else p_x
            W = weights.reshape(len(f_x), len(f_y), -1)
            fxs = f_x(p_x)
            fys = fxs if f_y is f_x and p_y is p_x else f_y(p_y)
            d_fxs = f_x.derivative(p_x)
            d_fys = d_fxs if f_y is f_x and p_y is p_x else f_y.derivative(p_y)
            d2_fxs = f_x.derivative(p_x, 2)
            d2_fys = d2_fxs if f_y is f_x and p_y is p_x else f_y.derivative(p_y, 2)
            output_xx = torch.einsum("xf,fgF,yg->xyF", d2_fxs, W, fys)
            output_xy = torch.einsum("xf,fgF,yg->xyF", d_fxs, W, d_fys)
            output_yy = torch.einsum("xf,fgF,yg->xyF", fxs, W, d2_fys)
            if points.x_varies_first:
                output_xx = output_xx.permute(1, 0, 2)
                output_xy = output_xy.permute(1, 0, 2)
                output_yy = output_yy.permute(1, 0, 2)
            output_xx = output_xx.reshape(-1, output_xx.shape[2])
            output_xy = output_xy.reshape(-1, output_xy.shape[2])
            output_yy = output_yy.reshape(-1, output_yy.shape[2])
            return self.maybe_flatten(
                torch.stack((output_xx, output_xy, output_yy), dim=-1)
            )
        else:
            F = self.basis.hessian(points)
            return self.maybe_flatten(torch.einsum("xbi,bf->xfi", F, weights))


    def __getitem__(self, ids: int | torch.Tensor) -> BSplineFunctions:
        """
        Selects a smaller number of functions from this Functions object, based
        on the indices `ids`. Note that `ids` should be a single integer or a
        Tensor of integers.
        """
        return BSplineFunctions(self.basis, self.weights[:, ids], self.offset)


    def __len__(self):
        if self.is_singleton():
            return None
        else:
            return self.weights.shape[1]
