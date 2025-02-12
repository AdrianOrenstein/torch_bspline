# Copyright 2025, Hugo Melchers, Eindhoven University of Technology

# 3rd Party
import torch
from torch import nn

# Local
from torch_bspline.bspline import BSpline
from torch_bspline.tensor_grid import TensorGrid


class TensorBasis(nn.Module):

    dim = 2


    def __init__(self, f_x:BSpline, f_y:BSpline=None):
        super().__init__()
        self.f_x = f_x
        self.f_y = f_y


    def forward(self, points:torch.Tensor|TensorGrid) -> torch.Tensor:

        f_x = self.f_x
        f_y = self.f_y or f_x

        out_eval = None

        if isinstance(points, TensorGrid):
            # Apparently it's easier to compute on this
            # user-defined grid than on the x,y coords
            f = f_x(points.xs)

            if self.f_y is None and points.ys is None:
                # This implies that the basis and points
                # are the same in both x and y
                g = f
            else:
                # Evaluate the basis in the y direction
                ys = points.ys if points.ys is not None else points.xs
                g = f_y(ys)

            out_eval = self.tensor_product(f, g, points.x_varies_first)
        else:
            X = points[:, 0]
            Y = points[:, 1]
            f = f_x(X)
            g = f_y(Y)
            out_eval = self.outer_product(f, g)

        return out_eval

    def gradient(self, points):

        out_grad_eval = None

        f_x = self.f_x
        f_y = self.f_y or f_x

        if isinstance(points, TensorGrid):

            f = f_x(points.xs)
            df = f_x.derivative(points.xs)

            if self.f_y is None and points.ys is None:
                g = f
                dg = df
            else:
                ys = points.ys if points.ys is not None else points.xs
                g = f_y(ys)
                dg = f_y.derivative(ys)

            out_grad_eval = torch.stack(
                (
                    self.tensor_product(df, g, points.x_varies_first),
                    self.tensor_product(f, dg, points.x_varies_first),
                ),
                -1,
            )
        else:
            X = points[:, 0]
            Y = points[:, 1]

            f = f_x(X)
            g = f_y(Y)

            df = f_x.derivative(X)
            dg = f_y.derivative(Y)

            out_grad_eval = torch.stack(
                (self.outer_product(df, g), self.outer_product(f, dg)), -1
            )

        return out_grad_eval


    def hessian(self, points):
        f_x = self.f_x
        f_y = self.f_y or f_x
        if isinstance(points, TensorGrid):
            f = f_x(points.xs)
            df = f_x.derivative(points.xs)
            ddf = f_x.derivative(points.xs, 2)
            if self.f_y is None and points.ys is None:
                g, dg, ddg = f, df, ddf
            else:
                ys = points.ys if points.ys is not None else points.xs
                g = f_y(ys)
                dg = f_y.derivative(ys)
                ddg = f_y.derivative(ys, 2)
            return torch.stack(
                (
                    self.tensor_product(ddf, g, points.x_varies_first),
                    self.tensor_product(df, dg, points.x_varies_first),
                    self.tensor_product(f, ddg, points.x_varies_first),
                ),
                -1,
            )
        else:
            X = points[:, 0]
            Y = points[:, 1]
            f = f_x(X)
            g = f_y(Y)
            df = f_x.derivative(X)
            dg = f_y.derivative(Y)
            ddf = f_x.derivative(X, 2)
            ddg = f_y.derivative(Y, 2)
            return torch.stack(
                (
                    self.outer_product(ddf, g),
                    self.outer_product(df, dg),
                    self.outer_product(f, ddg),
                ),
                -1,
            )


    # Laplacian of RBF interpolated GRF
    def laplacian(self, points:torch.Tensor):
        
        out_laplacian = None

        f_x = self.f_x
        f_y = self.f_y or f_x
        if isinstance(points, TensorGrid):
            f = f_x(points.xs)
            ddf = f_x.derivative(points.xs, 2)
            if self.f_y is None and points.ys is None:
                g, ddg = f, ddf
            else:
                ys = points.ys if points.ys is not None else points.xs
                g = f_y(ys)
                ddg = f_y.derivative(ys, 2)
            out_laplacian = (
                    self.tensor_product(ddf, g, points.x_varies_first)
                +   self.tensor_product(f, ddg, points.x_varies_first)
            ) 
        else:
            X = points[:, 0]
            Y = points[:, 1]
            f = f_x(X)
            g = f_y(Y)
            ddf = f_x.derivative(X, 2)
            ddg = f_y.derivative(Y, 2)
            out_laplacian = (
                    self.outer_product(ddf, g)
                +   self.outer_product(f, ddg)
            )

        return out_laplacian

    @staticmethod
    def outer_product(a:torch.Tensor, b:torch.Tensor) -> torch.Tensor:
        # a should be n × f
        # b should be n × g
        # output should then be n × (f*g)
        n = a.shape[0]
        f = a.shape[1]
        g = b.shape[1]

        # TODO: confirm the ordering of the reshaped tensor
        return (
            a.reshape(n, f, 1) * b.reshape(n, 1, g)
        ).reshape(n, f * g)


    @staticmethod
    def tensor_product(
        a:torch.Tensor,
        b:torch.Tensor,
        x_varies_first:bool
    ) -> torch.Tensor:
        # a should be m × f
        # b should be n × g
        # output should then be mn × fg

        m, f = a.shape
        n, g = b.shape

        if x_varies_first:
            # when reshaping: last index varies first. We therefore now want the last indices to be m, and g
            a = a.reshape(1, m, f, 1)
            b = b.reshape(n, 1, 1, g)
        else:
            a = a.reshape(m, 1, f, 1)
            b = b.reshape(1, n, 1, g)
        return (a * b).reshape(m * n, f * g)


    def __len__(self):
        f_x = self.f_x
        f_y = self.f_y or f_x
        return len(f_x) * len(f_y)
