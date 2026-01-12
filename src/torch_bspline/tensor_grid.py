import torch


class TensorGrid:
    def __init__(self, xs, ys=None, x_varies_first=False):
        """
        Construct a TensorGrid with the given x-coordinates and y-coordinates.
        This class exists mainly because a TensorBasis of functions can be
        evaluated more efficiently on TensorGrids of points than on a simple
        array of (x, y) coordinates.
        """
        self.xs = xs
        self.ys = ys
        self.x_varies_first = x_varies_first
        self.dtype = xs.dtype
        self.device = xs.device
        if ys is None:
            self.shape = (len(xs) ** 2, 2)
        else:
            self.shape = (len(xs) * len(ys), 2)

    def full(self):
        """
        Return a `torch.Tensor` containing all the points represented by this
        grid. The ordering of the points depends on `self.x_varies_first`
        (defaults to False). If this grid has `len(xs) == m` and `len(ys) == n`
        then the output will have shape `(m*n) × 2`, where

        x_varies_first == False: XY[n * i + j, :] = [xs[i], ys[j]]
        x_varies_first == True : XY[i + m * j, :] = [xs[i], ys[j]]
        """
        x = self.xs
        y = self.ys if self.ys is not None else x
        if self.x_varies_first:
            return torch.cartesian_prod(y, x).flip(1)
        else:
            return torch.cartesian_prod(x, y)

    @staticmethod
    def from_tensor(XY):
        """
        If `XY` is a list of points constructed from a tensor grid, turn it
        back into a `TensorGrid` representation. Note that this function always
        checks that output.full() == XY, and fails with an AssertionError if
        this isn't the case.
        """
        assert XY.shape[1] == 2, "TensorGrid.from_tensor: input must be mn × 2"
        mn = XY.shape[0]
        X = XY[:, 0]
        Y = XY[:, 1]

        x_varies_first = (mn > 1 and X[1] != X[0]).item()
        if x_varies_first:
            X, Y = Y, X

        x0 = X[0]
        n = 0
        while n < mn and X[n] == x0:
            n += 1

        assert mn % n == 0, f"TensorGrid.from_tensor: found len(xs) = {n} which isn't a divisor of {mn}"
        # Then, the first `n` y-values form the basis in the y-direction
        # And the x-basis is given by a strided slice of the x-values
        ys = Y[:n]
        xs = X[::n]

        if x_varies_first:
            xs, ys = ys, xs

        if len(xs) == len(ys) and all(ys == xs):
            ys = None

        grid = TensorGrid(xs, ys, x_varies_first)
        assert all(grid.full().ravel() == XY.ravel()), "TensorGrid.from_tensor: input was not of tensor product form"
        return grid

    @property
    def grid_shape(self):
        xs = self.xs
        ys = self.ys if self.ys is not None else xs
        return (xs.numel(), ys.numel())

    def __str__(self):
        m = self.xs.numel()
        n = m if self.ys is None else self.ys.numel()
        return f"TensorGrid({m}×{n})"

    def __repr__(self):
        return f"TensorGrid({self.xs}, {self.ys})"

    def to(self, device):
        xs = self.xs.to(device)
        ys = self.ys.to(device) if self.ys is not None else None
        return TensorGrid(xs, ys, self.x_varies_first)
