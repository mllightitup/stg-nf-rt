# The based unit of graph convolutional networks., based on awesome previous work by https://github.com/yysijie/st-gcn
import torch
import torch.nn as nn

from STG_NF.models.STG_NF.modules_pose import InvertibleConv1x1


class InvConvTemporalGraphical(nn.Module):
    r"""The basic module for applying a graph convolution.
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self, in_channels, LU_decomposed, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size  # typically 2
        self.conv = InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)

    def forward(self, x, A, logdetA, logdet=None, reverse=False):
        if not reverse:
            # Forward branch: apply invertible convolution.
            x, dlogdet = self.conv(x, logdetA, reverse=False)
            n, kc, t, v = x.size()
            c = kc // self.kernel_size
            # Split channel dim into spatial kernels and sub-channels.
            x = x.view(n, self.kernel_size, c, t, v)
            # Permute for matrix multiplication:
            # shape: (n, c, t, kernel_size * v)
            x_perm = x.permute(0, 2, 3, 1, 4).reshape(n, c, t, -1)
            # Reshape A from (kernel_size, v, v) to (kernel_size*v, v)
            A_reshaped = A.view(-1, v)
            # Apply spatial transformation.
            x_transformed = x_perm @ A_reshaped  # shape: (n, c, t, v)
            # Concatenate the first kernel slice with the transformed result.
            x_cat = torch.cat((x[:, 0], x_transformed), dim=1)  # shape: (n, 2*c, t, v)
            # Reshape back to (n, kernel_size, c, t, v)
            out = x_cat.view(n, self.kernel_size, c, t, v)
            total_logdet = (
                (dlogdet + logdetA[1]) if logdet is None else logdet + logdetA[1]
            )
            return out, total_logdet
        else:
            # Reverse branch: invert the spatial transformation.
            # Invert the second kernel matrix.
            A1_inv = A[1].inverse()
            x0 = x[:, 0]  # first slice: shape (n, c, t, v)
            # Undo the transformation: subtract the effect of A[0] and invert A[1]
            x1 = (x[:, 1] - (x0 @ A[0])) @ A1_inv
            # Concatenate and restore the original shape.
            x_rev = torch.cat((x0, x1), dim=1).view_as(x)
            # Apply the invertible convolution in reverse mode.
            z, dlogdet = self.conv(x_rev, logdetA, reverse=True)
            total_logdet = (
                (dlogdet - logdetA[1]) if logdet is None else logdet - logdetA[1]
            )
            return z, total_logdet
