# # STG-NF modules, based on awesome previous work by https://github.com/y0ast/Glow-PyTorch
#
#
# import math
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from STG_NF.models.STG_NF.utils import compute_same_pad, split_feature
#
#
# def gaussian_p(mean, logs, x):
#     """
#     lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
#             k = 1 (Independent)
#             Var = logs ** 2
#     """
#     var = torch.exp(2.0 * logs) + 1e-6
#     c = math.log(2 * math.pi)
#     return -0.5 * (2.0 * logs + ((x - mean) ** 2) / var + c)
#
#
# def gaussian_likelihood(mean, logs, x):
#     p = gaussian_p(mean, logs, x)
#     return torch.sum(p, dim=[1, 2, 3])
#
#
# def gaussian_sample(mean, logs, temperature=1):
#     # Sample from Gaussian with temperature
#     temperature = temperature or 1
#     return torch.normal(mean, torch.exp(logs) * temperature)
#
#
# def squeeze2d(input, factor):
#     if factor == 1:
#         return input
#     B, C, T, V = input.size()
#     assert T % factor == 0, "T modulo factor is not 0"
#     x = input.view(B, C, T // factor, factor, V)
#     x = x.permute(0, 1, 3, 2, 4).contiguous()
#     return x.view(B, C * factor, T // factor, V)
#
#
# def unsqueeze2d(input, factor):
#     if factor == 1:
#         return input
#     factor2 = factor**2
#     B, C, T, V = input.size()
#     assert C % factor2 == 0, "C modulo factor^2 is not 0"
#     x = input.view(B, C // factor2, factor, T, V)
#     x = x.permute(0, 1, 4, 2, 3).contiguous()
#     return x.view(B, C // factor2, T * factor, V)
#
#
# class _ActNorm(nn.Module):
#     """
#     Activation Normalization
#     Initialize the bias and scale with a given minibatch,
#     so that the output per-channel have zero mean and unit variance for that.
#
#     After initialization, `bias` and `logs` will be trained as parameters.
#     """
#
#     def __init__(self, num_features, scale=1.0):
#         super().__init__()
#         size = [1, num_features, 1, 1]
#         self.bias = nn.Parameter(torch.zeros(*size))
#         self.logs = nn.Parameter(torch.zeros(*size))
#         self.num_features = num_features
#         self.scale = scale
#         self.inited = False
#
#     def initialize_parameters(self, input):
#         if not self.training:
#             raise ValueError("In Eval mode, but ActNorm not initialized")
#         with torch.no_grad():
#             bias = -torch.mean(input, dim=[0, 2, 3], keepdim=True)
#             var = torch.mean((input + bias) ** 2, dim=[0, 2, 3], keepdim=True)
#             logs = torch.log(self.scale / (torch.sqrt(var) + 1e-6))
#             self.bias.data.copy_(bias)
#             self.logs.data.copy_(logs)
#             self.inited = True
#
#     def _center(self, input, reverse=False):
#         return input - self.bias if reverse else input + self.bias
#
#     def _scale(self, input, logdet=None, reverse=False):
#         if reverse:
#             input = input * torch.exp(-self.logs)
#         else:
#             input = input * torch.exp(self.logs)
#         if logdet is not None:
#             b, c, h, w = input.shape
#             dlogdet = torch.sum(self.logs) * h * w
#             logdet = logdet - dlogdet if reverse else logdet + dlogdet
#         return input, logdet
#
#     def forward(self, input, logdet=None, reverse=False):
#         self._check_input_dim(input)
#         if not self.inited:
#             self.initialize_parameters(input)
#         if reverse:
#             input, logdet = self._scale(input, logdet, reverse)
#             input = self._center(input, reverse)
#         else:
#             input = self._center(input, reverse)
#             input, logdet = self._scale(input, logdet, reverse)
#         return input, logdet
#
#
# class ActNorm2d(_ActNorm):
#     def __init__(self, num_features, scale=1.0):
#         super().__init__(num_features, scale)
#
#     def _check_input_dim(self, input):
#         assert len(input.size()) == 4, "Input must be 4D"
#         assert input.size(1) == self.num_features, (
#             f"Expected {self.num_features} channels but got {input.size(1)}"
#         )
#
#
# class LinearZeros(nn.Module):
#     def __init__(self, in_channels, out_channels, logscale_factor=3):
#         super().__init__()
#         self.linear = nn.Linear(in_channels, out_channels)
#         with torch.no_grad():
#             self.linear.weight.zero_()
#             self.linear.bias.zero_()
#         self.logscale_factor = logscale_factor
#         self.logs = nn.Parameter(torch.zeros(out_channels))
#
#     def forward(self, input):
#         output = self.linear(input)
#         return output * torch.exp(self.logs * self.logscale_factor)
#
#
# class Conv2d(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size=(3, 1),
#         stride=(1, 1),
#         padding="same",
#         do_actnorm=True,
#         weight_std=0.05,
#     ):
#         super().__init__()
#         if padding == "same":
#             padding = compute_same_pad(kernel_size, stride)
#         elif padding == "valid":
#             padding = 0
#
#         self.conv = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride,
#             padding,
#             bias=(not do_actnorm),
#         )
#         with torch.no_grad():
#             self.conv.weight.normal_(0.0, weight_std)
#             if not do_actnorm:
#                 self.conv.bias.zero_()
#         self.do_actnorm = do_actnorm
#         if do_actnorm:
#             self.actnorm = ActNorm2d(out_channels)
#
#     def forward(self, input):
#         x = self.conv(input.squeeze())
#         if self.do_actnorm:
#             x, _ = self.actnorm(x)
#         return x
#
#
# class Conv2dZeros(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size=(3, 1),
#         stride=(1, 1),
#         padding="same",
#         logscale_factor=3,
#     ):
#         super().__init__()
#         if padding == "same":
#             padding = compute_same_pad(kernel_size, stride)
#         elif padding == "valid":
#             padding = 0
#
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#         with torch.no_grad():
#             self.conv.weight.zero_()
#             self.conv.bias.zero_()
#         self.logscale_factor = logscale_factor
#         self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))
#
#     def forward(self, input):
#         output = self.conv(input)
#         return output * torch.exp(self.logs * self.logscale_factor)
#
#
# class Permute2d(nn.Module):
#     def __init__(self, num_channels, shuffle):
#         super().__init__()
#         self.num_channels = num_channels
#         self.indices = torch.arange(num_channels - 1, -1, -1, dtype=torch.long)
#         self.indices_inverse = torch.argsort(self.indices)
#         if shuffle:
#             self.reset_indices()
#
#     def reset_indices(self):
#         shuffle_idx = torch.randperm(self.indices.size(0))
#         self.indices = self.indices[shuffle_idx]
#         self.indices_inverse = torch.argsort(self.indices)
#
#     def forward(self, input, reverse=False):
#         assert len(input.size()) == 4, "Input must be 4D"
#         return (
#             input[:, self.indices_inverse, :, :]
#             if reverse
#             else input[:, self.indices, :, :]
#         )
#
#
# class Split2d(nn.Module):
#     def __init__(self, num_channels):
#         super().__init__()
#         self.conv = Conv2dZeros(num_channels // 2, num_channels)
#
#     def split2d_prior(self, z):
#         h = self.conv(z)
#         return split_feature(h, "cross")
#
#     def forward(self, input, logdet=0.0, reverse=False, temperature=None):
#         if reverse:
#             z1 = input
#             mean, logs = self.split2d_prior(z1)
#             z2 = gaussian_sample(mean, logs, temperature)
#             z = torch.cat((z1, z2), dim=1)
#             return z, logdet
#         else:
#             z1, z2 = split_feature(input, "split")
#             mean, logs = self.split2d_prior(z1)
#             logdet = gaussian_likelihood(mean, logs, z2) + logdet
#             return z1, logdet
#
#
# class SqueezeLayer(nn.Module):
#     def __init__(self, factor):
#         super().__init__()
#         self.factor = factor
#
#     def forward(self, input, logdet=None, reverse=False):
#         output = (
#             unsqueeze2d(input, self.factor)
#             if reverse
#             else squeeze2d(input, self.factor)
#         )
#         return output, logdet
#
#
# class InvertibleConv1x1(nn.Module):
#     def __init__(self, num_channels, LU_decomposed):
#         super().__init__()
#         w_shape = [num_channels, num_channels]
#         w_init = torch.linalg.qr(torch.randn(*w_shape))[0]
#         self.LU_decomposed = LU_decomposed
#         if not LU_decomposed:
#             self.weight = nn.Parameter(w_init)
#         else:
#             p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
#             s = torch.diag(upper)
#             self.sign_s = torch.sign(s)
#             self.log_s = nn.Parameter(torch.log(torch.abs(s)))
#             upper = torch.triu(upper, 1)
#             self.lower = nn.Parameter(lower)
#             self.upper = nn.Parameter(upper)
#             self.register_buffer("p", p)
#             l_mask = torch.tril(torch.ones(w_shape), -1)
#             eye = torch.eye(*w_shape)
#             self.register_buffer("l_mask", l_mask)
#             self.register_buffer("eye", eye)
#         self.w_shape = w_shape
#
#     def get_weight(self, input, reverse):
#         b, c, h, w = input.shape
#         if not self.LU_decomposed:
#             dlogdet = torch.slogdet(self.weight)[1] * h * w
#             weight = torch.inverse(self.weight) if reverse else self.weight
#         else:
#             self.l_mask = self.l_mask.to(self.lower.device)
#             self.eye = self.eye.to(self.lower.device)
#             lower = self.lower * self.l_mask + self.eye
#             u = self.upper * self.l_mask.transpose(0, 1).contiguous().to(
#                 self.upper.device
#             )
#             u = u + torch.diag(self.sign_s * torch.exp(self.log_s))
#             dlogdet = torch.sum(self.log_s) * h * w
#             if reverse:
#                 u_inv = torch.inverse(u)
#                 l_inv = torch.inverse(lower)
#                 p_inv = self.p.t()
#                 weight = u_inv @ l_inv @ p_inv
#             else:
#                 weight = self.p @ lower @ u
#         weight = weight.view(self.w_shape[0], self.w_shape[1], 1, 1).to(input.device)
#         return weight, dlogdet
#
#     def forward(self, input, logdet=None, reverse=False):
#         """
#         log-det = log|abs(|W|)| * pixels
#         """
#         weight, dlogdet = self.get_weight(input, reverse)
#         if not reverse:
#             z = F.conv2d(input, weight)
#             if logdet is not None:
#                 logdet = logdet + dlogdet
#             return z, logdet
#         else:
#             z = F.conv2d(input, weight)
#             if logdet is not None:
#                 logdet = logdet - dlogdet
#             return z, logdet
# STG-NF modules, based on awesome previous work by https://github.com/y0ast/Glow-PyTorch

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from STG_NF.models.STG_NF.utils import compute_same_pad, split_feature


def gaussian_p(mean, logs, x):
    """
    lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
            k = 1 (Independent)
            Var = exp(2 * logs)
    """
    var = torch.exp(2.0 * logs) + 1e-6
    c = math.log(2 * math.pi)
    return -0.5 * (2.0 * logs + ((x - mean) ** 2) / var + c)


def gaussian_likelihood(mean, logs, x):
    p = gaussian_p(mean, logs, x)
    return torch.sum(p, dim=[1, 2, 3])


def gaussian_sample(mean, logs, temperature=1):
    # Sample from Gaussian with temperature
    temperature = temperature or 1
    return torch.normal(mean, torch.exp(logs) * temperature)


def squeeze2d(input, factor):
    if factor == 1:
        return input
    B, C, T, V = input.size()
    assert T % factor == 0, "T modulo factor is not 0"
    x = input.view(B, C, T // factor, factor, V)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    return x.view(B, C * factor, T // factor, V)


def unsqueeze2d(input, factor):
    if factor == 1:
        return input
    factor2 = factor**2
    B, C, T, V = input.size()
    assert C % factor2 == 0, "C modulo factor^2 is not 0"
    x = input.view(B, C // factor2, factor, T, V)
    x = x.permute(0, 1, 4, 2, 3).contiguous()
    return x.view(B, C // factor2, T * factor, V)


class _ActNorm(nn.Module):
    """
    Activation Normalization.
    Инициализирует смещение и масштаб по выборке, так что выход по каналам имеет нулевое среднее и единичное отклонение.
    После инициализации bias и logs обучаются как параметры.
    """
    def __init__(self, num_features, scale=1.0):
        super().__init__()
        size = [1, num_features, 1, 1]
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = num_features
        self.scale = scale
        self.inited = False

    def initialize_parameters(self, input):
        if not self.training:
            raise ValueError("In Eval mode, but ActNorm not initialized")
        with torch.no_grad():
            bias = -torch.mean(input, dim=[0, 2, 3], keepdim=True)
            var = torch.mean((input + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(var) + 1e-6))
            self.bias.data.copy_(bias)
            self.logs.data.copy_(logs)
            self.inited = True

    def _center(self, input, reverse=False):
        return input - self.bias if reverse else input + self.bias

    def _scale(self, input, logdet=None, reverse=False):
        if reverse:
            input = input * torch.exp(-self.logs)
        else:
            input = input * torch.exp(self.logs)
        if logdet is not None:
            b, c, h, w = input.shape
            dlogdet = torch.sum(self.logs) * h * w
            logdet = logdet - dlogdet if reverse else logdet + dlogdet
        return input, logdet

    def forward(self, input, logdet=None, reverse=False):
        self._check_input_dim(input)
        if not self.inited:
            self.initialize_parameters(input)
        if reverse:
            input, logdet = self._scale(input, logdet, reverse)
            input = self._center(input, reverse)
        else:
            input = self._center(input, reverse)
            input, logdet = self._scale(input, logdet, reverse)
        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.0):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4, "Input must be 4D"
        assert input.size(1) == self.num_features, (
            f"Expected {self.num_features} channels but got {input.size(1)}"
        )


class LinearZeros(nn.Module):
    def __init__(self, in_channels, out_channels, logscale_factor=3):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()
        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels))

    def forward(self, input):
        output = self.linear(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 1),
        stride=(1, 1),
        padding="same",
        do_actnorm=True,
        weight_std=0.05,
    ):
        super().__init__()
        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=(not do_actnorm),
        )
        with torch.no_grad():
            self.conv.weight.normal_(0.0, weight_std)
            if not do_actnorm:
                self.conv.bias.zero_()
        self.do_actnorm = do_actnorm
        if do_actnorm:
            self.actnorm = ActNorm2d(out_channels)

    def forward(self, input):
        x = self.conv(input.squeeze())
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 1),
        stride=(1, 1),
        padding="same",
        logscale_factor=3,
    ):
        super().__init__()
        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.zero_()
        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, input):
        output = self.conv(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Permute2d(nn.Module):
    def __init__(self, num_channels, shuffle):
        super().__init__()
        self.num_channels = num_channels
        # Регистрируем индексы как буферы, чтобы они автоматически перемещались на нужное устройство
        self.register_buffer("indices", torch.arange(num_channels - 1, -1, -1, dtype=torch.long))
        self.register_buffer("indices_inverse", torch.argsort(self.indices))
        if shuffle:
            self.reset_indices()

    def reset_indices(self):
        # Обновляем буферы
        shuffle_idx = torch.randperm(self.indices.size(0))
        self.register_buffer("indices", self.indices[shuffle_idx])
        self.register_buffer("indices_inverse", torch.argsort(self.indices))

    def forward(self, input, reverse=False):
        assert len(input.size()) == 4, "Input must be 4D"
        if reverse:
            return input[:, self.indices_inverse, :, :]
        else:
            return input[:, self.indices, :, :]


class Split2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = Conv2dZeros(num_channels // 2, num_channels)

    def split2d_prior(self, z):
        h = self.conv(z)
        return split_feature(h, "cross")

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            z1 = input
            mean, logs = self.split2d_prior(z1)
            z2 = gaussian_sample(mean, logs, temperature)
            z = torch.cat((z1, z2), dim=1)
            return z, logdet
        else:
            z1, z2 = split_feature(input, "split")
            mean, logs = self.split2d_prior(z1)
            logdet = gaussian_likelihood(mean, logs, z2) + logdet
            return z1, logdet


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        output = (
            unsqueeze2d(input, self.factor)
            if reverse
            else squeeze2d(input, self.factor)
        )
        return output, logdet


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.linalg.qr(torch.randn(*w_shape))[0]
        self.LU_decomposed = LU_decomposed
        if not LU_decomposed:
            self.weight = nn.Parameter(w_init)
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            self.sign_s = torch.sign(s)
            self.log_s = nn.Parameter(torch.log(torch.abs(s)))
            upper = torch.triu(upper, 1)
            self.lower = nn.Parameter(lower)
            self.upper = nn.Parameter(upper)
            self.register_buffer("p", p)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)
            self.register_buffer("l_mask", l_mask)
            self.register_buffer("eye", eye)
        self.w_shape = w_shape

    def get_weight(self, input, reverse):
        b, c, h, w = input.shape
        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            weight = torch.inverse(self.weight) if reverse else self.weight
        else:
            lower = self.lower * self.l_mask + self.eye
            u = self.upper * self.l_mask.transpose(0, 1).contiguous().to(self.upper.device)
            u = u + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = torch.sum(self.log_s) * h * w
            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = self.p.t()
                weight = u_inv @ l_inv @ p_inv
            else:
                weight = self.p @ lower @ u
        weight = weight.view(self.w_shape[0], self.w_shape[1], 1, 1).to(input.device)
        return weight, dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet
