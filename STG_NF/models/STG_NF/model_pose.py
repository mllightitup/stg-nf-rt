# STG-NF model, based on awesome previous work by https://github.com/y0ast/Glow-PyTorch


import math

import torch
import torch.nn as nn

from STG_NF.models.STG_NF.graph import Graph
from STG_NF.models.STG_NF.modules_pose import (
    gaussian_sample,
    ActNorm2d,
    Conv2d,
    Conv2dZeros,
    InvertibleConv1x1,
    Permute2d,
    SqueezeLayer,
    Split2d,
    gaussian_likelihood,
)
from STG_NF.models.STG_NF.stgcn import st_gcn
from STG_NF.models.STG_NF.utils import split_feature


def nan_throw(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"{name} has NaNs")
        print(f"{name}: {tensor}")
    if torch.isinf(tensor).any():
        print(f"{name} has infs")
        print(f"{name}: {tensor}")


def get_stgcn(
    in_channels,
    hidden_channels,
    out_channels,
    temporal_kernel_size=9,
    spatial_kernel_size=2,
    first=False,
):
    kernel_size = (temporal_kernel_size, spatial_kernel_size)
    if hidden_channels == 0:
        return nn.ModuleList(
            [st_gcn(in_channels, out_channels, kernel_size, 1, residual=(not first))]
        )
    return nn.ModuleList(
        [
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=(not first)),
            st_gcn(hidden_channels, out_channels, kernel_size, 1, residual=(not first)),
        ]
    )


def get_block(in_channels, out_channels, hidden_channels):
    return nn.Sequential(
        Conv2d(in_channels, hidden_channels),
        nn.ReLU(inplace=False),
        Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1), stride=(1, 1)),
        nn.ReLU(inplace=False),
        Conv2dZeros(hidden_channels, out_channels),
    )


def _ensure4d(x):
    return x if x.dim() == 4 else x.unsqueeze(1)


class FlowStep(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        actnorm_scale: float,
        flow_permutation: str = "permute",
        flow_coupling: str = "additive",
        LU_decomposed: bool = False,
        A=None,
        temporal_kernel_size: int = 4,
        edge_importance_weighting: bool = False,
        last: bool = False,
        first: bool = False,
        strategy: str = "uniform",
        max_hops: int = 8,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.device = device
        self.flow_coupling = flow_coupling
        if A is None:
            g = Graph(strategy=strategy, max_hop=max_hops)
            self.A = torch.from_numpy(g.A).float().to(device)
        else:
            self.A = A

        self.actnorm = ActNorm2d(in_channels, actnorm_scale)

        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
            self.flow_permutation = lambda z, ld, rev: self.invconv(z, ld, rev)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
            self.flow_permutation = lambda z, ld, rev: (self.shuffle(z, rev), ld)
        else:
            self.reverse_perm = Permute2d(in_channels, shuffle=False)
            self.flow_permutation = lambda z, ld, rev: (self.reverse_perm(z, rev), ld)

        if flow_coupling == "additive":
            self.block = get_stgcn(
                in_channels // 2,
                in_channels // 2,
                hidden_channels,
                temporal_kernel_size=temporal_kernel_size,
                spatial_kernel_size=self.A.size(0),
                first=first,
            )
        elif flow_coupling == "affine":
            self.block = get_stgcn(
                in_channels // 2,
                hidden_channels,
                in_channels,
                temporal_kernel_size=temporal_kernel_size,
                spatial_kernel_size=self.A.size(0),
                first=first,
            )

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(self.A.size())) for _ in self.block]
            )
        else:
            self.edge_importance = [1] * len(self.block)

    def _affine_coupling(self, z1):
        z1 = _ensure4d(z1)
        h = z1.clone()
        for gcn, imp in zip(self.block, self.edge_importance):
            h, _ = gcn(h, self.A * imp)
        shift, scale = split_feature(h, "cross")
        shift, scale = _ensure4d(shift), _ensure4d(scale)
        scale = torch.sigmoid(scale + 2.0) + 1e-6
        return shift, scale

    def forward(self, input, logdet=None, reverse=False, label=None):
        if not reverse:
            return self.normal_flow(input, logdet)
        return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)
        z, logdet = self.flow_permutation(z, logdet, False)
        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.block(z1)
        elif self.flow_coupling == "affine":
            shift, scale = self._affine_coupling(z1)
            z2 = (z2 + shift) * scale
            logdet = logdet + torch.sum(torch.log(scale), dim=[1, 2, 3])
        z = torch.cat((z1, z2), dim=1)
        return z, logdet
        # def normal_flow(self, input, logdet):
        #     # 1. actnorm
        #     z, logdet = self.actnorm(input, logdet=logdet, reverse=False)
        #
        #     # 2. permute
        #     z, logdet = self.flow_permutation(z, logdet, False)
        #
        #     # 3. coupling
        #     z1, z2 = split_feature(z, "split")
        #     if self.flow_coupling == "additive":
        #         z2 = z2 + self.block(z1)
        #     elif self.flow_coupling == "affine":
        #         if len(z1.shape) == 3:
        #             z1 = z1.unsqueeze(dim=1)
        #         if len(z2.shape) == 3:
        #             z2 = z2.unsqueeze(dim=1)
        #         h = z1.clone()
        #         for gcn, importance in zip(self.block, self.edge_importance):
        #             # h = gcn(h)
        #             h, _ = gcn(h, self.A * importance)
        #         shift, scale = split_feature(h, "cross")
        #         if len(scale.shape) == 3:
        #             scale = scale.unsqueeze(dim=1)
        #         if len(shift.shape) == 3:
        #             shift = shift.unsqueeze(dim=1)
        #         scale = torch.sigmoid(scale + 2.0) + 1e-6
        #         z2 = z2 + shift
        #         z2 = z2 * scale
        #         logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        #     z = torch.cat((z1, z2), dim=1)

    def reverse_flow(self, input, logdet):
        z1, z2 = split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            shift, scale = self._affine_coupling(z1)
            z2 = z2 / scale - shift
            logdet = logdet - torch.sum(torch.log(scale), dim=[1, 2, 3])
        z = torch.cat((z1, z2), dim=1)
        z, logdet = self.flow_permutation(z, logdet, True)
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        return z, logdet


class FlowNet(nn.Module):
    def __init__(
        self,
        pose_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        edge_importance=False,
        temporal_kernel_size=None,
        strategy="uniform",
        max_hops=8,
        device="cuda:0",
    ):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = K
        C, T, V = pose_shape
        for i in range(L):
            if i > 1:
                C, T, V = C * 2, T // 2, V
                self.layers.append(SqueezeLayer(factor=2))
                self.output_shapes.append([-1, C, T, V])
            tk = (
                temporal_kernel_size if temporal_kernel_size is not None else T // 2 + 1
            )
            for k in range(K):
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hidden_channels=hidden_channels,
                        actnorm_scale=actnorm_scale,
                        flow_permutation=flow_permutation,
                        flow_coupling=flow_coupling,
                        LU_decomposed=LU_decomposed,
                        temporal_kernel_size=tk,
                        edge_importance_weighting=edge_importance,
                        last=(k == K - 1),
                        first=(k == 0),
                        strategy=strategy,
                        max_hops=max_hops,
                        device=device,
                    )
                )
                self.output_shapes.append([-1, C, T, V])

    def encode(self, z, logdet=0.0):
        logdet = torch.zeros(z.shape[0]).to(self.device)
        for layer in self.layers:
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, temperature=None):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(z, logdet=0, reverse=True, temperature=temperature)
            else:
                z, logdet = layer(z, logdet=0, reverse=True)
        return z

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            return self.decode(input, temperature)
        return self.encode(input, logdet)


class STG_NF(nn.Module):
    def __init__(
        self,
        pose_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        learn_top,
        R=0,
        edge_importance=False,
        temporal_kernel_size=None,
        strategy="uniform",
        max_hops=8,
        device="cuda:0",
    ):
        super().__init__()
        self.flow = FlowNet(
            pose_shape=pose_shape,
            hidden_channels=hidden_channels,
            K=K,
            L=L,
            actnorm_scale=actnorm_scale,
            flow_permutation=flow_permutation,
            flow_coupling=flow_coupling,
            LU_decomposed=LU_decomposed,
            edge_importance=edge_importance,
            temporal_kernel_size=temporal_kernel_size,
            strategy=strategy,
            max_hops=max_hops,
            device=device,
        )
        self.R = R
        self.learn_top = learn_top
        if learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top_fn = Conv2dZeros(C * 2, C * 2)
        final_shape = self.flow.output_shapes[-1]
        self.register_buffer(
            "prior_h",
            torch.zeros([1, final_shape[1] * 2, final_shape[2], final_shape[3]]),
        )
        normal = torch.ones([final_shape[1], final_shape[2], final_shape[3]]) * self.R
        zeros = torch.zeros([final_shape[1], final_shape[2], final_shape[3]])
        self.register_buffer("prior_h_normal", torch.cat((normal, zeros), dim=0))
        abnormal = torch.ones([final_shape[1], final_shape[2], final_shape[3]]) * (
            -self.R
        )
        self.register_buffer("prior_h_abnormal", torch.cat((abnormal, zeros), dim=0))

    def prior(self, data, label=None):
        if data is not None:
            h = self.prior_h.expand(data.shape[0], -1, -1, -1).clone()
            if label is not None:
                h[label == 1] = self.prior_h_normal
                h[label == -1] = self.prior_h_abnormal
        else:
            h = self.prior_h_normal.expand(32, -1, -1, -1).clone()
        if self.learn_top:
            h = self.learn_top_fn(h)
        return split_feature(h, "split")

    def normal_flow(self, x, label, score):
        b, c, t, v = x.shape
        z, objective = self.flow(x, reverse=False)
        mean, logs = self.prior(x, label)
        objective += gaussian_likelihood(mean, logs, z)
        nll = (-objective) / (math.log(2.0) * c * t * v)
        return z, nll

    def reverse_flow(self, z, temperature):
        with torch.no_grad():
            if z is None:
                mean, logs = self.prior(z)
                z = gaussian_sample(mean, logs, temperature)
            x = self.flow(z, temperature=temperature, reverse=True)
        return x

    def forward(
        self, x=None, z=None, temperature=None, reverse=False, label=None, score=1
    ):
        if reverse:
            return self.reverse_flow(z, temperature)
        return self.normal_flow(x, label, score)

    def set_actnorm_init(self):
        for m in self.modules():
            if isinstance(m, ActNorm2d):
                m.inited = True
