

from __future__ import annotations

import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from numpy import prod
from pyro.nn import PyroModule, PyroSample
from torch import nn


class ElementStandardScaler:
    def fit(self, x):
        self.mean = x.mean()
        self.std = x.std(unbiased=False)

    def transform(self, x):
        eps = 1e-20
        x = x - self.mean
        return x / (self.std + eps)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class ProbRasterLatent(PyroModule):
    def __init__(
        self,
        process_predictor: nn.Module,
        dims=(256, 256),
        latent_dims=(16, 16),
        interpolation="bilinear",
        prior_scale=0.01,
        obs_scale=0.01,
        prior_std=0.01,
        device=None,
    ):
        super().__init__()
        self.dims = dims
        self.device = device
        self.prior_std = prior_std
        if latent_dims is None:
            latent_dims = dims
        self.latent_dims = latent_dims
        self.interpolation = interpolation
        self.prior_scale = prior_scale
        self.obs_scale = torch.tensor(obs_scale, device=self.device, dtype=torch.float)
        self.process_predictor = process_predictor
        process_predictor.train(False)
        # Do not fit the process predictor weights
        for param in self.process_predictor.parameters():
            param.requires_grad = False
        _m, _s = (
            torch.tensor([0], device=self.device, dtype=torch.float),
            torch.tensor([self.prior_std], device=self.device, dtype=torch.float),
        )
        self.latent = PyroSample(dist.Normal(_m, _s).expand(latent_dims).to_event(2))

    def get_latent(self):
        if self.latent_dims == self.dims:
            return self.latent.unsqueeze(0)
        # `mini-batch x channels x [optional depth] x [optional height] x width`.
        return F.interpolate(
            self.latent.unsqueeze(1),
            self.dims,
            mode=self.interpolation,
            align_corners=False,
        ).squeeze(0)  # squeeze/unsqueeze is because of weird interpolate semantics

    def latent2source(self, latent):
        if latent.shape == self.dims:
            return latent.unsqueeze(0)
        # `mini-batch x channels x [optional depth] x [optional height] x width`.
        return F.interpolate(
            latent.unsqueeze(1), self.dims, mode=self.interpolation, align_corners=False
        ).squeeze(0)  # squeeze/unsqueeze is because of weird interpolate semantics

    def forward(self, grid, y=None):
        # overwrite process predictor batch with my own latent
        x = self.get_latent()
        # print("forward:x.shape,grid.shape=",x.shape,grid.shape)
        mean = self.process_predictor(x.to(self.device), grid.to(self.device))
        return pyro.sample("obs", dist.Normal(mean, self.obs_scale).to_event(2), obs=y)


class InitialConditionInterp(nn.Module):
    """
    InitialConditionInterp
    Class for the initial conditions using interpoliation. Works for 1d,2d and 3d

    model_ic = InitialConditionInterp([16],[8])
    model_ic = InitialConditionInterp([16,16],[8,8])
    model_ic = InitialConditionInterp([16,16,16],[8,8,8])

    June 2022, F.Alesiani
    """

    def __init__(self, dims, hidden_dim):
        super().__init__()
        self.spatial_dim = len(hidden_dim)
        self.dims = [1, *dims] if len(dims) == 1 else dims
        # self.dims = [1,1,1]+dims
        self.hidden_dim = [1, *hidden_dim] if len(hidden_dim) == 1 else hidden_dim
        self.interpolation = "bilinear" if len(hidden_dim) < 3 else "trilinear"
        self.scale = 1 / prod(hidden_dim)
        self.latent = nn.Parameter(
            self.scale * torch.rand(1, 1, *self.hidden_dim, dtype=torch.float)
        )
        # print(self.latent.shape)

    def latent2source(self, latent):
        if latent.shape[2:] == self.dims:
            return latent
        # `mini-batch x channels x [optional depth] x [optional height] x width`.
        latent = F.interpolate(
            latent, self.dims, mode=self.interpolation, align_corners=False
        )
        return latent.view(self.dims)

    def forward(self):
        x = self.latent2source(self.latent)
        if self.spatial_dim == 1:
            x = x.squeeze(0)
        return x
