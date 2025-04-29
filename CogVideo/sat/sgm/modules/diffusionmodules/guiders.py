import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from functools import partial
import math

import torch
from einops import rearrange, repeat

from ...util import append_dims, default, instantiate_from_config


class Guider(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        pass

    def prepare_inputs(self, x: torch.Tensor, s: float, c: Dict, uc: Dict) -> Tuple[torch.Tensor, float, Dict]:
        pass


class VanillaCFG:
    """
    implements parallelized CFG
    """

    def __init__(self, scale, dyn_thresh_config=None):
        self.scale = scale
        scale_schedule = lambda scale, sigma: scale  # independent of step
        self.scale_schedule = partial(scale_schedule, scale)
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {"target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"},
            )
        )

    def __call__(self, x, sigma, scale=None):
        x_u, x_c = x.chunk(2)
        scale_value = default(scale, self.scale_schedule(sigma))
        x_pred = self.dyn_thresh(x_u, x_c, scale_value)
        return x_pred

    def prepare_inputs(self, x, s, c, uc):
        # self.guider.prepare_inputs(x, alpha_cumprod_sqrt, cond, uc)
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat"]:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out
        # (z, text)---- (1,0), (1,1)
    
    def prepare_inputs_edit(self, x, z, s, c, uc):
        # self.guider.prepare_inputs(x, z, alpha_cumprod_sqrt, cond, uc)
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat"]:
                c_out[k] = torch.cat((uc[k], uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        # for edit
        x_z = torch.concat([x, z], dim=2)
        x_z0 = torch.concat([x, torch.zeros_like(z)], dim=2)

        return torch.cat((x_z0, x_z, x_z), 0), torch.cat([s] * 3), c_out
        # (x, z, text) ---- (1,0,0), (1,1,0), (1,1,1)


class DynamicCFG(VanillaCFG):
    def __init__(self, scale, exp, num_steps, dyn_thresh_config=None):
        super().__init__(scale, dyn_thresh_config)
        scale_schedule = (
            lambda scale, sigma, step_index: 1 + scale * (1 - math.cos(math.pi * (step_index / num_steps) ** exp)) / 2
        )
        self.scale_schedule = partial(scale_schedule, scale)
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {"target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"},
            )
        )

    def __call__(self, x, sigma, step_index, scale=None):
        x_u, x_c = x.chunk(2)
        scale_value = self.scale_schedule(sigma, step_index.item())
        x_pred = self.dyn_thresh(x_u, x_c, scale_value)
        return x_pred
    

class DynamicCFG_edit(VanillaCFG):
    def __init__(self, scale, scale_img, exp, num_steps, dyn_thresh_config=None):
        super().__init__(scale, dyn_thresh_config)
        scale_schedule = (
            lambda scale, sigma, step_index: 1 + scale * (1 - math.cos(math.pi * (step_index / num_steps) ** exp)) / 2
        )
        self.scale_schedule = partial(scale_schedule, scale)

        scale_img_schedule = (
            lambda scale_img, sigma, step_index: 1 + scale_img * (1 - math.cos(math.pi * (step_index / num_steps) ** exp)) / 2
        )
        self.scale_img_schedule = partial(scale_img_schedule, scale_img)

        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {"target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding_edit"},
            )
        )

    def __call__(self, x, sigma, step_index, scale=None):
        x_uu, x_cu, x_cc = x.chunk(3)
        # (x, z, text) ---- (1,0,0), (1,1,0), (1,1,1)
        scale_value = self.scale_schedule(sigma, step_index.item())
        scale_img_schedule = self.scale_img_schedule(sigma, step_index.item())
        x_pred = self.dyn_thresh(x_uu, x_cu, x_cc, scale_value, scale_img_schedule)
        return x_pred


class IdentityGuider:
    def __call__(self, x, sigma):
        return x

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            c_out[k] = c[k]

        return x, s, c_out
