from typing import Dict, Union

import torch
import torch.nn as nn

from ...util import append_dims, instantiate_from_config


class Denoiser(nn.Module):
    def __init__(self, weighting_config, scaling_config):
        super().__init__()

        self.weighting = instantiate_from_config(weighting_config)
        self.scaling = instantiate_from_config(scaling_config)
        # class VideoScaling:  # similar to VScaling
        #     def __call__(
        #         self, alphas_cumprod_sqrt: torch.Tensor, **additional_model_inputs
        #     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        #         c_skip = alphas_cumprod_sqrt
        #         c_out = -((1 - alphas_cumprod_sqrt**2) ** 0.5)
        #         c_in = torch.ones_like(alphas_cumprod_sqrt, device=alphas_cumprod_sqrt.device)
        #         c_noise = additional_model_inputs["idx"].clone()
        #         return c_skip, c_out, c_in, c_noise

    def possibly_quantize_sigma(self, sigma):
        return sigma

    def possibly_quantize_c_noise(self, c_noise):
        return c_noise

    def w(self, sigma):
        return self.weighting(sigma)

    def forward(
        self,
        network: nn.Module, # dit_video_concat.DiffusionTransformer
        input: torch.Tensor, # if edit, channel double
        sigma: torch.Tensor,
        cond: Dict,
        **additional_model_inputs,
    ) -> torch.Tensor:
        # 用于eval，用lambda方法省略了model
        # denoised = denoiser(
        #     *self.guider.prepare_inputs_edit(x, z, alpha_cumprod_sqrt, cond, uc), **additional_model_inputs
        # ).to(torch.float32)
        # 其中prepare_inputs_edit返回 torch.cat([x] * 2), torch.cat([s] * 2), c_out
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma, **additional_model_inputs)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))

        # output = network(input * c_in, c_noise, cond, **additional_model_inputs) * c_out

        input[:, :, :input.shape[2] // 2] *= c_in
        output = network(input, c_noise, cond, **additional_model_inputs) * c_out
        output = output + input[:, :, :output.shape[2]] * c_skip

        # input[:, :, :input.shape[2] // 2] *= c_in
        # output = network(input, c_noise, cond, **additional_model_inputs)
        # output = output + input[:, :, :output.shape[2]] * c_skip
        return output


class DiscreteDenoiser(Denoiser):
    def __init__(
        self,
        weighting_config,
        scaling_config,
        num_idx,
        discretization_config,
        do_append_zero=False,
        quantize_c_noise=True,
        flip=True,
    ):
        super().__init__(weighting_config, scaling_config)
        sigmas = instantiate_from_config(discretization_config)(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.sigmas = sigmas
        # self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise

    def sigma_to_idx(self, sigma):
        dists = sigma - self.sigmas.to(sigma.device)[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx):
        return self.sigmas.to(idx.device)[idx]

    def possibly_quantize_sigma(self, sigma):
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise):
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise
