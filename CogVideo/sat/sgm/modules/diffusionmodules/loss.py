from typing import List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import ListConfig
from ...util import append_dims, instantiate_from_config
from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from sat import mpu


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config,
        type="l2",
        offset_noise_level=0.0,
        use_lpips=False,
        lpips_weight=0.0,
        frameloss_weight=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        super().__init__()

        assert type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)

        self.type = type
        self.offset_noise_level = offset_noise_level

        self.frameloss_weight = frameloss_weight

        if type == "lpips":
            self.lpips = LPIPS().eval()

        self.use_lpips = use_lpips
        if self.use_lpips:
            self.lpips_weight = lpips_weight
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )
            noise = noise.to(input.dtype)
        noised_input = input.float() + noise * append_dims(sigmas, input.ndim)
        model_output = denoiser(network, noised_input, sigmas, cond, **additional_model_inputs)
        w = append_dims(denoiser.w(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


class VideoDiffusionLoss(StandardDiffusionLoss):
    def __init__(self, block_scale=None, block_size=None, min_snr_value=None, fixed_frames=0, **kwargs):
        self.fixed_frames = fixed_frames
        self.block_scale = block_scale
        self.block_size = block_size
        self.min_snr_value = min_snr_value
        self.is_edit = False
        super().__init__(**kwargs)

    def __call__(self, network, denoiser, conditioner, input, batch, input_edit=None):
        if input_edit is not None:
            self.is_edit = True

        cond = conditioner(batch)
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)
        idx = idx.to(input.device)

        noise = torch.randn_like(input)

        # broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        global_rank = torch.distributed.get_rank() // mp_size
        src = global_rank * mp_size
        torch.distributed.broadcast(idx, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group())

        additional_model_inputs["idx"] = idx

        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )

        if self.is_edit:
            noised_input = input_edit.float() * append_dims(alphas_cumprod_sqrt, input_edit.ndim) + noise * append_dims(
                (1 - alphas_cumprod_sqrt**2) ** 0.5, input_edit.ndim
            )
        else:
            noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims(
                (1 - alphas_cumprod_sqrt**2) ** 0.5, input.ndim
            )
        
        additional_model_inputs["is_edit"] = self.is_edit
        
        if self.is_edit:
            noised_input = torch.concat([noised_input, input], dim=2)

        if "concat_images" in batch.keys():
            cond["concat"] = batch["concat_images"]

        # [2, 13, 16, 60, 90],[2] dict_keys(['crossattn', 'concat'])  dict_keys(['idx'])
        model_output = denoiser(network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs)
        w = append_dims(1 / (1 - alphas_cumprod_sqrt**2), input.ndim)  # v-pred

        if self.min_snr_value is not None:
            w = min(w, self.min_snr_value)
        
        if input_edit is not None: 
            return self.get_loss(model_output, input_edit, input, w)
        else:
            return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, input, w):
        if self.type == "l2":
            if self.use_lpips:
                loss = (1 - self.lpips_weight) * torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1) + self.lpips_weight * self.lpips(model_output, target).reshape(-1)
                return loss
            else:
                B, T = target.shape[0], target.shape[1]
                loss_pre = (w * (model_output - target) ** 2)
                loss1 = torch.mean(loss_pre.reshape(B, -1), 1)
                loss2 = 0.0
                for i in range(T):
                    loss2 += torch.mean(loss_pre[:, i, :, :, :].reshape(B, -1), 1) * ((1 - i / T) ** 2)
                return self.frameloss_weight * loss2 / T + (1 - self.frameloss_weight) * loss1, model_output
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
