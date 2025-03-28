# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.nn as nn

from models.wan.modules.model import WanModel


class WanModelSCM(WanModel):
    def __init__(self, *args, logvar=False, logvar_scale_factor=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.logvar_linear = None
        if logvar:
            self.logvar_scale_factor = logvar_scale_factor
            self.logvar_linear = nn.Linear(self.hidden_size, 1)

    def forward(self,x,t,context,seq_len,clip_fea=None,y=None, return_logvar=False, jvp=False, **kwargs):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            return_logvar (bool, *optional*, defaults to False):
                Whether to return predicted variance along with the denoised output
            jvp (bool, *optional*, defaults to False):
                Whether to compute Jacobian-vector product during forward pass

        Returns:
            List[Tensor] or Tuple[List[Tensor], Tensor]:
                If return_logvar is False:
                    List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
                If return_logvar is True:
                    Tuple of (denoised tensors, predicted variance)
        """
        
        # TrigFlow --> Flow Transformation
        # the input now is [0, np.pi/2], arctan(N(P_mean, P_std))
        t = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))

        # stabilize large resolution training
        pretrain_timestep = t * 1000
        t = t.view(-1, 1, 1, 1)

        # scale input based on timestep
        x = x * torch.sqrt(t**2 + (1 - t) ** 2)

        # forward in original flow
        if return_logvar and self.logvar_linear is not None:
            model_out = super().forward(x,t,context,seq_len,clip_fea=None,y=None, jvp=jvp, **kwargs)
            logvar = self.logvar_linear(t) * self.logvar_scale_factor
        else:
            model_out = super().forward(self,x,t,context,seq_len,clip_fea=None,y=None,jvp=jvp, **kwargs)

        # Flow --> TrigFlow Transformation
        trigflow_model_out = ((1 - 2 * t) * x + (1 - 2 * t + 2 * t**2) * model_out) / torch.sqrt(
            t**2 + (1 - t) ** 2
        )

        if return_logvar:
            return trigflow_model_out, logvar
        else:
            return trigflow_model_out