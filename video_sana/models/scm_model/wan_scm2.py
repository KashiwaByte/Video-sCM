# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.nn as nn
import pdb
import math


from models.wan.modules.model import WanModel

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential( nn.Linear(frequency_embedding_size, hidden_size, bias=True).cuda(), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True).cuda(), )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp( -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half ).cuda()
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).cuda()
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

# Configuration for WAN 1.3B model
WAN_1_3B_CONFIG = {
    'patch_size': (1, 2, 2),
    'dim': 1536,
    'ffn_dim': 8960,
    'freq_dim': 256,
    'num_heads': 12,
    'num_layers': 30,
    'window_size': (-1, -1),
    'qk_norm': True,
    'cross_attn_norm': True,
    'eps': 1e-6
}

class WanModelSCM(WanModel):
    def __init__(self, *args, logvar=False, logvar_scale_factor=1.0, **kwargs):
        config = WAN_1_3B_CONFIG.copy()
        config.update(kwargs)
        super().__init__(*args, **config)
        self.logvar_scale_factor=logvar_scale_factor
        self.logvar_linear = None        


        if logvar:
            self.logvar_scale_factor = logvar_scale_factor
            self.logvar_linear = nn.Linear(self.dim, 1).cuda()
                    

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
        
        with torch.no_grad():

            # TrigFlow --> Flow Transformation
            # the input now is [0, np.pi/2], arctan(N(P_mean, P_std))
            t = torch.sin(t) / (torch.cos(t) + torch.sin(t))

            # stabilize large resolution training
            pretrain_timestep = t * 1000

            # scale input based on timestep
            t = t.view(-1, 1, 1, 1, 1)
            scale_factor = torch.sqrt(t**2 + (1 - t) ** 2)
            x = [x_i * scale_factor for x_i in x]
            x = [x_i.squeeze(0) if x_i.shape[0] == 1 and len(x_i.shape) > 4 else x_i for x_i in x]

        # t = t.flatten()
        # pdb.set_trace()
        # forward in original flow
        model_out = super().forward(x,pretrain_timestep,context,seq_len,clip_fea=None,y=None, **kwargs)
        
        if not isinstance(model_out, list):
            model_out = [model_out]

        # Flow --> TrigFlow Transformation
        # Directly apply TrigFlow transformation regardless of jvp flag
        trigflow_model_out = [((1 - 2 * t) * x_i + (1 - 2 * t + 2 * t**2) * model_out_i) / torch.sqrt(t**2 + (1 - t) ** 2)
                             for x_i, model_out_i in zip(x, model_out)]

        if return_logvar :
            # self.logvar_scale_factor = 1.0
            self.logvar_linear = nn.Linear(self.dim, 1)
            self.logvar_linear.cuda()
            self.t_embedder = TimestepEmbedder(self.dim)
            t = self.t_embedder(pretrain_timestep).cuda()
            # pdb.set_trace()
            logvar = self.logvar_linear(t) * self.logvar_scale_factor
            return trigflow_model_out, logvar
        return trigflow_model_out