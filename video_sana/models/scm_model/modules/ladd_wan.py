import torch
import torch.nn as nn
import os
import sys
current_file_path = os.path.abspath(__file__)
up_levels = 4
for _ in range(up_levels):
    current_file_path = os.path.dirname(current_file_path)
print(current_file_path)

sys.path.append(current_file_path)


from video_sana.models.modules.ladd_blocks import DiscHead
from video_sana.models.wan.wan_scm import WanModelSCM

class WanModelSCMDiscriminator(nn.Module):
    def __init__(self, pretrained_model: WanModelSCM, is_multiscale=False, head_block_ids=None):
        super().__init__()
        self.transformer = pretrained_model
        self.transformer.requires_grad_(False)

        if head_block_ids is None or len(head_block_ids) == 0:
            self.block_hooks = {2, 8, 10, 12, 19} if is_multiscale else {self.transformer.num_layers - 1}
        else:
            self.block_hooks = head_block_ids

        heads = []
        for i in range(len(self.block_hooks)):
            heads.append(DiscHead(self.transformer.hidden_size, 0, 0))
        self.heads = nn.ModuleList(heads)

    def get_head_inputs(self):
        return self.head_inputs

    def forward(self, x, t, context,seq_len, **kwargs):
        feat_list = []
        self.head_inputs = []

        def get_features(module, input, output):
            feat_list.append(output)
            return output

        hooks = []
        for i, block in enumerate(self.transformer.blocks):
            if i in self.block_hooks:
                hooks.append(block.register_forward_hook(get_features))

        self.transformer(x, t, context=context, seq_len=seq_len,return_logvar=False, **kwargs)

        for hook in hooks:
            hook.remove()

        res_list = []
        for feat, head in zip(feat_list, self.heads):
            B, N, C = feat.shape
            feat = feat.transpose(1, 2)  # [B, C, N]
            self.head_inputs.append(feat)
            res_list.append(head(feat, None).reshape(feat.shape[0], -1))

        concat_res = torch.cat(res_list, dim=1)

        return concat_res

    @property
    def model(self):
        return self.transformer

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)


class DiscHeadModel:
    def __init__(self, disc):
        self.disc = disc

    def state_dict(self):
        return {name: param for name, param in self.disc.state_dict().items() if not name.startswith("transformer.")}

    def __getattr__(self, name):
        return getattr(self.disc, name)


