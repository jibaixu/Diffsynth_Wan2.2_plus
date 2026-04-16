import torch
from torch import nn

from .wan_video_dit import DiTBlock


class TrackContextWanAttentionBlock(DiTBlock):
    def __init__(
        self,
        has_image_input,
        has_text_input,
        dim,
        num_heads,
        ffn_dim,
        eps=1e-6,
    ):
        super().__init__(has_image_input, has_text_input, dim, num_heads, ffn_dim, eps=eps)

    def forward(self, c, context, t_mod, freqs):
        return super().forward(c, context, t_mod, freqs)


class TrackContextWanModel(nn.Module):
    def __init__(
        self,
        track_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
        track_in_dim=16,
        patch_size=(1, 2, 2),
        has_image_input=False,
        has_text_input=True,
        dim=1536,
        num_heads=12,
        ffn_dim=8960,
        eps=1e-6,
    ):
        super().__init__()
        self.track_layers = tuple(int(layer_id) for layer_id in track_layers)
        self.track_in_dim = int(track_in_dim)
        self.track_layers_mapping = {layer_id: idx for idx, layer_id in enumerate(self.track_layers)}
        self.track_blocks = nn.ModuleList(
            [
                TrackContextWanAttentionBlock(
                    has_image_input=has_image_input,
                    has_text_input=has_text_input,
                    dim=dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for layer_id in self.track_layers
            ]
        )
        self.track_output_projs = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in self.track_layers]
        )
        self.track_patch_embedding = nn.Conv3d(
            self.track_in_dim,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(
        self,
        track_context_latents,
        context,
        t_mod,
        freqs,
        text_token_count=0,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        if track_context_latents is None:
            return ()

        c = [self.track_patch_embedding(latent.unsqueeze(0)) for latent in track_context_latents]
        c = [latent.flatten(2).transpose(1, 2) for latent in c]
        c = torch.cat(c, dim=0)
        hints = []

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        for block in self.track_blocks:
            block.cross_attn.text_token_count = text_token_count
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    c = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        c,
                        context,
                        t_mod,
                        freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                c = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    c,
                    context,
                    t_mod,
                    freqs,
                    use_reentrant=False,
                    )
            else:
                c = block(c, context, t_mod, freqs)
            hints.append(self.track_output_projs[len(hints)](c))
        return tuple(hints)

    def init_from_dit(self, dit: "WanModel", zero_init_extra: bool = True):
        from .wan_video_dit import WanModel

        if not isinstance(dit, WanModel):
            return

        for track_idx, layer_id in enumerate(self.track_layers):
            if layer_id >= len(dit.blocks):
                continue
            src = dit.blocks[layer_id]
            dst = self.track_blocks[track_idx]
            for name, module in src.named_children():
                if not hasattr(dst, name):
                    continue
                dst_module = getattr(dst, name)
                if isinstance(module, nn.Module) and isinstance(dst_module, nn.Module):
                    try:
                        dst_module.load_state_dict(module.state_dict(), strict=True)
                    except Exception:
                        pass
            if zero_init_extra:
                proj = self.track_output_projs[track_idx]
                nn.init.zeros_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
