"""Shared stability-aware CLIP forward, with optional analysis attentions.

Default embedding computation is identical to encode_visual.py. Analysis
reconstructs softmax coefficients from the actual layer inputs and bias,
without switching the embedding forward from SDPA to eager attention.
"""
import torch


class StabilityAwareClipImageEncoder(torch.nn.Module):
    def __init__(self, clip_model, stability_layer_indices, top_k, sigma, eps):
        super().__init__()
        self.clip_model = clip_model
        self.stability_layer_indices = set(stability_layer_indices)
        self.top_k = top_k
        self.sigma = sigma
        self.eps = eps
        self.logged_stability_layers = set()

    def gaussian_low_pass_kernel(self, feature_dim, device, dtype):
        positions = torch.arange(
            -feature_dim // 2 + 1,
            feature_dim // 2 + 1,
            device=device,
            dtype=torch.float32,
        )
        sigma = float(self.sigma) if self.sigma is not None else feature_dim ** 0.5
        sigma = max(sigma, float(self.eps))
        kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
        kernel = kernel / kernel.max().clamp_min(float(self.eps))
        return kernel.to(dtype=dtype).view(1, 1, feature_dim)

    def make_stability_patch_scores(self, hidden_states, layer_index):
        patch_tokens = hidden_states[:, 1:, :]
        if patch_tokens.size(1) == 0:
            return None

        batch_size = patch_tokens.size(0)
        num_patches = patch_tokens.size(1)
        feature_dim = patch_tokens.size(-1)
        top_k = min(int(self.top_k), num_patches)
        if layer_index not in self.logged_stability_layers:
            print(
                "stability-aware layer {} patch tokens: {}, top_k: {}".format(
                    layer_index,
                    num_patches,
                    top_k,
                )
            )
            self.logged_stability_layers.add(layer_index)

        compute_tokens = patch_tokens.float()
        low_pass_kernel = self.gaussian_low_pass_kernel(
            feature_dim=feature_dim,
            device=compute_tokens.device,
            dtype=compute_tokens.dtype,
        )

        freq_tokens = torch.fft.fft(compute_tokens, dim=-1)
        freq_tokens = torch.fft.fftshift(freq_tokens, dim=-1)
        freq_tokens = freq_tokens * low_pass_kernel
        freq_tokens = torch.fft.ifftshift(freq_tokens, dim=-1)
        filtered_tokens = torch.fft.ifft(freq_tokens, dim=-1).real

        stability = torch.abs(compute_tokens) / (
            torch.abs(filtered_tokens - compute_tokens) + float(self.eps)
        )
        indices = torch.topk(
            stability,
            k=top_k,
            dim=1,
            largest=True,
        ).indices
        flat_indices = indices.permute(0, 2, 1).reshape(
            batch_size,
            feature_dim * top_k,
        )
        votes = torch.zeros(
            batch_size,
            num_patches,
            device=indices.device,
            dtype=torch.long,
        )
        votes.scatter_add_(
            1,
            flat_indices,
            torch.ones_like(flat_indices, dtype=votes.dtype),
        )

        # Keep vote normalization and logarithms in float32 under fp16 CLIP.
        return votes.float()

    def make_stability_attention_bias(self, hidden_states, layer_index):
        patch_scores = self.make_stability_patch_scores(hidden_states, layer_index)
        if patch_scores is None:
            return None

        batch_size, _ = patch_scores.shape
        score_min = patch_scores.amin(dim=1, keepdim=True)
        score_max = patch_scores.amax(dim=1, keepdim=True)
        denom = score_max - score_min + float(self.eps)
        normalized_scores = (patch_scores - score_min) / denom
        patch_weights = normalized_scores

        cls_weights = torch.ones(
            batch_size,
            1,
            device=hidden_states.device,
            dtype=patch_weights.dtype,
        )
        source_weights = torch.cat([cls_weights, patch_weights], dim=1)
        attention_bias = torch.log(source_weights.clamp_min(float(self.eps)))
        return attention_bias[:, None, None, :].to(dtype=hidden_states.dtype)

    def forward(self, pixel_values, return_attentions=False):
        attentions = []
        vision_model = self.clip_model.vision_model
        hidden_states = vision_model.embeddings(pixel_values=pixel_values)
        hidden_states = vision_model.pre_layrnorm(hidden_states)

        for layer_index, encoder_layer in enumerate(
            vision_model.encoder.layers,
            start=1,
        ):
            attention_bias = None
            if layer_index in self.stability_layer_indices:
                attention_bias = self.make_stability_attention_bias(
                    hidden_states=hidden_states,
                    layer_index=layer_index,
                )
            if return_attentions:
                # Switching SDPA to eager changes fp16 hidden states and can
                # change later stability votes. Observe q/k alongside the
                # original forward, rather than changing that forward.
                attention = encoder_layer.self_attn
                normalized = encoder_layer.layer_norm1(hidden_states)
                batch, length, _ = normalized.shape
                queries = attention.q_proj(normalized).view(batch, length, -1, attention.head_dim).transpose(1, 2)
                keys = attention.k_proj(normalized).view(batch, length, -1, attention.head_dim).transpose(1, 2)
                logits = (queries.float() @ keys.float().transpose(-1, -2)) * attention.scale
                if attention_bias is not None:
                    logits = logits + attention_bias.float()
                attentions.append(torch.softmax(logits, dim=-1).mean(dim=1))
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_bias,
                causal_attention_mask=None,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]

        pooled_output = hidden_states[:, 0, :]
        pooled_output = vision_model.post_layernorm(pooled_output)
        embedding = self.clip_model.visual_projection(pooled_output)
        if return_attentions:
            return {"embedding": embedding, "attentions": attentions}
        return embedding

