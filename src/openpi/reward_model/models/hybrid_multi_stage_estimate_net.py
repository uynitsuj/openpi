import torch
import torch.nn as nn

class StageTransformer(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 vis_emb_dim: int = 512,
                 text_emb_dim: int = 512,
                 state_dim: int = 7,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_len: int = 128,
                 num_cameras: int = 1,
                 # heads per scheme
                 num_classes_sparse: int = 4,   
                 num_classes_dense: int = 8,    
                 ):
        super().__init__()
        self.d_model = d_model
        self.num_cameras = num_cameras
        self.max_seq_len = max_seq_len

        # Projections
        self.lang_proj = nn.Linear(text_emb_dim, d_model)
        self.visual_proj = nn.Linear(vis_emb_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)

        # Positional bias (only tag the first visual frame of each camera)
        self.first_pos = nn.Parameter(torch.zeros(1, d_model))

        # Shared fusion MLP (produces a d_model feature per timestep)
        fused_in = d_model * (num_cameras + 2)
        self.fusion_backbone = nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, d_model),
            nn.ReLU(),
        )

        # Scheme-specific heads
        self.heads = nn.ModuleDict({
            "sparse": nn.Linear(d_model, num_classes_sparse),
            "dense":  nn.Linear(d_model, num_classes_dense),
        })

    def _prep_lang(self, lang_emb: torch.Tensor, B: int, T: int, D: int) -> torch.Tensor:
        """
        Accepts lang_emb of shape:
          - (B, text_emb_dim)  -> broadcast across time
          - (B, T, text_emb_dim) -> per-timestep (dense)
        Returns (B, 1, T, D)
        """
        if lang_emb.dim() == 3:
            # (B, T, E) -> (B, T, D) -> (B, 1, T, D)
            lang_proj = self.lang_proj(lang_emb).unsqueeze(1)
        else:
            # (B, E) -> (B, 1, 1, D) -> expand to (B, 1, T, D)
            lang_proj = self.lang_proj(lang_emb).unsqueeze(1).unsqueeze(2).expand(B, 1, T, D)
        return lang_proj

    def forward(self,
                img_seq: torch.Tensor,     # (B, N, T, vis_emb_dim)
                lang_emb: torch.Tensor,    # (B, E) or (B, T, E)
                state: torch.Tensor,       # (B, T, state_dim)
                lengths: torch.Tensor,     # (B,)
                scheme: str = "sparse",    # "sparse" or "dense"
                ) -> torch.Tensor:
        assert scheme in self.heads, f"Unknown scheme '{scheme}'. Use one of {list(self.heads.keys())}."
        B, N, T, _ = img_seq.shape
        D = self.d_model
        device = img_seq.device

        # Project inputs
        vis_proj = self.visual_proj(img_seq)                 # (B, N, T, D)
        state_proj = self.state_proj(state).unsqueeze(1)     # (B, 1, T, D)
        lang_proj = self._prep_lang(lang_emb, B, T, D)       # (B, 1, T, D)

        # Concatenate streams (cameras + lang + state)
        x = torch.cat([vis_proj, lang_proj, state_proj], dim=1)   # (B, N+2, T, D)

        # Tag first frame for each camera stream
        x[:, :N, 0, :] += self.first_pos

        # Flatten to tokens for Transformer
        x_tokens = x.view(B, (N + 2) * T, D)

        # Key padding mask (broadcast the time mask to all streams)
        base_mask = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)  # (B, T)
        mask = base_mask.unsqueeze(1).expand(B, N + 2, T).reshape(B, (N + 2) * T)

        # Encode
        h = self.transformer(x_tokens, src_key_padding_mask=mask)  # (B, (N+2)*T, D)

        # Back to (B, T, (N+2)*D) then fuse to (B, T, D)
        h = h.view(B, N + 2, T, D).permute(0, 2, 1, 3).reshape(B, T, (N + 2) * D)
        fused = self.fusion_backbone(h)                             # (B, T, D)

        # Scheme-specific logits
        logits = self.heads[scheme](fused)                          # (B, T, C_scheme)
        return logits
