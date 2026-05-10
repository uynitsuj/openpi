import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

from openpi.reward_model.models.encoders.clip_encoder import FrozenCLIPEncoder
from openpi.reward_model.models.hybrid_multi_stage_estimate_net import StageTransformer
from openpi.reward_model.models.hybrid_multi_stage_reward_net import RewardTransformer

def normalize_sparse(x):
    x = np.asarray(x, dtype=float)
    if np.any((x < 0) | (x > 5)):
        raise ValueError("x must be within [0, 5].")
    # breakpoints and mapped values
    xp = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    fp = np.array([0.0, 0.05, 0.1, 0.3, 0.9, 1.0], dtype=float)
    return np.interp(x, xp, fp)

def comply_rm_lerobot_batch_multi_stage(batch: dict, camera_names: List[str] = ["top_camera-images-rgb"]) -> dict:
    """Comply with lerobot dataset batch format."""
    # convert to diffusion dataset format
    # this is a hack to make it work with lerobot dataset
    result =  {
        "image_frames": {},
        "targets": batch["targets"],
        "lengths": torch.as_tensor(batch["lengths"], device="cuda:0"),
        "state": torch.as_tensor(batch["state"], device="cuda:0"),
        "frame_relative_indices": batch["frame_relative_indices"],
        "tasks": batch["task"],
    }

    for cam_name in camera_names:
        result["image_frames"][cam_name] = torch.as_tensor(batch[cam_name], device="cuda:0")

    return result

def comply_lerobot_batch(batch: dict, camera_names: List[str] = ["left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"]) -> dict:
    """Comply with lerobot dataset batch format."""
    # convert to diffusion dataset format
    # this is a hack to make it work with lerobot dataset
    result = {
        "obs": {
        "image_frames": {},
        "state": torch.as_tensor(batch["state"], device="cuda:0"),
        },
        "action": batch["actions"],
        "masks": batch["mask"],
    }

    for cam_name in camera_names:
        result["obs"]["image_frames"][cam_name] = torch.as_tensor(batch[cam_name], device="cuda:0")

    return result

class RunningMeanStd:
    """Numerically stable running mean & std (Welford). Mean is clamped > 0."""
    def __init__(self, min_mean=1e-8, eps=1e-8):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_mean = float(min_mean)
        self.eps = float(eps)

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64).ravel()
        for v in x:
            self.count += 1
            delta = v - self.mean
            self.mean += delta / self.count
            delta2 = v - self.mean
            self.M2 += delta * delta2
        # enforce strictly positive mean
        if self.mean <= 0:
            self.mean = self.min_mean

    @property
    def std(self):
        if self.count < 2:
            return 0.0
        return float(np.sqrt(self.M2 / (self.count - 1)))
    
@dataclass
class RMConfig:
    device: str = "cuda:0"
    camera_names: List[str] = field(default_factory=lambda: ["top_camera-images-rgb"])
    d_model: int = 768
    state_dim: int = 14
    n_heads: int = 12
    dropout: float = 0.1
    max_seq_len: int = 128
    num_classes_sparse: int = 6
    num_classes_dense: int = 9
    n_obs_steps: int = 8
    vision_ckpt: str = "openai/clip-vit-base-patch32"
    # n_layers: int = 6
    # ckpt_path: str = "/nfs_us/david_chen/reward_model_ckpt/hybrid/025-08-16/08-14-36/rewind_reward_fixed_seq_model_frame_gap_multi_stage/fold_tshirt_hybird/checkpoints"
    # ckpt_path: str = "/nfs_old/david_chen/rewind_tshirt/outputs/2025-08-16/08-14-36/rewind_reward_fixed_seq_model_frame_gap_multi_stage/fold_tshirt_hybird/checkpoints"
    # reward_model_ckpt: str = "reward_step_080000_loss_0.004.pt"
    # stage_model_ckpt: str = "stage_step_080000_loss_0.011.pt"
    n_layers: int = 8
    ckpt_path: str = "/nfs_us/david_chen/reward_model_ckpt/2025-09-04/01-51-16/rewind_reward_fixed_seq_model_frame_gap_multi_stage/fold_tshirt_hybird/checkpoints"
    # ckpt_path: str = "/nfs_old/david_chen/rewind_tshirt/outputs/2025-09-04/01-51-16/rewind_reward_fixed_seq_model_frame_gap_multi_stage/fold_tshirt_hybird/checkpoints"
    reward_model_ckpt: str = "reward_step_050000_loss_0.002.pt"
    stage_model_ckpt: str = "stage_step_050000_loss_0.002.pt"
    model: nn.Module = field(init=False, default=None)
    no_state: bool = False

                                     
                                     
class HybridRM:
    def __init__(self, cfg: RMConfig):
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        vis_dim = 512
        txt_dim = 512
        self.camera_names = cfg.camera_names
        self.reward_model = RewardTransformer(d_model=cfg.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.state_dim,
                                  n_layers=cfg.n_layers,
                                  n_heads=cfg.n_heads,
                                  dropout=cfg.dropout,
                                  max_seq_len=cfg.max_seq_len,
                                  num_cameras=len(self.camera_names),
                                  ).to(self.device)
        self.stage_model = StageTransformer(d_model=cfg.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.state_dim,
                                  n_layers=cfg.n_layers,
                                  n_heads=cfg.n_heads,
                                  dropout=cfg.dropout,
                                  max_seq_len=cfg.max_seq_len,
                                  num_cameras=len(self.camera_names),
                                  num_classes_sparse=cfg.num_classes_sparse,
                                  num_classes_dense=cfg.num_classes_dense
                                  ).to(self.device)
        
        reward_model_path = Path(cfg.ckpt_path) / cfg.reward_model_ckpt
        stage_model_path = Path(cfg.ckpt_path) / cfg.stage_model_ckpt
        reward_ckpt = torch.load(reward_model_path, map_location=self.device)
        stage_ckpt = torch.load(stage_model_path, map_location=self.device)
        self.reward_model.load_state_dict(reward_ckpt["model"]); self.stage_model.load_state_dict(stage_ckpt["model"])
        self.reward_model.to(self.device); self.stage_model.to(self.device)
        self.reward_model.eval(); self.stage_model.eval()
        self.clip_encoder = FrozenCLIPEncoder(cfg.vision_ckpt, self.device)
        print("[INIT]: Reward Model is loaded")

        
    @torch.no_grad()
    def eval_step(self, batch, anno_type="sparse"):
        B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
        img_list = []
        for key in self.camera_names:
            imgs = batch["image_frames"][key].flatten(0, 1).to(self.device) # (B*T, C, H, W)
            img_list.append(imgs)
        
        lang_strs = ["fold the tshirt"]
        lens = batch["lengths"].to(self.device)
        state = batch["state"].to(self.device)
        # state = state_normalizer.normalize(state)
       
        # CLIP
        imgs_all = torch.cat(img_list, dim=0)  # (N * B * T, C, H, W)
        imgs_all = imgs_all.clamp(0, 1)
        img_emb = self.clip_encoder.encode_image(imgs_all)  # (N * B * T, D)
        img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
        lang_emb = self.clip_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)

        if self.cfg.no_state:
            state = torch.zeros_like(state, device=self.device)
        # stage_prob = self.stage_model(img_emb, lang_emb, state, lens, scheme=anno_type)  # (B, N, T, num_classes)
        stage_prob = self.stage_model(img_emb, lang_emb, state, lens, scheme=anno_type).softmax(dim=-1)
        stage_pred = stage_prob.argmax(dim=-1)
        stage_conf = stage_prob.gather(-1, stage_pred.unsqueeze(-1)).squeeze(-1)  # (B, T)
        stage_conf = stage_conf[:, self.cfg.n_obs_steps].detach().cpu().numpy()  # (B,)
        reward_pred = self.reward_model(img_emb, lang_emb, state, lens, scheme=anno_type)
        num_classes = self.cfg.num_classes_sparse 
        pred = torch.clip(reward_pred + stage_pred.float(), 0, num_classes-1)  # (B, T)
        current_pred = pred[:, self.cfg.n_obs_steps].detach().cpu().numpy()  # (B,)
        norm_pred = normalize_sparse(current_pred)  # (B,)

        return norm_pred, stage_conf
    
    @torch.no_grad()
    def eval_reward(self, rm_batch_curr, rm_batch_next):
        pred_curr, conf_curr = self.eval_step(rm_batch_curr)
        pred_next, conf_next = self.eval_step(rm_batch_next)
        raw_reward = pred_next - pred_curr
        
        # apply mask to reward with conf
        reward = raw_reward.copy()
        mask = (conf_curr < 0.9) | (conf_next < 0.9)
        reward[mask] = 1
        mean_conf = (np.mean(conf_curr)+np.mean(conf_next))/2
        
        return raw_reward, reward, mean_conf
    
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # Use default config
    cfg = RMConfig()

    # Instantiate
    rm = HybridRM(cfg)

    # Dummy batch (B can be anything; use 2 for a quick check)
    B, T, C, H, W = 2, 13, 3, 224, 224
    device = rm.device

    # Build image_frames dict for all cameras in config
    image_frames = {
        cam: torch.randn(B, T, C, H, W, device=device)
        for cam in cfg.camera_names
    }

    batch = {
        "image_frames": image_frames,                         # {cam: (B, T, C, H, W)}
        "lengths": torch.full((B,), T, dtype=torch.long),     # (B,)
        "state": torch.randn(B, T, cfg.state_dim, device=device),  # (B, T, state_dim)
    }

    anno_type = "sparse"  # or "dense", depending on your model's expected scheme

    with torch.no_grad():
        out = rm.eval_step(batch, anno_type)

    print("Normalized prediction shape:", out.shape)
    print("Normalized prediction values:", out)  # (B,)
