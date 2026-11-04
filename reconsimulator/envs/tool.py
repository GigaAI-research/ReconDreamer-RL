import sys
import os

current_dir = os.getcwd()
target_path = os.path.join(current_dir, "reconsimulator", "render")
sys.path.append(current_dir)
sys.path.append(target_path)

import copy
import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Optional
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from reconsimulator.render.utils.misc import import_str
from reconsimulator.envs import nus_config as cfg


# ----------------------------- 路径工具 ----------------------------- #
def _ckpt_path(scene: int) -> str:
    return os.path.join(cfg.BASE_DATA_DIR, f"{scene:03d}", "3DGS_without_prior", "checkpoint_final.pth")


def _trainer_config_path() -> str:
    prefer = os.path.join(cfg.INFO_DIR, "config.yaml")
    fallback = os.path.join("assets", "nus", "others", "config.yaml")
    if os.path.exists(prefer):
        return prefer
    return fallback


def get_splat(device: str, scene: int):
    """
    加载重建 trainer 与时间步（统一路径风格 + 健壮性处理）
    """
    ckpt_name = _ckpt_path(scene)
    if not os.path.exists(ckpt_name):
        raise FileNotFoundError(f"[get_splat] checkpoint not found: {ckpt_name}")

    checkpoint = torch.load(ckpt_name, map_location=device, weights_only=True)

    cfg_path = _trainer_config_path()
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"[get_splat] trainer config not found: {cfg_path}")

    conf = OmegaConf.load(cfg_path)
    conf = OmegaConf.merge(conf, OmegaConf.from_cli([]))

    try:
        embeds = checkpoint["models"]["CamPose"]["embeds.weight"]
        num_timesteps = embeds.shape[0] // 6
    except Exception as e:
        raise KeyError(
            "[get_splat] cannot infer num_timesteps from checkpoint; "
            "expect checkpoint['models']['CamPose']['embeds.weight']"
        ) from e

    recon_trainer = import_str(conf.trainer.type)(
        **conf.trainer,
        num_timesteps=num_timesteps,
        model_config=conf.model,
        num_train_images=num_timesteps * 6,
        num_full_images=num_timesteps * 6,
        device=device,
    )

    num_timesteps = (num_timesteps - 1) // 6 * 6
    recon_trainer.resume_from_checkpoint(ckpt_path=ckpt_name, load_only_model=True)
    return recon_trainer, num_timesteps


# ----------------------------- 视锥采样 ----------------------------- #
def get_sky_view(c2w: torch.Tensor,
                 intrinsics: torch.Tensor,
                 device: str,
                 img_height: int,
                 img_width: int):
    try:
        x, y = torch.meshgrid(
            torch.arange(img_width, device=device),
            torch.arange(img_height, device=device),
            indexing="xy",
        )
    except TypeError:
        x = torch.arange(img_width, device=device)
        y = torch.arange(img_height, device=device)
        x, y = torch.meshgrid(x, y)

    x = x.flatten()
    y = y.flatten()

    origins, viewdirs, direction_norm = get_rays(x, y, c2w, intrinsics)
    origins = origins.reshape(img_height, img_width, 3)
    viewdirs = viewdirs.reshape(img_height, img_width, 3)
    direction_norm = direction_norm.reshape(img_height, img_width, 1)
    return origins, viewdirs, direction_norm


def get_rays(
    x: Tensor, y: Tensor, c2w: Tensor, intrinsic: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    if len(intrinsic.shape) == 2:
        intrinsic = intrinsic[None, :, :]
    if len(c2w.shape) == 2:
        c2w = c2w[None, :, :]

    camera_dirs = torch.nn.functional.pad(
        torch.stack(
            [
                (x - intrinsic[:, 0, 2] + 0.5) / intrinsic[:, 0, 0],
                (y - intrinsic[:, 1, 2] + 0.5) / intrinsic[:, 1, 1],
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )
    directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
    direction_norm = torch.linalg.norm(directions, dim=-1, keepdims=True)
    viewdirs = directions / (direction_norm + 1e-8)
    return origins, viewdirs, direction_norm


# ----------------------------- 渲染状态 ----------------------------- #
def get_state(trainer, loaded_image_infos, loaded_cam_infos, now_frame: Optional[int] = None):
    device = next(trainer.parameters()).device if hasattr(trainer, "parameters") else torch.device("cuda")
    loaded_image_infos = move_to_device(loaded_image_infos, device)
    loaded_cam_infos = move_to_device(loaded_cam_infos, device)

    cam_infos1 = copy.deepcopy(loaded_cam_infos)
    image_infos1 = copy.deepcopy(loaded_image_infos)
    results = trainer(image_infos1, cam_infos1)

    normalized_rgb = results["rgb"].clamp(0.0, 1.0).detach().cpu().numpy()
    scaled_rgb = (normalized_rgb * 255).astype(np.uint8)
    return scaled_rgb


# ----------------------------- 设备搬运 ----------------------------- #
def move_to_device(data, device):
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


# ----------------------------- SLERP ----------------------------- #
def slerp(r1: R, r2: R, t: float) -> R:
    q1 = r1.as_quat()
    q2 = r2.as_quat()

    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2

    if np.abs(dot) > 0.9995:
        q = (1 - t) * q1 + t * q2
    else:
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta = np.sin(theta)
        q = (np.sin((1 - t) * theta) / sin_theta) * q1 + (np.sin(t * theta) / sin_theta) * q2

    return R.from_quat(q)
