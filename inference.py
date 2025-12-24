import os
import sys
import json
import torch
import logging
import argparse
import random
import math
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
from einops import rearrange
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDIMScheduler
from torch.utils.data import Dataset
from datetime import datetime

from genphoto.pipelines.pipeline_inversion import GenPhotoInversionPipeline
from genphoto.models.unet import UNet3DConditionModelCameraCond
from genphoto.models.camera_adaptor import CameraCameraEncoder
from genphoto.utils.util import save_videos_grid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SETTINGS_CONFIG = {
    'bokehK': {
        'config': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_bokehK.yaml',
        'type_key': 'bokeh' 
    },
    'shutter': {
        'config': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_shutter_speed.yaml',
        'type_key': 'shutter'
    },
    'focal': {
        'config': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_focal_length.yaml',
        'type_key': 'focal'
    },
    'color': {
        'config': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_color_temperature.yaml',
        'type_key': 'color'
    }
}

def crop_focal_length(img_pil, base_focal_length, target_focal_length, target_height, target_width, sensor_height=24.0, sensor_width=36.0):
    width, height = img_pil.size
    base_x_fov = 2.0 * math.atan(sensor_width * 0.5 / base_focal_length)
    base_y_fov = 2.0 * math.atan(sensor_height * 0.5 / base_focal_length)
    target_x_fov = 2.0 * math.atan(sensor_width * 0.5 / target_focal_length)
    target_y_fov = 2.0 * math.atan(sensor_height * 0.5 / target_focal_length)

    crop_ratio = min(target_x_fov / base_x_fov, target_y_fov / base_y_fov)
    crop_width = int(round(crop_ratio * width))
    crop_height = int(round(crop_ratio * height))
    
    crop_width = max(1, min(width, crop_width))
    crop_height = max(1, min(height, crop_height))

    left = int((width - crop_width) / 2)
    top = int((height - crop_height) / 2)
    right = int((width + crop_width) / 2)
    bottom = int((height + crop_height) / 2)

    zoomed_img = img_pil.crop((left, top, right, bottom))
    resized_img = zoomed_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return resized_img

def kelvin_to_rgb(kelvin):
    if torch.is_tensor(kelvin): kelvin = kelvin.cpu().item()  
    temp = kelvin / 100.0
    if temp <= 66:
        red = 255
        green = 99.4708025861 * np.log(temp) - 161.1195681661 if temp > 0 else 0
        blue = 0 if temp <= 19 else 138.5177312231 * np.log(temp - 10) - 305.0447927307
    elif 66 < temp <= 88:
        red = 0.5 * (255 + 329.698727446 * ((temp - 60) ** -0.19332047592))
        green = 0.5 * (288.1221695283 * ((temp - 60) ** -0.1155148492) + (99.4708025861 * np.log(temp) - 161.1195681661 if temp > 0 else 0))
        blue = 0.5 * (138.5177312231 * np.log(temp - 10) - 305.0447927307 + 255)
    else:
        red = 329.698727446 * ((temp - 60) ** -0.19332047592)
        green = 288.1221695283 * ((temp - 60) ** -0.1155148492)
        blue = 255
    return np.array([red, green, blue], dtype=np.float32) / 255.0

def create_bokehK_embedding(bokehK_values, target_height, target_width):
    bokehK_values = bokehK_values.cpu().float()
    f = bokehK_values.shape[0]
    bokehK_embedding = torch.zeros((f, 3, target_height, target_width), dtype=torch.float32)
    for i in range(f):
        K_value = bokehK_values[i].item()
        kernel_size = max(K_value, 1.0)
        sigma = K_value / 3.0
        ax = np.linspace(-(kernel_size / 2), kernel_size / 2, int(np.ceil(kernel_size)))
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= np.sum(kernel)
        scale = kernel[int(np.ceil(kernel_size) / 2), int(np.ceil(kernel_size) / 2)]
        bokehK_embedding[i] = scale
    return bokehK_embedding

def create_focal_length_embedding(focal_length_values, target_height, target_width, base_focal_length=24.0, sensor_height=24.0, sensor_width=36.0):
    device = 'cpu'
    focal_length_values = focal_length_values.to(device).float()
    f = focal_length_values.shape[0]
    sensor_width_t = torch.tensor(sensor_width, device=device)
    sensor_height_t = torch.tensor(sensor_height, device=device)
    base_focal_length_t = torch.tensor(base_focal_length, device=device)

    base_fov_x = 2.0 * torch.atan(sensor_width_t * 0.5 / base_focal_length_t)
    base_fov_y = 2.0 * torch.atan(sensor_height_t * 0.5 / base_focal_length_t)
    target_fov_x = 2.0 * torch.atan(sensor_width_t * 0.5 / focal_length_values)
    target_fov_y = 2.0 * torch.atan(sensor_height_t * 0.5 / focal_length_values)

    crop_ratio_xs = target_fov_x / base_fov_x
    crop_ratio_ys = target_fov_y / base_fov_y
    center_h, center_w = target_height // 2, target_width // 2
    focal_length_embedding = torch.zeros((f, 3, target_height, target_width), dtype=torch.float32, device=device)

    for i in range(f):
        idx_val_y = crop_ratio_ys[i].item()
        idx_val_x = crop_ratio_xs[i].item()
        crop_h = max(1, min(target_height, torch.round(torch.tensor(idx_val_y) * target_height).int().item()))
        crop_w = max(1, min(target_width, torch.round(torch.tensor(idx_val_x) * target_width).int().item()))
        focal_length_embedding[i, :, center_h - crop_h // 2: center_h + crop_h // 2, center_w - crop_w // 2: center_w + crop_w // 2] = 1.0
    return focal_length_embedding

def create_color_temperature_embedding(color_temperature_values, target_height, target_width, min_color_temperature=2000, max_color_temperature=10000):
    values = color_temperature_values.cpu()
    f = values.shape[0]
    rgb_factors = []
    iter_values = values.view(-1)
    for val_tensor in iter_values:
        val = val_tensor.item()
        kelvin = min_color_temperature + (val * (max_color_temperature - min_color_temperature))
        rgb = kelvin_to_rgb(kelvin)
        rgb_factors.append(rgb)
    rgb_factors = torch.tensor(np.array(rgb_factors)).float().unsqueeze(2).unsqueeze(3)
    return rgb_factors.expand(f, 3, target_height, target_width)

def create_shutter_speed_embedding(shutter_speed_values, target_height, target_width, base_exposure=0.5):
    shutter_speed_values = shutter_speed_values.cpu().float()
    f = shutter_speed_values.shape[0]
    fwc = 32000.0
    scales = (shutter_speed_values / base_exposure) * (fwc / (fwc + 0.0001))
    return scales.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(f, 3, target_height, target_width)

class Universal_Camera_Embedding(Dataset):
    def __init__(self, setting_type, values, tokenizer, text_encoder, device, sample_size=[256, 384]):
        self.setting_type = setting_type
        self.values = values.to(device).float()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device  
        self.sample_size = sample_size

    def load(self):
        prompts = []
        for v in self.values.view(-1):
            val = v.item()
            if self.setting_type == 'bokeh': prompt = f"<bokeh kernel size: {val}>"
            elif self.setting_type == 'focal': prompt = f"<focal length: {val}>"
            elif self.setting_type == 'shutter': prompt = f"<exposure: {val}>"
            elif self.setting_type == 'color': prompt = f"<color temperature: {val}>"
            else: raise ValueError(f"Unknown setting: {self.setting_type}")
            prompts.append(prompt)

        with torch.no_grad():
            prompt_ids = self.tokenizer(prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(self.device)
            hidden = self.text_encoder(input_ids=prompt_ids).last_hidden_state

        diffs = [hidden[i] - hidden[i-1] for i in range(1, hidden.size(0))]
        if hidden.size(0) > 0: diffs.append(hidden[-1] - hidden[0])
        concat_diffs = torch.cat([d.unsqueeze(0) for d in diffs], dim=0) if diffs else torch.zeros_like(hidden)
        
        pad_len = 128 - concat_diffs.size(1)
        if pad_len > 0: concat_diffs = F.pad(concat_diffs, (0, 0, 0, pad_len))
        
        ccl = concat_diffs.reshape(concat_diffs.size(0), self.sample_size[0], self.sample_size[1]).unsqueeze(1).expand(-1, 3, -1, -1).to(self.device)
        
        vals_cpu = self.values.cpu()
        if self.setting_type == 'bokeh': vis = create_bokehK_embedding(vals_cpu, *self.sample_size)
        elif self.setting_type == 'focal': vis = create_focal_length_embedding(vals_cpu, *self.sample_size)
        elif self.setting_type == 'shutter': vis = create_shutter_speed_embedding(vals_cpu, *self.sample_size)
        elif self.setting_type == 'color': vis = create_color_temperature_embedding(vals_cpu, *self.sample_size)
        
        return torch.cat((vis.to(self.device), ccl), dim=1)

def load_models(cfg_path, device):
    cfg = OmegaConf.load(cfg_path)
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(cfg.noise_scheduler_kwargs))
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    
    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_path, subfolder="text_encoder").to(device)
    text_encoder.requires_grad_(False)
    
    unet = UNet3DConditionModelCameraCond.from_pretrained_2d(
        cfg.pretrained_model_path,
        subfolder=cfg.unet_subfolder,
        unet_additional_kwargs=cfg.unet_additional_kwargs
    ).to(device)
    unet.requires_grad_(False)

    camera_encoder = CameraCameraEncoder(**cfg.camera_encoder_kwargs).to(device)
    camera_encoder.requires_grad_(False)
    
    pipeline = GenPhotoInversionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=noise_scheduler, camera_encoder=camera_encoder
    ).to(device)
    pipeline.enable_vae_slicing()

    unet.set_all_attn_processor(
        add_spatial_lora=cfg.lora_ckpt is not None,
        add_motion_lora=cfg.motion_lora_rank > 0, 
        lora_kwargs={"lora_rank": cfg.lora_rank, "lora_scale": cfg.lora_scale},
        motion_lora_kwargs={"lora_rank": cfg.motion_lora_rank, "lora_scale": cfg.motion_lora_scale},
        **cfg.attention_processor_kwargs
    )

    if cfg.lora_ckpt:
        st = torch.load(cfg.lora_ckpt, map_location=device)
        unet.load_state_dict(st['lora_state_dict'] if 'lora_state_dict' in st else st, strict=False)
    
    if cfg.motion_module_ckpt:
        st = torch.load(cfg.motion_module_ckpt, map_location=device)
        unet.load_state_dict(st, strict=False)
        
    if cfg.camera_adaptor_ckpt:
        st = torch.load(cfg.camera_adaptor_ckpt, map_location=device)
        camera_encoder.load_state_dict(st['camera_encoder_state_dict'], strict=False)
        unet.load_state_dict(st['attention_processor_state_dict'], strict=False)
    
    pipeline.to(device)
    return pipeline, tokenizer, text_encoder

class OrderedInferenceChain:
    def __init__(self, device='cuda'):
        self.device = device
        self.pixel_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def run_t2i(self, pipeline, tokenizer, text_encoder, prompt, params, setting_type):
        video_len = len(params)
        target_vals = torch.tensor(params, dtype=torch.float32)
        
        embedder = Universal_Camera_Embedding(setting_type, target_vals, tokenizer, text_encoder, self.device)
        emb_raw = embedder.load()
        emb_3d = rearrange(emb_raw, "f c h w -> 1 c f h w")
        
        logger.info(f"  [T2I] Generating with prompt: '{prompt}', Setting: {setting_type}")
        output = pipeline(
            prompt=prompt,
            camera_embedding=emb_3d,
            video_length=video_len,
            height=256, width=384,
            num_inference_steps=25,
            guidance_scale=1.5,
            image=None
        ).videos
        return output

    def run_i2i(self, pipeline, tokenizer, text_encoder, source_img_pil, prompt, params, setting_type):
        video_len = len(params)
        target_vals = torch.tensor(params, dtype=torch.float32)
        input_pil = source_img_pil.resize((384, 256), Image.Resampling.LANCZOS)

        if setting_type == 'focal':
            logger.info("    > Mode: Focal Length (Physical Warp + Target Embedding Inversion)")
            frames = []
            base_fl = 24.0 
            for val in params:
                warped = crop_focal_length(input_pil, base_fl, val, 256, 384)
                frames.append(self.pixel_transforms(warped))
            
            input_tensor = torch.stack(frames).unsqueeze(0)
            input_tensor = rearrange(input_tensor, "b f c h w -> b c f h w")
            inv_image = input_tensor.to(self.device)
            
            tgt_embedder = Universal_Camera_Embedding(setting_type, target_vals, tokenizer, text_encoder, self.device)
            target_emb_3d = rearrange(tgt_embedder.load(), "f c h w -> 1 c f h w")
            inv_emb = target_emb_3d
            
        else:
            logger.info(f"    > Mode: {setting_type} (Static Image + Source Embedding Inversion)")
            single_tensor = self.pixel_transforms(input_pil)
            input_tensor = single_tensor.unsqueeze(0).unsqueeze(2).repeat(1, 1, video_len, 1, 1).to(self.device)
            inv_image = input_tensor
            
            initial_val = params[0]
            source_vals = torch.tensor([initial_val] * video_len, dtype=torch.float32)
            
            src_embedder = Universal_Camera_Embedding(setting_type, source_vals, tokenizer, text_encoder, self.device)
            inv_emb = rearrange(src_embedder.load(), "f c h w -> 1 c f h w")
            
            tgt_embedder = Universal_Camera_Embedding(setting_type, target_vals, tokenizer, text_encoder, self.device)
            target_emb_3d = rearrange(tgt_embedder.load(), "f c h w -> 1 c f h w")

        logger.info(f"    > Inverting...")
        latents = pipeline.invert(
            image=inv_image,
            prompt=prompt,
            camera_embedding=inv_emb,
            num_inference_steps=25,
            video_length=video_len
        )

        logger.info(f"    > Generating...")
        output = pipeline(
            prompt=prompt,
            latents=latents,
            camera_embedding=target_emb_3d,
            video_length=video_len,
            height=256, width=384,
            num_inference_steps=25,
            guidance_scale=1.5
        ).videos
        return output

    def get_middle_frame(self, video_tensor):
        frames = video_tensor.squeeze(0).permute(1, 0, 2, 3)
        total_frames = len(frames)
        
        mid_idx = total_frames // 2
        
        mid_frame_tensor = frames[mid_idx]
        
        frame_np = (mid_frame_tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        logger.info(f"Selecting frame {mid_idx}/{total_frames} as next input.")
        return Image.fromarray(frame_np)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ordered Camera Inference")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bokehK", type=str, default=None)
    parser.add_argument("--focal", type=str, default=None)
    parser.add_argument("--shutter", type=str, default=None)
    parser.add_argument("--color", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    logger.info(f"Enforcing Seed: {args.seed}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    runner = OrderedInferenceChain()
        
    input_args = sys.argv[1:]
    arg_indices = {}
    
    keys_map = {
        '--bokehK': ('bokehK', args.bokehK),
        '--focal': ('focal', args.focal),
        '--shutter': ('shutter', args.shutter),
        '--color': ('color', args.color)
    }

    detected_chain = []
    
    for flag, (config_key, val) in keys_map.items():
        if val is not None:
            try:
                idx = input_args.index(flag)
                detected_chain.append((idx, config_key, val))
            except ValueError:
                detected_chain.append((999, config_key, val))
    
    detected_chain.sort(key=lambda x: x[0])
    
    chain = []
    for _, key, val_str in detected_chain:
        try:
            val_list = json.loads(val_str)
            chain.append((val_list, key))
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON for {key}: {val_str}")
            exit(1)

    if not chain:
        logger.error("No camera parameters provided.")
        exit(1)

    logger.info("Detected Execution Order: " + " -> ".join([k for _, k in chain]))
        
    current_image = None 
    
    for i, (params, config_key) in enumerate(chain):
        setting_type = SETTINGS_CONFIG[config_key]['type_key']
        config_path = SETTINGS_CONFIG[config_key]['config']
        
        logger.info(f"=== Stage {i+1}: {config_key} (Type: {setting_type}) ===")
        
        pipeline, tokenizer, text_encoder = load_models(config_path, runner.device)
        
        output_video = None
        method = ""
        
        if current_image is None:
            output_video = runner.run_t2i(pipeline, tokenizer, text_encoder, args.prompt, params, setting_type)
            method = "T2I"
        else:
            output_video = runner.run_i2i(pipeline, tokenizer, text_encoder, current_image, args.prompt, params, setting_type)
            method = "I2I"
            
        filename = f"stage_{i+1}_{method}_{config_key}.gif"
        save_path = os.path.join(args.output_dir, filename)
        save_videos_grid(output_video, save_path)
        logger.info(f"Saved GIF to {save_path}")
        
        current_image = runner.get_middle_frame(output_video)
        
        del pipeline, tokenizer, text_encoder, output_video
        torch.cuda.empty_cache()

    logger.info("Ordered Inference Chain Completed.")

    '''
    python inference.py   \
        --prompt "A blue sky with mountains."   \
        --color "[5455.0, 5155.0, 5555.0, 6555.0, 7555.0]" \
        --bokehK "[2.44, 8.3, 10.1, 17.2, 24.0]" \
        --shutter "[0.1, 0.3, 0.52, 0.7, 0.8]" \
        --focal "[25.0, 35.0, 45.0, 55.0, 65.0]" \
        --output_dir "inference_output" \
        --seed 42

    python inference.py   \
        --prompt "A variety of potted plants are displayed on a windowsill, \
            with some of them placed in yellow and white bowls. The plants \
            are arranged in a visually appealing manner, creating a pleasant atmosphere in the room."   \
        --bokehK "[2.44, 8.3, 10.1, 17.2, 24.0]" \
        --shutter "[0.1, 0.3, 0.52, 0.7, 0.8]" \
        --color "[5455.0, 5155.0, 5555.0, 6555.0, 7555.0]" \
        --focal "[25.0, 35.0, 45.0, 55.0, 65.0]" \
        --output_dir "inference_output" \
        --seed 42
    '''