import os
import json
import torch
import logging
import argparse
import random
import math
import glob
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
from einops import rearrange
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDIMScheduler
from torch.utils.data import Dataset
from tqdm import tqdm
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
        'json': 'camera_settings/camera_bokehK/annotations/validation.json',
        'list_key': 'bokehK_list',
        'type_key': 'bokeh' 
    },
    'shutter_speed': {
        'config': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_shutter_speed.yaml',
        'json': 'camera_settings/camera_shutter_speed/annotations/validation.json',
        'list_key': 'shutter_speed_list',
        'type_key': 'shutter'
    },
    'focal_length': {
        'config': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_focal_length.yaml',
        'json': 'camera_settings/camera_focal_length/annotations/validation.json',
        'list_key': 'focal_length_list',
        'type_key': 'focal'
    },
    'color_temperature': {
        'config': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_color_temperature.yaml',
        'json': 'camera_settings/camera_color_temperature/annotations/validation.json',
        'list_key': 'color_temperature_list',
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
        idx_val_y = crop_ratio_ys[i] if crop_ratio_ys.ndim == 1 else crop_ratio_ys[i].item()
        idx_val_x = crop_ratio_xs[i] if crop_ratio_xs.ndim == 1 else crop_ratio_xs[i].item()
        crop_h = max(1, min(target_height, torch.round(torch.tensor(idx_val_y) * target_height).int().item()))
        crop_w = max(1, min(target_width, torch.round(torch.tensor(idx_val_x) * target_width).int().item()))
        focal_length_embedding[i, :, center_h - crop_h // 2: center_h + crop_h // 2, center_w - crop_w // 2: center_w + crop_w // 2] = 1.0
    return focal_length_embedding

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

def create_color_temperature_embedding(color_temperature_values, target_height, target_width):
    values = color_temperature_values.cpu()
    f = values.shape[0]
    rgb_factors = [kelvin_to_rgb(2000 + (v.item() * (10000 - 2000))) for v in values.view(-1)]
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


def load_models_inversion(cfg, device):
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

    logger.info("Setting attention processors and Loading LoRAs...")
    unet.set_all_attn_processor(
        add_spatial_lora=cfg.lora_ckpt is not None,
        add_motion_lora=cfg.motion_lora_rank > 0, 
        lora_kwargs={"lora_rank": cfg.lora_rank, "lora_scale": cfg.lora_scale},
        motion_lora_kwargs={"lora_rank": cfg.motion_lora_rank, "lora_scale": cfg.motion_lora_scale},
        **cfg.attention_processor_kwargs
    )

    if cfg.lora_ckpt:
        logger.info(f"Loading Spatial LoRA: {cfg.lora_ckpt}")
        st = torch.load(cfg.lora_ckpt, map_location=device)
        unet.load_state_dict(st['lora_state_dict'] if 'lora_state_dict' in st else st, strict=False)
    
    if cfg.motion_module_ckpt:
        logger.info(f"Loading Motion Module: {cfg.motion_module_ckpt}")
        st = torch.load(cfg.motion_module_ckpt, map_location=device)
        unet.load_state_dict(st, strict=False)
        
    if cfg.camera_adaptor_ckpt:
        logger.info(f"Loading Camera Adaptor: {cfg.camera_adaptor_ckpt}")
        st = torch.load(cfg.camera_adaptor_ckpt, map_location=device)
        camera_encoder.load_state_dict(st['camera_encoder_state_dict'], strict=False)
        unet.load_state_dict(st['attention_processor_state_dict'], strict=False)
    else:
        raise ValueError("Critical: No Camera Adaptor Checkpoint defined!")
        
    pipeline.to(device)
    return pipeline

CHAIN_CONFIGS = {
    'bokehK':            ['bokehK', 'color_temperature', 'shutter_speed', 'focal_length'],
    'shutter_speed':     ['shutter_speed', 'focal_length', 'bokehK', 'color_temperature'],
    'focal_length':      ['focal_length', 'bokehK', 'color_temperature', 'shutter_speed'],
    'color_temperature': ['color_temperature', 'shutter_speed', 'focal_length', 'bokehK']
}

class ExperimentRunner:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pixel_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def save_split_frames(self, video_tensor, metadata_base, save_dir, param_list, setting_key, base_name):
        video_tensor = video_tensor.squeeze(0).permute(1, 0, 2, 3) 
        saved_records = []
        logger.info(f"  Saving {len(video_tensor)} frames to {save_dir}...")
        
        for i, frame in enumerate(video_tensor):
            frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(frame_np)
            val = param_list[i]
            
            filename = f"{base_name}_{setting_key}{val}.png"
            path = os.path.join(save_dir, filename)
            
            img.save(path)
            
            rec = metadata_base.copy()
            rec['filename'] = filename
            rec['local_path'] = path
            new_hist = rec.get('history', []).copy()
            
            new_hist.append({setting_key: val})
            rec['history'] = new_hist
            
            saved_records.append(rec)
            
        return saved_records

    def save_gif(self, video_tensor, save_dir, base_name, setting_key, param_list):
        start = param_list[0]
        end = param_list[-1]
        filename = f"{base_name}_{setting_key}_val{start}_to_{end}.gif"
        save_path = os.path.join(save_dir, filename)
        save_videos_grid(video_tensor, save_path)
        logger.info(f"  Saved GIF to {save_path}")

    def process_batch_i2i(self, pipeline, image_input, prompt, param_list, setting_type):
        video_len = len(param_list)
        target_vals = torch.tensor(param_list, dtype=torch.float32)
        target_embedder = Universal_Camera_Embedding(setting_type, target_vals, pipeline.tokenizer, pipeline.text_encoder, self.device)
        target_emb_raw = target_embedder.load()
        target_emb_3d = rearrange(target_emb_raw, "f c h w -> 1 c f h w")

        if setting_type == 'focal':
            input_pil = image_input.resize((384, 256), Image.Resampling.LANCZOS)
            frames = []
            for val in param_list:
                warped = crop_focal_length(input_pil, 24.0, val, 256, 384)
                frames.append(self.pixel_transforms(warped))
            input_tensor = torch.stack(frames).unsqueeze(0) 
            input_tensor = rearrange(input_tensor, "b f c h w -> b c f h w") 
            inv_image = input_tensor.to(self.device)
            inv_emb = target_emb_3d 
        else:
            input_pil = image_input.resize((384, 256), Image.Resampling.LANCZOS)
            single_tensor = self.pixel_transforms(input_pil) 
            input_tensor = single_tensor.unsqueeze(0).unsqueeze(2).repeat(1, 1, video_len, 1, 1).to(self.device)
            initial_val = param_list[0]
            source_vals = torch.tensor([initial_val] * video_len, dtype=torch.float32)
            source_embedder = Universal_Camera_Embedding(setting_type, source_vals, pipeline.tokenizer, pipeline.text_encoder, self.device)
            source_emb_raw = source_embedder.load()
            source_emb_3d = rearrange(source_emb_raw, "f c h w -> 1 c f h w")
            inv_image = input_tensor
            inv_emb = source_emb_3d

        logger.info(f"[Running Inversion] Steps={self.args.steps}, Frames={video_len}")
        latents = pipeline.invert(
            image=inv_image,
            prompt=prompt,
            camera_embedding=inv_emb,
            num_inference_steps=self.args.steps,
            video_length=video_len
        )

        logger.info(f"[Running Generation] Steps={self.args.steps}, Frames={video_len}")
        output_video = pipeline(
            prompt=prompt,
            latents=latents,
            camera_embedding=target_emb_3d,
            video_length=video_len,
            height=256, width=384,
            num_inference_steps=self.args.steps,
            guidance_scale=self.args.cfg
        ).videos

        return output_video

    def process_batch_t2i(self, pipeline, prompt, param_list, setting_type):
        video_len = len(param_list)
        target_vals = torch.tensor(param_list, dtype=torch.float32)
        target_embedder = Universal_Camera_Embedding(setting_type, target_vals, pipeline.tokenizer, pipeline.text_encoder, self.device)
        target_emb_raw = target_embedder.load()
        target_emb_3d = rearrange(target_emb_raw, "f c h w -> 1 c f h w")
        
        logger.info(f"[Running T2I Generation] Steps={self.args.steps}, Frames={video_len}")
        output_video = pipeline(
            prompt=prompt,
            camera_embedding=target_emb_3d,
            video_length=video_len,
            height=256, width=384,
            num_inference_steps=self.args.steps,
            guidance_scale=self.args.cfg,
            image=None 
        ).videos
        return output_video

    def run_stage_1(self):
        """
        Stage 1: Generates BOTH Baseline (T2I) and Ours (I2I) for ALL settings.
        """
        logger.info("=== Starting Stage 1 (Generating All Baselines & Ours) ===")
        
        for setting_name, info in SETTINGS_CONFIG.items():
            logger.info(f"--- Processing Setting: {setting_name} ---")
            
            cfg = OmegaConf.load(info['config'])
            pipeline = load_models_inversion(cfg, self.device)
            with open(info['json'], 'r') as f: data = json.load(f)

            out_baseline = os.path.join(self.args.output_dir, f"stage1_baseline_{setting_name}")
            out_ours = os.path.join(self.args.output_dir, f"stage1_ours_{setting_name}")
            os.makedirs(out_baseline, exist_ok=True)
            os.makedirs(out_ours, exist_ok=True)
            
            meta_baseline, meta_ours = [], []
            setting_type = info['type_key']
            
            image_root = os.path.join("camera_settings", f"camera_{setting_name}")

            for item in tqdm(data, desc=f"Stg1 {setting_name}"):
                base_img_path = item.get('base_image_path')
                caption = item.get('caption')
                try: param_list = json.loads(item[info['list_key']])
                except: continue
                
                if base_img_path: base_name = os.path.splitext(os.path.basename(base_img_path))[0]
                else: base_name = f"sample_{len(meta_baseline)}"

                res_video_t2i = self.process_batch_t2i(pipeline, caption, param_list, setting_type)
                self.save_gif(res_video_t2i, out_baseline, base_name, setting_type, param_list)
                recs_t = self.save_split_frames(res_video_t2i, {
                    "origin_source": base_img_path,
                    "prompt": caption,
                    "stage": 1,
                    "mode": "T2I_Baseline",
                    "chain_root": setting_name
                }, out_baseline, param_list, setting_type, base_name)
                meta_baseline.extend(recs_t)

                full_img_path = None
                if base_img_path:
                    full_img_path = os.path.join(image_root, base_img_path)

                if full_img_path and os.path.exists(full_img_path):
                    raw_img = Image.open(full_img_path).convert("RGB")
                    res_video_i2i = self.process_batch_i2i(pipeline, raw_img, caption, param_list, setting_type)
                    self.save_gif(res_video_i2i, out_ours, base_name, setting_type, param_list)
                    recs_o = self.save_split_frames(res_video_i2i, {
                        "origin_source": full_img_path,
                        "prompt": caption,
                        "stage": 1,
                        "mode": "I2I_Ours"
                    }, out_ours, param_list, setting_type, base_name)
                    meta_ours.extend(recs_o)
                else:
                    logger.warning(f"⚠️ Skipping Stage 1 I2I: Input image not found at '{full_img_path}'")

            with open(os.path.join(out_baseline, "metadata.json"), "w") as f: json.dump(meta_baseline, f, indent=4)
            with open(os.path.join(out_ours, "metadata.json"), "w") as f: json.dump(meta_ours, f, indent=4)
            
            del pipeline
            torch.cuda.empty_cache()

    def build_image_pool_strict(self, folder_path):
        if not os.path.exists(folder_path):
            logger.warning(f"Missing input folder: {folder_path}")
            return []
        
        meta_path = os.path.join(folder_path, "metadata.json")
        if not os.path.exists(meta_path):
            logger.warning(f"Missing metadata: {meta_path}")
            return []

        pool = []
        with open(meta_path, 'r') as f:
            items = json.load(f)
            for item in items: 
                item['_abs_path'] = os.path.join(folder_path, item['filename'])
                pool.extend(items)
        return pool

    def run_stage_sequential(self, stage_num):
        logger.info(f"=== Starting Stage {stage_num} (Multi-Chain) ===")

        for chain_root, chain_order in CHAIN_CONFIGS.items():
            if stage_num > len(chain_order):
                continue

            current_setting_name = chain_order[stage_num - 1] 
            prev_setting_name = chain_order[stage_num - 2]
            
            if stage_num == 2:
                input_dir_name = f"stage1_baseline_{prev_setting_name}"
            else:
                input_dir_name = f"stage{stage_num-1}_{prev_setting_name}"

            input_path = os.path.join(self.args.output_dir, input_dir_name)
            logger.info(f"🔗 Chain [{chain_root}] | Processing '{current_setting_name}' using input from: {input_dir_name}")

            pool = self.build_image_pool_strict(input_path)
            if not pool:
                logger.warning(f"Skipping chain '{chain_root}': Input pool empty.")
                continue

            info = SETTINGS_CONFIG[current_setting_name]
            cfg = OmegaConf.load(info['config'])
            pipeline = load_models_inversion(cfg, self.device)
            setting_type = info['type_key']

            with open(info['json'], 'r') as f: json_data = json.load(f)

            out_dir_name = f"stage{stage_num}_{current_setting_name}"
            out_dir = os.path.join(self.args.output_dir, out_dir_name)
            os.makedirs(out_dir, exist_ok=True)
            new_metadata = []

            for item in tqdm(json_data, desc=f"[{chain_root}] Stg{stage_num}"):
                try: param_list = json.loads(item[info['list_key']])
                except: continue
                
                candidates = []
                for p_item in pool:
                    hist_keys = [list(h.keys())[0] for h in p_item.get('history', [])]
                    if setting_type not in hist_keys:
                        candidates.append(p_item)
                
                if not candidates: continue

                src_item = random.choice(candidates)
                src_path = src_item['_abs_path']
                if not os.path.exists(src_path): continue
                
                src_img = Image.open(src_path).convert("RGB")
                src_prompt = src_item['prompt']
                base_name = os.path.splitext(src_item['filename'])[0]

                res_video = self.process_batch_i2i(pipeline, src_img, src_prompt, param_list, setting_type)
                
                self.save_gif(res_video, out_dir, base_name, setting_type, param_list)
                recs = self.save_split_frames(res_video, {
                    "origin_source": src_item.get('origin_source', ''),
                    "prompt": src_prompt,
                    "stage": stage_num,
                    "chain_root": chain_root,
                    "parent_image": src_item['filename'],
                    "history": src_item.get('history', [])
                }, out_dir, param_list, setting_type, base_name)
                
                new_metadata.extend(recs)
            
            with open(os.path.join(out_dir, "metadata.json"), "w") as f: json.dump(new_metadata, f, indent=4)
            del pipeline
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--base_model", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5") 
    parser.add_argument("--output_dir", type=str, default="experiments_final")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    runner = ExperimentRunner(args)
    
    if args.stage == 1:
        runner.run_stage_1()
    else:
        runner.run_stage_sequential(args.stage)

'''
python experiment_runner.py \
  --stage 1 \
  --output_dir "experiments_final" \
  --seed 42

python experiment_runner.py \
  --stage 2 \
  --output_dir "experiments_final" \
  --seed 42

python experiment_runner.py \
  --stage 3 \
  --output_dir "experiments_final" \
  --seed 42

python experiment_runner.py \
  --stage 4 \
  --output_dir "experiments_final" \
  --seed 42
'''