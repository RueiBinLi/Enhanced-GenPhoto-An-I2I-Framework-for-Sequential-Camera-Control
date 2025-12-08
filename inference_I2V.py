import os
import torch
import logging
import argparse
import json
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
from einops import rearrange
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDIMScheduler

# 沿用原本的 imports
from genphoto.pipelines.pipeline_animation import GenPhotoPipeline
from genphoto.models.unet import UNet3DConditionModelCameraCond
from genphoto.models.camera_adaptor import CameraCameraEncoder, CameraAdaptor
from genphoto.utils.util import save_videos_grid
# 這裡 import 你的 embedding 處理類別 (假設在 inference_bokehK.py 裡面有定義或你把它移到了 utils)
# 為了方便，這裡我將 inference_bokehK.py 中的 Camera_Embedding 複製過來改寫，或者你可以直接 import 它
from inference_bokehK import Camera_Embedding
import inspect
print(f"DEBUG: Loading GenPhotoPipeline from: {inspect.getfile(GenPhotoPipeline)}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_models(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    camera_adaptor = CameraAdaptor(unet, camera_encoder)
    camera_adaptor.requires_grad_(False)
    camera_adaptor.to(device)

    logger.info("Setting the attention processors")
    unet.set_all_attn_processor(
        add_spatial_lora=cfg.lora_ckpt is not None,
        add_motion_lora=cfg.motion_lora_rank > 0,
        lora_kwargs={"lora_rank": cfg.lora_rank, "lora_scale": cfg.lora_scale},
        motion_lora_kwargs={"lora_rank": cfg.motion_lora_rank, "lora_scale": cfg.motion_lora_scale},
        **cfg.attention_processor_kwargs
    )

    if cfg.lora_ckpt is not None:
        print(f"Loading the lora checkpoint from {cfg.lora_ckpt}")
        lora_checkpoints = torch.load(cfg.lora_ckpt, map_location=unet.device)
        if 'lora_state_dict' in lora_checkpoints.keys():
            lora_checkpoints = lora_checkpoints['lora_state_dict']
        _, lora_u = unet.load_state_dict(lora_checkpoints, strict=False)
        assert len(lora_u) == 0
        print(f'Loading done')

    if cfg.motion_module_ckpt is not None:
        print(f"Loading the motion module checkpoint from {cfg.motion_module_ckpt}")
        mm_checkpoints = torch.load(cfg.motion_module_ckpt, map_location=unet.device)
        _, mm_u = unet.load_state_dict(mm_checkpoints, strict=False)
        assert len(mm_u) == 0
        print("Loading done")
    

    if cfg.camera_adaptor_ckpt is not None:
        logger.info(f"Loading camera adaptor from {cfg.camera_adaptor_ckpt}")
        camera_adaptor_checkpoint = torch.load(cfg.camera_adaptor_ckpt, map_location=device)
        camera_encoder_state_dict = camera_adaptor_checkpoint['camera_encoder_state_dict']
        attention_processor_state_dict = camera_adaptor_checkpoint['attention_processor_state_dict']
        camera_enc_m, camera_enc_u = camera_adaptor.camera_encoder.load_state_dict(camera_encoder_state_dict, strict=False)
        assert len(camera_enc_m) == 0 and len(camera_enc_u) == 0
        _, attention_processor_u = camera_adaptor.unet.load_state_dict(attention_processor_state_dict, strict=False)
        assert len(attention_processor_u) == 0
        
        logger.info("Camera Adaptor loading done")
    else:
        logger.info("No Camera Adaptor checkpoint used")

    pipeline = GenPhotoPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        camera_encoder=camera_encoder
    ).to(device)
    pipeline.enable_vae_slicing
    return pipeline, device

def preprocess_image(image_path, height, width):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        # [CRITICAL FIX] 必須給 3 個 mean 和 3 個 std，對應 R, G, B
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    ])
    return transform(image).unsqueeze(0)

def run_i2v_inference(pipeline, tokenizer, text_encoder, base_scene, bokehK_list, input_image_path, strength, output_dir, device, video_length=5, height=256, width=384):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 準備輸入圖片
    init_image = preprocess_image(input_image_path, height, width).to(device)

    # 2. 準備相機參數 (這裡示範 Bokeh，若要混用其他參數可在此擴充)
    bokehK_list_str = bokehK_list
    bokehK_values = json.loads(bokehK_list_str)
    bokehK_values = torch.tensor(bokehK_values).unsqueeze(1)
    
    # 建立 Camera Embedding
    # 注意: Camera_Embedding 類別來自 inference_bokehK.py，確保它能被 import
    camera_embedding_obj = Camera_Embedding(bokehK_values, tokenizer, text_encoder, device, sample_size=[height, width])
    camera_embedding = camera_embedding_obj.load()
    camera_embedding = rearrange(camera_embedding.unsqueeze(0), "b f c h w -> b c f h w")

    logger.info(f"Running Image-to-Video with strength: {strength}")

    # 3. 執行 Pipeline (新增了 image 和 strength 參數)
    with torch.no_grad():
        sample = pipeline(
            prompt=base_scene,
            camera_embedding=camera_embedding,
            video_length=video_length,
            height=height,
            width=width,
            num_inference_steps=25,
            guidance_scale=8.0,
            image=init_image,    # <--- 輸入圖片
            strength=strength    # <--- 重繪強度 (建議 0.6 - 0.9)
        ).videos[0]

    sample_save_path = os.path.join(output_dir, f"i2v_sample_str{strength}.gif")
    save_videos_grid(sample[None, ...], sample_save_path)
    logger.info(f"Saved generated I2V sample to {sample_save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--base_scene", type=str, required=True, help="Prompt")
    parser.add_argument("--bokehK_list", type=str, required=True, help="Camera settings (JSON)")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--strength", type=float, default=0.75, help="Denoising strength (0.0-1.0)")
    parser.add_argument("--output_dir", type=str, default="outputs/i2v_test")
    
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    pipeline, device = load_models(cfg) # 重用 inference_bokehK 的 load_models
    
    run_i2v_inference(
        pipeline, pipeline.tokenizer, pipeline.text_encoder, 
        args.base_scene, args.bokehK_list, args.input_image, 
        args.strength, args.output_dir, device
    )
if __name__ == "__main__":
    main()

'''
python inference_I2V.py \
  --config configs/inference_genphoto/adv3_256_384_genphoto_relora_bokehK.yaml \
  --base_scene "A photo of a park with green grass and trees" \
  --bokehK_list "[2.44, 8.3, 10.1, 17.2, 24.0]" \
  --input_image ./input_image/my_park_photo.jpg \
  --strength 0.8
'''
