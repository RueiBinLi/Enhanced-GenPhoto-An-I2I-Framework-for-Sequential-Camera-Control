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
from torch.utils.data import Dataset

from genphoto.pipelines.pipeline_inversion import GenPhotoInversionPipeline
from genphoto.models.unet import UNet3DConditionModelCameraCond
from genphoto.models.camera_adaptor import CameraCameraEncoder, CameraAdaptor
from genphoto.utils.util import save_videos_grid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_bokehK_embedding(bokehK_values, target_height, target_width):
    f = bokehK_values.shape[0]
    bokehK_embedding = torch.zeros((f, 3, target_height, target_width), dtype=bokehK_values.dtype)
    
    for i in range(f):
        K_value = bokehK_values[i].item()
        kernel_size = max(K_value, 1)
        sigma = K_value / 3.0

        ax = np.linspace(-(kernel_size / 2), kernel_size / 2, int(np.ceil(kernel_size)))
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= np.sum(kernel)
        scale = kernel[int(np.ceil(kernel_size) / 2), int(np.ceil(kernel_size) / 2)]
        
        bokehK_embedding[i] = scale
    
    return bokehK_embedding

class Camera_Embedding(Dataset):
    def __init__(self, bokehK_values, tokenizer, text_encoder, device, sample_size=[256, 384]):
        self.bokehK_values = bokehK_values.to(device)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device  
        self.sample_size = sample_size

    def load(self):
        
        prompts = []
        for bb in self.bokehK_values:
            prompt = f"<bokeh kernel size: {bb.item()}>"
            prompts.append(prompt)
        
        with torch.no_grad():
            prompt_ids = self.tokenizer(
                prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)

            encoder_hidden_states = self.text_encoder(input_ids=prompt_ids).last_hidden_state

        differences = []
        for i in range(1, encoder_hidden_states.size(0)):
            diff = encoder_hidden_states[i] - encoder_hidden_states[i - 1]
            diff = diff.unsqueeze(0)
            differences.append(diff)  

        final_diff = encoder_hidden_states[-1] - encoder_hidden_states[0]
        final_diff = final_diff.unsqueeze(0)
        differences.append(final_diff)

        concatenated_differences = torch.cat(differences, dim=0)
        
        pad_length = 128 - concatenated_differences.size(1)
        if pad_length > 0:
            concatenated_differences = F.pad(concatenated_differences, (0, 0, 0, pad_length))

        frame = concatenated_differences.size(0)
        ccl_embedding = concatenated_differences.reshape(frame, self.sample_size[0], self.sample_size[1])
        ccl_embedding = ccl_embedding.unsqueeze(1)  
        ccl_embedding = ccl_embedding.expand(-1, 3, -1, -1)
        ccl_embedding = ccl_embedding.to(self.device)
        
        bokehK_embedding_tensor = create_bokehK_embedding(self.bokehK_values, self.sample_size[0], self.sample_size[1]).to(self.device)
        camera_embedding = torch.cat((bokehK_embedding_tensor, ccl_embedding), dim=1)
        return camera_embedding


def load_models_inversion(cfg):
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
    
    pipeline = GenPhotoInversionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        camera_encoder=camera_encoder
    ).to(device)
    
    pipeline.enable_vae_slicing()

    logger.info("Setting the attention processors...")

    unet.set_all_attn_processor(
        add_spatial_lora=cfg.lora_ckpt is not None,
        add_motion_lora=cfg.motion_lora_rank > 0,
        lora_kwargs={"lora_rank": cfg.lora_rank, "lora_scale": cfg.lora_scale},
        motion_lora_kwargs={"lora_rank": cfg.motion_lora_rank, "lora_scale": cfg.motion_lora_scale},
        **cfg.attention_processor_kwargs
    )

    if cfg.lora_ckpt is not None:
        logger.info(f"Loading LoRA from {cfg.lora_ckpt}")
        lora_state = torch.load(cfg.lora_ckpt, map_location=device)
        if 'lora_state_dict' in lora_state: lora_state = lora_state['lora_state_dict']
        unet.load_state_dict(lora_state, strict=False)

    if cfg.motion_module_ckpt is not None:
        logger.info(f"Loading Motion Module from {cfg.motion_module_ckpt}")
        mm_state = torch.load(cfg.motion_module_ckpt, map_location=device)
        unet.load_state_dict(mm_state, strict=False)

    if cfg.camera_adaptor_ckpt is not None:
        logger.info(f"Loading Camera Adaptor from {cfg.camera_adaptor_ckpt}")
        ca_state = torch.load(cfg.camera_adaptor_ckpt, map_location=device)
        camera_encoder.load_state_dict(ca_state['camera_encoder_state_dict'], strict=False)
        unet.load_state_dict(ca_state['attention_processor_state_dict'], strict=False)

    pipeline.to(device)
    return pipeline, device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--base_scene", type=str, required=True, help="Prompt text")
    parser.add_argument("--bokehK_list", type=str, required=True, help="Target bokeh values list (JSON)")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="outputs/ddim_inversion", help="Output directory")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    pipeline, device = load_models_inversion(cfg)
    
    raw_image = Image.open(args.input_image).convert("RGB").resize((384, 256))
    image_tensor = transforms.ToTensor()(raw_image).unsqueeze(0).to(device)
    image_tensor = image_tensor * 2.0 - 1.0

    target_vals = json.loads(args.bokehK_list)
    video_len = len(target_vals)
    
    initial_val = target_vals[0]
    source_vals = [initial_val] * video_len
    
    source_bokeh_tensor = torch.tensor(source_vals).unsqueeze(1)
    source_cam_embed_obj = Camera_Embedding(source_bokeh_tensor, pipeline.tokenizer, pipeline.text_encoder, device)
    source_camera_embedding = source_cam_embed_obj.load()
    source_camera_embedding = rearrange(source_camera_embedding.unsqueeze(0), "b f c h w -> b c f h w")

    target_bokeh_tensor = torch.tensor(target_vals).unsqueeze(1)
    target_cam_embed_obj = Camera_Embedding(target_bokeh_tensor, pipeline.tokenizer, pipeline.text_encoder, device)
    target_camera_embedding = target_cam_embed_obj.load()
    target_camera_embedding = rearrange(target_camera_embedding.unsqueeze(0), "b f c h w -> b c f h w")

    logger.info("Running DDIM Inversion with STATIC camera parameters...")
    inverted_latents = pipeline.invert(
        image=image_tensor,
        prompt=args.base_scene,
        camera_embedding=source_camera_embedding,
        num_inference_steps=25,
        video_length=video_len
    )

    logger.info("Running Generation with DYNAMIC camera parameters...")
    with torch.no_grad():
        output = pipeline(
            prompt=args.base_scene,
            camera_embedding=target_camera_embedding,
            video_length=video_len,
            height=256,
            width=384,
            num_inference_steps=25,
            guidance_scale=2.0,
            latents=inverted_latents
        ).videos[0]

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "ddim_result.gif")
    save_videos_grid(output[None, ...], save_path)
    logger.info(f"Saved result to {save_path}")

if __name__ == "__main__":
    main()
'''
python inference_ddim_2.py \
  --config configs/inference_genphoto/adv3_256_384_genphoto_relora_bokehK.yaml \
  --base_scene "A photo of a park with green grass and trees" \
  --bokehK_list "[2.44, 8.3, 10.1, 17.2, 24.0]" \
  --input_image ./input_image/my_park_photo.jpg \
  --output_dir outputs/ddim_test_2
'''