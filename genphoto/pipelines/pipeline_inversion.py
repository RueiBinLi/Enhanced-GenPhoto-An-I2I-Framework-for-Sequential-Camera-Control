import torch
from typing import Union, List, Optional
from genphoto.pipelines.pipeline_animation import GenPhotoPipeline
from einops import rearrange

class GenPhotoInversionPipeline(GenPhotoPipeline):
    
    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt: Union[str, List[str]],
        camera_embedding: torch.Tensor, 
        num_inference_steps: int = 50,
        num_videos_per_prompt: int = 1,
        video_length: int = 5,
        generator: Optional[torch.Generator] = None,
        multidiff_total_steps: int = 1,
        multidiff_overlaps: int = 12,
    ):
        """
        執行 DDIM Inversion: 從原圖 (x0) 反推回雜訊 (z_T)
        """
        device = self.unet.device
        dtype = self.unet.dtype
        
        # 1. 準備 Camera Embedding
        if camera_embedding.ndim == 5:
            bs_embed = camera_embedding.shape[0]
            camera_embedding_features = self.camera_encoder(camera_embedding.to(device, dtype=dtype)) 
            camera_embedding_features = [
                rearrange(x, '(b f) c h w -> b c f h w', b=bs_embed) 
                for x in camera_embedding_features
            ]
        else:
            raise ValueError(f"Camera embedding shape error: {camera_embedding.shape}")

        # 2. Encode Image to Latents
        # 使用 .mode() 確保確定性 (Deterministic)，這是還原的關鍵
        image = image.to(device=device, dtype=self.vae.dtype)
        init_latents = self.vae.encode(image).latent_dist.mode()
        init_latents = self.vae.config.scaling_factor * init_latents

        if init_latents.ndim == 4:
            init_latents = init_latents.unsqueeze(2).repeat(1, 1, video_length, 1, 1)
        
        latents = init_latents.to(device=device, dtype=dtype)

        # 3. Prepare Timesteps
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance=False, negative_prompt=None
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        # 取得完整的 timesteps (例如: 1, 41, ..., 961)
        timesteps = reversed(self.scheduler.timesteps)
        timesteps = list(timesteps)

        # [CRITICAL FIX] 修正 "All Noise" 問題
        # Generation 從 t=961 開始，所以我們只需要 Invert 到 t=961。
        # 原本的程式碼多跑了一步 (961 -> 1000)，導致輸入給 Generation 的雜訊過大。
        # 我們移除最後一個 step，只執行 timesteps[:-1]。
        run_timesteps = timesteps[:-1]

        print(f"[Inversion] Running {len(run_timesteps)} steps (Stopping at t={run_timesteps[-1]}) to match Generation start.")

        # 4. Inversion Loop
        with self.progress_bar(total=len(run_timesteps)) as progress_bar:
            for i, t in enumerate(run_timesteps):
                # 預測雜訊
                noise_pred = self.unet(
                    latents, 
                    t, 
                    encoder_hidden_states=text_embeddings,
                    camera_embedding_features=camera_embedding_features
                ).sample

                # 準備計算參數 (Float32)
                current_t = t
                # 下一步是 timesteps 列表中的下一個元素
                # 例如: t=1 -> next_t=41; ...; t=921 -> next_t=961
                next_t = timesteps[i + 1]

                alpha_prod_t = self.scheduler.alphas_cumprod[current_t].to(device=device, dtype=torch.float32)
                beta_prod_t = 1 - alpha_prod_t
                
                alpha_prod_t_next = self.scheduler.alphas_cumprod[next_t].to(device=device, dtype=torch.float32)
                beta_prod_t_next = 1 - alpha_prod_t_next
                
                latents_fp32 = latents.to(dtype=torch.float32)
                noise_pred_fp32 = noise_pred.to(dtype=torch.float32)

                # DDIM Inversion Equation
                # 1. 預測 x0
                f_theta = (latents_fp32 - beta_prod_t ** 0.5 * noise_pred_fp32) / (alpha_prod_t ** 0.5)
                
                # 2. 推導下一步 Latent (t -> next_t)
                latents_next = alpha_prod_t_next ** 0.5 * f_theta + beta_prod_t_next ** 0.5 * noise_pred_fp32

                latents = latents_next.to(dtype=dtype)
                progress_bar.update()

        return latents