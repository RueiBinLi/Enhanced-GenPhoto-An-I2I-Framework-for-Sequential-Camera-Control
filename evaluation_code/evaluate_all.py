import os
import argparse
import json
import math
import numpy as np
import cv2
import glob
import sys
import re
from PIL import Image, ImageSequence
sys.path.append(os.getcwd())

if os.path.exists("clip.py"):
    print("\n[WARNING] Found 'clip.py' in the current directory! Please rename it.\n")

try:
    import lpips
    import clip
    import torch
    import torch.nn.functional as F
except ImportError:
    print("\n[ERROR] Missing libraries. Install: pip install lpips git+https://github.com/openai/CLIP.git")
    sys.exit(1)

MANUAL_PARAMS = {
    'bokeh': [
        [1.0, 5.0, 10.0, 15.0, 20.0], 
        [14.0, 4.0, 24.0, 5.0, 15.0], 
        [25.0, 18.0, 12.0, 8.0, 2.0],
        [2.0, 6.0, 12.0, 18.0, 24.3] 
    ],
    'color': [
        [5455.0, 5155.0, 5555.0, 6555.0, 7555.0],
        [4000.0, 3500.0, 3000.0, 2500.0, 2000.0],
        [4500.0, 5500.0, 6500.0, 7500.0, 8500.0]
    ],
    'focal': [
        [25.1, 36.1, 47.1, 58.1, 69.1],
        [64.0, 53.0, 35.0, 50.0, 70.0],
        [25.0, 36.0, 47.0, 58.0, 69.0]
    ],
    'shutter': [
        [0.11, 0.22, 0.33, 0.44, 0.55],
        [0.94, 0.75, 0.55, 0.42, 0.21],
        [0.29, 0.49, 0.69, 0.79, 0.89]
    ]
}

GALLERY_PATHS = {
    'bokeh':   ["camera_settings/camera_bokehK/gallery"],
    'color':   ["camera_settings/camera_color_temperature/gallery"],
    'focal':   ["camera_settings/camera_focal_length/gallery"],
    'shutter': ["camera_settings/camera_shutter_speed/gallery"],
}

VALIDATION_PATHS = {
    'bokeh': { 'paths': ["camera_settings/camera_bokehK/annotations/validation.json"], 'key': 'bokehK_list' },
    'color': { 'paths': ["camera_settings/camera_temperature/annotations/validation.json", "camera_settings/camera_color_temperature/annotations/validation.json"], 'key': 'color_temperature_list' },
    'focal': { 'paths': ["camera_settings/camera_focal_length/annotations/validation.json"], 'key': 'focal_length_list' },
    'shutter': { 'paths': ["camera_settings/camera_shutter_speed/annotations/validation.json"], 'key': 'shutter_speed_list' }
}

DEPTH_DIR = os.path.join("camera_settings", "camera_bokehK", "depth")

def kelvin_to_rgb_smooth(kelvin):
    temp = kelvin / 100.0
    if temp <= 66:
        red = 255
        green = 99.4708 * np.log(temp) - 161.119 if temp > 0 else 0
        if temp <= 19: blue = 0
        else: blue = 138.517 * np.log(temp - 10) - 305.044
    else:
        red = 329.698 * ((temp - 60) ** -0.193)
        green = 288.122 * ((temp - 60) ** -0.115)
        blue = 255
    return np.array([np.clip(red, 0, 255), np.clip(green, 0, 255), np.clip(blue, 0, 255)], dtype=np.float32)

def interpolate_white_balance(image_np, kelvin, **kwargs):
    balance_rgb = kelvin_to_rgb_smooth(kelvin)
    image = image_np.astype(np.float32)
    r = image[:, :, 0] * (balance_rgb[0] / 255.0)
    g = image[:, :, 1] * (balance_rgb[1] / 255.0)
    b = image[:, :, 2] * (balance_rgb[2] / 255.0)
    return np.clip(cv2.merge([r, g, b]), 0, 255).astype(np.uint8)

def sensor_image_simulation(image_np, shutter, **kwargs):
    fwc = 32000
    avg_PPP = (0.6 * shutter + 0.1) * fwc
    photon_flux = image_np.astype(np.float32)
    mean_flux = np.mean(photon_flux) + 1e-6
    theta = photon_flux * (avg_PPP / mean_flux)
    theta = np.clip(theta, 0, fwc)
    theta = np.round(theta * 1 * 255 / fwc)
    return np.clip(theta, 0, 255).astype(np.uint8)

def crop_focal_length(image_np, focal, **kwargs):
    img = Image.fromarray(image_np.astype(np.uint8))
    w, h = img.size
    sensor_w, sensor_h = 36.0, 24.0
    base_focal = 24.0
    base_fov_x = 2 * math.atan(sensor_w / (2 * base_focal))
    base_fov_y = 2 * math.atan(sensor_h / (2 * base_focal))
    target_fov_x = 2 * math.atan(sensor_w / (2 * focal))
    target_fov_y = 2 * math.atan(sensor_h / (2 * focal))
    crop_ratio = min(target_fov_x/base_fov_x, target_fov_y/base_fov_y)
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    left, top = (w - cw)//2, (h - ch)//2
    cropped = img.crop((left, top, left+cw, top+ch))
    return np.array(cropped.resize((w, h), Image.Resampling.LANCZOS))

def simulate_bokeh(image_np, k_val, depth_map=None, **kwargs):
    if depth_map is None: return image_np
    try:
        from genphoto.data.BokehMe.classical_renderer.scatter import ModuleRenderScatter
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if isinstance(depth_map, str):
            disp = cv2.imread(depth_map, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        else:
            disp = depth_map.astype(np.float32)
            
        disp /= 255.0
        disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)
        
        img_t = torch.from_numpy(image_np.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
        disp_focus = 0.96 
        defocus = k_val * (disp - disp_focus) / 10.0
        defocus_t = torch.from_numpy(defocus).unsqueeze(0).unsqueeze(0).to(device)
        
        renderer = ModuleRenderScatter().to(device)
        with torch.no_grad():
            bokeh, _ = renderer(img_t**2.2, defocus_t*10.0)
        
        bokeh = bokeh.squeeze(0).permute(1, 2, 0).cpu().numpy() ** (1/2.2)
        return np.clip(bokeh * 255, 0, 255).astype(np.uint8)
    except Exception:
        return image_np

SIMULATORS = {
    'color': interpolate_white_balance,
    'shutter': sensor_image_simulation,
    'focal': crop_focal_length,
    'bokeh': simulate_bokeh
}

def load_all_prompts(data_root):
    """
    Loads all prompts from validation.json files into a dictionary.
    Structure: { 'bokeh': {'000101': 'prompt'}, 'focal': {'000101': 'prompt'}, ... }
    """
    print("Loading prompts from validation files...")
    prompts_db = {}
    
    for setting_key, info in VALIDATION_PATHS.items():
        prompts_db[setting_key] = {}
        for rel_path in info['paths']:
            full_path = os.path.join(data_root, rel_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                        for item in data:
                            base_path = item.get('base_image_path', '')
                            img_id = os.path.splitext(os.path.basename(base_path))[0]
                            caption = item.get('caption', '').strip()
                            if img_id and caption:
                                prompts_db[setting_key][img_id] = caption
                except Exception as e:
                    print(f"[Error] Failed to load JSON {full_path}: {e}")
            else:
                pass
                
    return prompts_db

class Evaluator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading Evaluator Models on {self.device}...")
        self.lpips = lpips.LPIPS(net='vgg').to(self.device).eval()
        self.clip_model, self.clip_prep = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()

    def calc_accuracy(self, ref_frames, sam_frames, setting):
        if len(ref_frames) < 2 or len(sam_frames) < 2: return 0.0
        try:
            if setting == 'color':
                def get_means(fs): return np.array([np.mean(f, axis=(0,1)) for f in fs])
                d_ref, d_sam = np.diff(get_means(ref_frames), axis=0), np.diff(get_means(sam_frames), axis=0)
                corrs = [np.corrcoef(d_ref[:,i], d_sam[:,i])[0,1] for i in range(3) if np.std(d_ref[:,i]) > 1e-5]
                return np.mean(corrs) if corrs else 0.0
            
            elif setting == 'shutter':
                def get_val(fs): return [np.mean(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)) for f in fs]
                v_ref, v_sam = np.diff(get_val(ref_frames)), np.diff(get_val(sam_frames))
                return np.corrcoef(v_ref, v_sam)[0,1] if np.std(v_ref) > 1e-5 else 0.0
            
            elif setting == 'bokeh':
                def get_blur(fs): return [cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in fs]
                v_ref, v_sam = np.diff(get_blur(ref_frames)), np.diff(get_blur(sam_frames))
                return np.corrcoef(v_ref, v_sam)[0,1] if np.std(v_ref) > 1e-5 else 0.0
            
            elif setting == 'focal':
                sift = cv2.SIFT_create()
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                def get_scales(fs):
                    scales = []
                    for i in range(len(fs)-1):
                        g1 = cv2.cvtColor(fs[i], cv2.COLOR_RGB2GRAY)
                        g2 = cv2.cvtColor(fs[i+1], cv2.COLOR_RGB2GRAY)
                        kp1, des1 = sift.detectAndCompute(g1, None)
                        kp2, des2 = sift.detectAndCompute(g2, None)
                        if des1 is None or des2 is None: scales.append(0); continue
                        matches = bf.match(des1, des2)
                        if len(matches) < 4: scales.append(0); continue
                        src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                        dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                        M, _ = cv2.estimateAffinePartial2D(src, dst)
                        scales.append(np.sqrt(M[0,0]**2 + M[0,1]**2) if M is not None else 0)
                    return scales
                s_ref = get_scales(ref_frames)
                s_sam = get_scales(sam_frames)
                valid = [i for i, v in enumerate(s_sam) if v > 0]
                if len(valid) < 2: return 0.0
                return np.corrcoef([s_ref[i] for i in valid], [s_sam[i] for i in valid])[0, 1]
        except Exception:
            return 0.0
        return 0.0

    def calc_lpips(self, ref_frames, sam_frames):
        if len(sam_frames) < 2: return 0.0
        scores = []
        with torch.no_grad():
            prev_t = torch.tensor(sam_frames[0].astype(np.float32)/255*2-1).permute(2,0,1).unsqueeze(0).to(self.device)
            
            for i in range(1, len(sam_frames)):
                curr_t = torch.tensor(sam_frames[i].astype(np.float32)/255*2-1).permute(2,0,1).unsqueeze(0).to(self.device)
                
                dist = self.lpips(prev_t, curr_t).item()
                scores.append(dist)
                
                prev_t = curr_t
                
        return np.mean(scores) if scores else 0.0

    def calc_clip(self, sam_frames, prompt):
        if not prompt: return 0.0
        scores = []
        with torch.no_grad():
            text_tok = clip.tokenize([prompt], truncate=True).to(self.device)
            text_emb = self.clip_model.encode_text(text_tok)
            for f in sam_frames:
                pil_img = Image.fromarray(f)
                img_in = self.clip_prep(pil_img).unsqueeze(0).to(self.device)
                img_emb = self.clip_model.encode_image(img_in)
                scores.append(F.cosine_similarity(img_emb, text_emb).item())
        return np.mean(scores) if scores else 0.0

def resolve_path(root, options):
    for p in options:
        full_p = os.path.join(root, p)
        if os.path.exists(full_p): return full_p
    return None

def find_base_image_smart(gif_name, all_folders, current_setting):
    name_no_ext = os.path.splitext(gif_name)[0]
    token = f"_{current_setting}"
    
    if token in name_no_ext:
        base_prefix = name_no_ext.split(token)[0]
        for folder in all_folders:
            candidate_png = os.path.join(folder, f"{base_prefix}.png")
            candidate_jpg = os.path.join(folder, f"{base_prefix}.jpg")
            if os.path.exists(candidate_png): return candidate_png
            if os.path.exists(candidate_jpg): return candidate_jpg
    return None

def find_matching_param_list(gif_name, setting):
    pattern = rf"{setting}_val([0-9\.]+)"
    match = re.search(pattern, gif_name)
    
    if not match:
        pattern = rf"{setting}([0-9\.]+)"
        match = re.search(pattern, gif_name)

    if match:
        try:
            start_val = float(match.group(1))
            if setting in MANUAL_PARAMS:
                for candidate_list in MANUAL_PARAMS[setting]:
                    if not candidate_list: continue
                    if math.isclose(candidate_list[0], start_val, abs_tol=0.1):
                        return candidate_list
        except:
            return None
    return None

def process_experiment_folder(folder_path, root_dir, data_root, evaluator, prompts_db):
    fname = os.path.basename(folder_path).lower()
    setting = next((s for s in ['color', 'shutter', 'focal', 'bokeh'] if s in fname), None)
    if not setting: return None

    stage = 1
    if "stage2" in fname: stage = 2
    if "stage3" in fname: stage = 3

    print(f"\nEvaluating: {os.path.basename(folder_path)} (Type: {setting.upper()}, Stage: {stage})")
    
    all_exp_folders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    scores = {'acc': [], 'lpips': [], 'clip': []}
    
    found_gifs = glob.glob(os.path.join(folder_path, "*.gif"))
    found_gifs = [g for g in found_gifs if "ref" not in os.path.basename(g)]

    if not found_gifs:
        print("  [Warn] No GIFs found.")
        return None

    for gif_path in found_gifs:
        gif_name = os.path.basename(gif_path)
        img_id = gif_name.split('_')[0] 

        sim_params = find_matching_param_list(gif_name, setting)
        if not sim_params:
            continue

        try:
            with Image.open(gif_path) as gif:
                sam_frames = [np.array(f.convert('RGB')) for f in ImageSequence.Iterator(gif)]
        except: continue
        
        if not sam_frames: continue
        target_h, target_w = sam_frames[0].shape[:2]

        base_img_full = None
        if stage == 1:
            for task in GALLERY_PATHS:
                for gp in GALLERY_PATHS[task]:
                    full_gp = os.path.join(data_root, gp)
                    base_p = os.path.join(full_gp, f"{img_id}.png")
                    if not os.path.exists(base_p): base_p = os.path.join(full_gp, f"{img_id}.jpg")
                    if os.path.exists(base_p):
                        base_img_full = cv2.cvtColor(cv2.imread(base_p), cv2.COLOR_BGR2RGB)
                        break
                if base_img_full is not None: break
        else:
            found_base_path = find_base_image_smart(gif_name, all_exp_folders, setting)
            if found_base_path:
                base_img_full = cv2.cvtColor(cv2.imread(found_base_path), cv2.COLOR_BGR2RGB)
            else:
                for task in GALLERY_PATHS:
                    for gp in GALLERY_PATHS[task]:
                        full_gp = os.path.join(data_root, gp)
                        base_p = os.path.join(full_gp, f"{img_id}.png")
                        if not os.path.exists(base_p): base_p = os.path.join(full_gp, f"{img_id}.jpg")
                        if os.path.exists(base_p):
                            base_img_full = cv2.cvtColor(cv2.imread(base_p), cv2.COLOR_BGR2RGB)
                            break
                    if base_img_full is not None: break

        if base_img_full is None:
            print(f"  [Skip] {img_id}: Base image missing.")
            continue

        depth_data_full = None
        if setting == 'bokeh':
            d_paths = [
                os.path.join(data_root, DEPTH_DIR, f"{img_id}.png")
            ]
            d_path = next((p for p in d_paths if os.path.exists(p)), None)
            
            if d_path:
                depth_data_full = cv2.imread(d_path, cv2.IMREAD_GRAYSCALE)
                if depth_data_full.shape[:2] != base_img_full.shape[:2]:
                    depth_data_full = cv2.resize(depth_data_full, (base_img_full.shape[1], base_img_full.shape[0]), interpolation=cv2.INTER_NEAREST)
            elif stage == 1:
                continue 

        ref_frames = []
        for val in sim_params:
            sim_full = SIMULATORS[setting](base_img_full, val, depth_map=depth_data_full)
            sim_resized = cv2.resize(sim_full, (target_w, target_h), interpolation=cv2.INTER_AREA)
            ref_frames.append(sim_resized)

        min_len = min(len(sam_frames), len(ref_frames))
        if min_len < 2: continue
        sam_frames = sam_frames[:min_len]
        ref_frames = ref_frames[:min_len]

        
        prompt = "A photo" # for initial
        source_setting = None
        
        name_parts = os.path.splitext(gif_name)[0].split('_')
        
        known_keys = ['bokeh', 'focal', 'shutter', 'color']
        
        if len(name_parts) > 1:
            for part in name_parts[1:]:
                for k in known_keys:
                    if part.startswith(k):
                        source_setting = k
                        break
                if source_setting: break
        
        if source_setting and source_setting in prompts_db:
            if img_id in prompts_db[source_setting]:
                prompt = prompts_db[source_setting][img_id]
        
        if prompt == "A photo":
            for k in known_keys:
                if k in prompts_db and img_id in prompts_db[k]:
                    prompt = prompts_db[k][img_id]
                    break
        
        
        acc = evaluator.calc_accuracy(ref_frames, sam_frames, setting)
        lpips_score = evaluator.calc_lpips(ref_frames, sam_frames)
        clip_score = evaluator.calc_clip(sam_frames, prompt)

        scores['acc'].append(acc)
        scores['lpips'].append(lpips_score)
        scores['clip'].append(clip_score)

    if not scores['acc']:
        print("  [Warn] No valid sequences processed.")
        return None

    return {
        'name': os.path.basename(folder_path),
        'acc': np.mean(scores['acc']),
        'lpips': np.mean(scores['lpips']),
        'clip': np.mean(scores['clip'])
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--data_root", default=".")
    parser.add_argument("--output_dir", default="Evaluation_Results")
    args = parser.parse_args()

    prompts_db = load_all_prompts(args.data_root)

    evaluator = Evaluator()
    results = []

    for entry in sorted(os.scandir(args.root_dir), key=lambda x: x.name):
        if entry.is_dir():
            res = process_experiment_folder(entry.path, args.root_dir, args.data_root, evaluator, prompts_db)
            if res: results.append(res)

    if results:
        print("\n" + "="*75)
        print(f"{'EXPERIMENT FOLDER':<45} | {'ACC':<7} | {'LPIPS':<7} | {'CLIP':<7}")
        print("="*75)
        for r in results:
            print(f"{r['name']:<45} | {r['acc']:.4f}  | {r['lpips']:.4f}  | {r['clip']:.4f}\n")
        print("="*75)
        
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "final_summary.txt"), "w") as f:
            f.write(f"{'EXPERIMENT FOLDER':<45} | {'ACC':<7} | {'LPIPS':<7} | {'CLIP':<7}\n")
            for r in results:
                f.write(f"{r['name']:<45} | {r['acc']:.4f}  | {r['lpips']:.4f}  | {r['clip']:.4f}\n")

if __name__ == "__main__":
    main()

'''
python evaluation_code/evaluate_all.py \
  --root_dir "experiments_final" \
  --data_root "." \
  --output_dir "Evaluation_Results_test"
'''