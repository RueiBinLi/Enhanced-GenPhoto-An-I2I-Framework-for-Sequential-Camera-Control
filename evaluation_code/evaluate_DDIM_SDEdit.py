import os
import argparse
import math
import numpy as np
import cv2
import glob
import sys
import torch
import torch.nn.functional as F
import json
import csv
from PIL import Image, ImageSequence
sys.path.append('genphoto/data/BokehMe/')

VALS_SETTINGS = {
    'color':  [5455.0, 5155.0, 5555.0, 6555.0, 7555.0],
    'shutter': [0.1, 0.3, 0.52, 0.7, 0.8],
    'focal':   [25.0, 35.0, 45.0, 55.0, 65.0],
    'bokeh':   [2.44, 8.3, 10.1, 17.2, 24.0]
}


class LPIPS_Evaluator:
    def __init__(self):
        try:
            import lpips
        except ImportError:
            print("[Error] 'lpips' library not installed. LPIPS metrics will be skipped.")
            self.model = None
            return

        print("Loading LPIPS model (VGG)...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = lpips.LPIPS(net='vgg').to(self.device)
        self.model.eval()

    def preprocess_frame(self, pil_image):
        img_np = np.array(pil_image.convert('RGB')).astype(np.float32)
        img_norm = (img_np / 255.0) * 2 - 1
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device)

    def calculate_gif_consistency(self, frames):
        """Calculates average LPIPS distance between consecutive frames."""
        if self.model is None or len(frames) < 2: return None
        scores = []
        try:
            with torch.no_grad():
                for i in range(len(frames) - 1):
                    t1 = self.preprocess_frame(frames[i])
                    t2 = self.preprocess_frame(frames[i+1])
                    dist = self.model(t1, t2).item()
                    scores.append(dist)
            return np.mean(scores)
        except Exception as e:
            print(f"  [LPIPS Error]: {e}")
            return None

class CLIPEvaluator:
    def __init__(self, prompt_text):
        try:
            import clip
        except ImportError:
            print("[Error] 'clip' library not installed. CLIP metrics will be skipped.")
            self.model = None
            return

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading CLIP model (ViT-B/32) on {self.device}...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        
        print(f"Encoding Prompt: '{prompt_text}'")
        self.text_features = None
        if prompt_text:
            text_tokens = clip.tokenize([prompt_text]).to(self.device)
            with torch.no_grad():
                self.text_features = self.model.encode_text(text_tokens)

    def calculate_gif_score(self, frames):
        if self.model is None or self.text_features is None or not frames: return None
        frame_scores = []
        try:
            with torch.no_grad():
                for frame in frames:
                    image_input = self.preprocess(frame).unsqueeze(0).to(self.device)
                    image_features = self.model.encode_image(image_input)
                    similarity = F.cosine_similarity(image_features, self.text_features, dim=1)
                    frame_scores.append(similarity.item())
            return np.mean(frame_scores)
        except Exception as e:
            print(f"  [CLIP Error]: {e}")
            return None

def kelvin_to_rgb_smooth(kelvin):
    temp = kelvin / 100.0
    if temp <= 66:
        red = 255
        green = 99.4708025861 * np.log(temp) - 161.1195681661 if temp > 0 else 0
        if temp <= 19: blue = 0
        else: blue = 138.5177312231 * np.log(temp - 10) - 305.0447927307
    elif 66 < temp <= 88:
        red = 0.5 * (255 + 329.698727446 * ((temp - 60) ** -0.19332047592))
        green = 0.5 * (288.1221695283 * ((temp - 60) ** -0.1155148492) + (99.4708025861 * np.log(temp) - 161.1195681661 if temp > 0 else 0))
        blue = 0.5 * (138.5177312231 * np.log(temp - 10) - 305.0447927307 + 255)
    else:
        red = 329.698727446 * ((temp - 60) ** -0.19332047592)
        green = 288.1221695283 * ((temp - 60) ** -0.1155148492)
        blue = 255
    return np.array([np.clip(red, 0, 255), np.clip(green, 0, 255), np.clip(blue, 0, 255)], dtype=np.float32)

def interpolate_white_balance(image_rgb_np, kelvin, **kwargs):
    balance_rgb = kelvin_to_rgb_smooth(kelvin)
    image = image_rgb_np.astype(np.float32)
    r = image[:, :, 0] * (balance_rgb[0] / 255.0)
    g = image[:, :, 1] * (balance_rgb[1] / 255.0)
    b = image[:, :, 2] * (balance_rgb[2] / 255.0)
    balanced = cv2.merge([r, g, b])
    return np.clip(balanced, 0, 255).astype(np.uint8)

def sensor_image_simulation_numpy(image_rgb_np, shutter_speed, **kwargs):
    fwc = 32000
    gain = 1
    Nbits = 8
    avg_PPP = (0.6 * shutter_speed + 0.1) * fwc
    photon_flux = image_rgb_np.astype(np.float32)
    min_val, max_val = 0, 2 ** Nbits - 1
    mean_flux = np.mean(photon_flux) + 1e-6
    theta = photon_flux * (avg_PPP / mean_flux)
    theta = np.clip(theta, 0, fwc)
    theta = np.round(theta * gain * max_val / fwc)
    return np.clip(theta, min_val, max_val).astype(np.uint8)

def crop_focal_length(image_np, focal_length, **kwargs):
    img = Image.fromarray(image_np.astype(np.uint8))
    width, height = img.size
    sensor_w, sensor_h = 36.0, 24.0
    base_focal_length = 24.0
    
    base_x_fov = 2.0 * math.atan(sensor_w * 0.5 / base_focal_length)
    base_y_fov = 2.0 * math.atan(sensor_h * 0.5 / base_focal_length)
    target_x_fov = 2.0 * math.atan(sensor_w * 0.5 / focal_length)
    target_y_fov = 2.0 * math.atan(sensor_h * 0.5 / focal_length)

    crop_ratio = min(target_x_fov / base_x_fov, target_y_fov / base_y_fov)
    crop_w = int(round(crop_ratio * width))
    crop_h = int(round(crop_ratio * height))
    crop_w, crop_h = max(1, min(width, crop_w)), max(1, min(height, crop_h))

    left, top = (width - crop_w) // 2, (height - crop_h) // 2
    right, bottom = left + crop_w, top + crop_h

    zoomed = img.crop((left, top, right, bottom))
    return np.array(zoomed.resize((width, height), Image.Resampling.LANCZOS)).astype(np.uint8)

def simulate_bokeh(image_np, K, depth_map_path=None, **kwargs):
    try:
        from classical_renderer.scatter import ModuleRenderScatter
    except ImportError:
        print(f"  [Warn] genphoto/Bokeh renderer not found. Skipping Bokeh sim.")
        return image_np

    if depth_map_path is None or not os.path.exists(depth_map_path):
        return image_np
        
    disp = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    disp /= 255.0
    disp = (disp - disp.min()) / (disp.max() - disp.min())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    renderer = ModuleRenderScatter().to(device)
    
    img_tensor = torch.from_numpy(image_np.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
    disp_focus = 0.96
    defocus = K * (disp - disp_focus) / 10.0
    defocus_tensor = torch.from_numpy(defocus).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        bokeh, _ = renderer(img_tensor**2.2, defocus_tensor*10.0)
    
    bokeh = bokeh ** (1/2.2)
    bokeh = bokeh.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(bokeh * 255, 0, 255).astype(np.uint8)

SIM_FUNCS = {
    'color': interpolate_white_balance,
    'shutter': sensor_image_simulation_numpy,
    'focal': crop_focal_length,
    'bokeh': simulate_bokeh
}

def metric_color_acc(ref_frames, sample_frames):
    def get_val(f): return np.mean(np.array(f.convert('RGB')), axis=(0, 1))
    changes1 = np.diff([get_val(f) for f in ref_frames], axis=0)
    changes2 = np.diff([get_val(f) for f in sample_frames], axis=0)
    corrs = []
    for i in range(3):
        if np.std(changes1[:, i]) > 1e-6 and np.std(changes2[:, i]) > 1e-6:
            corrs.append(np.corrcoef(changes1[:, i], changes2[:, i])[0, 1])
    return np.mean(corrs) if corrs else 0.0

def metric_shutter_acc(ref_frames, sample_frames):
    def get_val(f): return np.mean(np.array(f.convert('L')))
    diff1 = np.diff([get_val(f) for f in ref_frames])
    diff2 = np.diff([get_val(f) for f in sample_frames])
    if np.std(diff1) > 1e-6 and np.std(diff2) > 1e-6:
        return np.corrcoef(diff1, diff2)[0, 1]
    return 0.0

def metric_focal_acc(ref_frames, sample_frames):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    def get_scale(f1, f2):
        g1 = cv2.cvtColor(np.array(f1), cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(np.array(f2), cv2.COLOR_RGB2GRAY)
        kp1, des1 = sift.detectAndCompute(g1, None)
        kp2, des2 = sift.detectAndCompute(g2, None)
        if des1 is None or des2 is None: return 0
        matches = bf.match(des1, des2)
        if len(matches) < 4: return 0
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        return np.sqrt(M[0, 0]**2 + M[0, 1]**2) if M is not None else 0

    sc1 = [get_scale(ref_frames[i], ref_frames[i+1]) for i in range(len(ref_frames)-1)]
    sc2 = [get_scale(sample_frames[i], sample_frames[i+1]) for i in range(len(sample_frames)-1)]
    valid = [i for i, s in enumerate(sc2) if s > 0]
    if len(valid) < 2: return 0.0
    c1 = [sc1[i] for i in valid]
    c2 = [sc2[i] for i in valid]
    return np.corrcoef(c1, c2)[0, 1] if (np.std(c1) > 1e-6 and np.std(c2) > 1e-6) else 0.0

def metric_bokeh_acc(ref_frames, sample_frames):
    def get_blur(f): return cv2.Laplacian(np.array(f.convert('L')), cv2.CV_64F).var()
    diff1 = np.diff([get_blur(f) for f in ref_frames])
    diff2 = np.diff([get_blur(f) for f in sample_frames])
    if np.std(diff1) > 1e-6 and np.std(diff2) > 1e-6:
        return np.corrcoef(diff1, diff2)[0, 1]
    return 0.0

METRIC_FUNCS = {
    'color': metric_color_acc,
    'shutter': metric_shutter_acc,
    'focal': metric_focal_acc,
    'bokeh': metric_bokeh_acc
}

def generate_depth_map(image_input, output_path, cache_dir="./model_cache"):
    if os.path.exists(output_path): return output_path
    print(f"  > Generating depth map: {output_path}...")
    try:
        from transformers import pipeline
        import torch
        os.makedirs(cache_dir, exist_ok=True)
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", cache_dir=cache_dir, device=device)
        
        image = Image.open(image_input) if isinstance(image_input, str) else Image.fromarray(image_input.astype(np.uint8))
        depth = pipe(image)["depth"]
        depth.save(output_path)
        return output_path
    except Exception as e:
        print(f"  [Error] Failed to generate depth map: {e}")
        return None

def load_frames(path):
    with Image.open(path) as gif:
        return [frame.copy().convert('RGB') for frame in ImageSequence.Iterator(gif)]

def find_file(directory, pattern):
    matches = glob.glob(os.path.join(directory, pattern))
    return matches[0] if matches else None

def save_gif(frames, path):
    if frames:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=200, loop=0)

def detect_stages(folder_name):
    known_stages = ['color', 'shutter', 'focal', 'bokeh']
    found = []
    parts = folder_name.lower().replace('/', '_').split('_')
    for p in parts:
        for k in known_stages:
            if k in p and k not in found:
                found.append(k)
    return found

def get_depth_for_simulation(setting_name, image_source, depth_folder, file_suffix):
    if setting_name != 'bokeh': return None
    os.makedirs(depth_folder, exist_ok=True)
    depth_path = os.path.join(depth_folder, f"depth_{file_suffix}.jpg")
    return generate_depth_map(image_source, depth_path)

def process_single_folder(data_dir, base_image_path, output_dir, folder_id, accumulators, evaluators):
    stages = detect_stages(folder_id)
    if len(stages) != 3: return

    method_output_dir = os.path.join(output_dir, folder_id)
    depth_cache_dir = os.path.join(method_output_dir, "depth_maps")
    os.makedirs(method_output_dir, exist_ok=True)
    
    base_img_cv = cv2.imread(base_image_path)
    if base_img_cv is None: print(f"Error: Base image not found at {base_image_path}"); return
    base_rgb = cv2.cvtColor(base_img_cv, cv2.COLOR_BGR2RGB)

    lpips_eval, clip_eval = evaluators

    s1_name = stages[0]
    s1_vals = VALS_SETTINGS[s1_name]
    
    d_map_s1 = get_depth_for_simulation(s1_name, base_rgb, depth_cache_dir, "stage1_base")
    ref_s1 = [Image.fromarray(SIM_FUNCS[s1_name](base_rgb, v, depth_map_path=d_map_s1)) for v in s1_vals]
    save_gif(ref_s1, os.path.join(method_output_dir, "refs", "ref_stage1.gif"))
    
    sample_s1 = []
    for i in range(len(s1_vals)):
        path = find_file(data_dir, f"*_{i}_0*.gif")
        if path: 
            frames = load_frames(path)
            sample_s1.append(frames[0])
            
            if lpips_eval: accumulators[s1_name]['lpips'].append(lpips_eval.calculate_gif_consistency(frames))
            if clip_eval:  accumulators[s1_name]['clip'].append(clip_eval.calculate_gif_score(frames))

    if len(sample_s1) == len(ref_s1):
        acc = METRIC_FUNCS[s1_name](ref_s1, sample_s1)
        accumulators[s1_name]['sim_acc'].append(acc)

    s2_name = stages[1]
    s2_vals = VALS_SETTINGS[s2_name]
    s2_sim_scores = []

    for i, s1_val in enumerate(s1_vals):
        init_path = find_file(data_dir, f"*_{i}_0*.gif")
        if not init_path: continue
        base_frame = np.array(load_frames(init_path)[0]) 

        d_map_s2 = get_depth_for_simulation(s2_name, base_frame, depth_cache_dir, f"stage2_group_{i}")
        ref_s2 = [Image.fromarray(SIM_FUNCS[s2_name](base_frame, v, depth_map_path=d_map_s2)) for v in s2_vals]

        sample_s2 = []
        for j in range(len(s2_vals)):
            path = find_file(data_dir, f"*_{i}_{j}*.gif")
            if path: 
                frames = load_frames(path)
                sample_s2.append(frames[0])
                
                if lpips_eval: accumulators[s2_name]['lpips'].append(lpips_eval.calculate_gif_consistency(frames))
                if clip_eval:  accumulators[s2_name]['clip'].append(clip_eval.calculate_gif_score(frames))
        
        if len(sample_s2) == len(ref_s2):
            s2_sim_scores.append(METRIC_FUNCS[s2_name](ref_s2, sample_s2))

    if s2_sim_scores: accumulators[s2_name]['sim_acc'].append(np.mean(s2_sim_scores))

    s3_name = stages[2]
    s3_vals = VALS_SETTINGS[s3_name]
    s3_sim_scores = []
    
    for path in sorted(glob.glob(os.path.join(data_dir, "*.gif"))):
        if "ref_" in os.path.basename(path): continue
        frames = load_frames(path)
        
        if lpips_eval: accumulators[s3_name]['lpips'].append(lpips_eval.calculate_gif_consistency(frames))
        if clip_eval:  accumulators[s3_name]['clip'].append(clip_eval.calculate_gif_score(frames))

        if len(frames) == len(s3_vals):
            base_frame = np.array(frames[0]) 
            fname_safe = os.path.basename(path).replace('.gif', '')
            d_map_s3 = get_depth_for_simulation(s3_name, base_frame, depth_cache_dir, f"stage3_{fname_safe}")
            ref_s3 = [Image.fromarray(SIM_FUNCS[s3_name](base_frame, v, depth_map_path=d_map_s3)) for v in s3_vals]
            s3_sim_scores.append(METRIC_FUNCS[s3_name](ref_s3, frames))

    if s3_sim_scores: accumulators[s3_name]['sim_acc'].append(np.mean(s3_sim_scores))


def print_unified_report(title, results_dict):
    print("\n" + "="*80)
    print(f"      {title}      ")
    print("="*80)
    print(f"{'Setting':<10} | {'Sim Acc (Corr)':<15} | {'LPIPS (Low=Good)':<18} | {'CLIP (High=Good)':<18} | {'Count'}")
    print("-" * 80)

    for setting, metrics in results_dict.items():
        sim = np.mean(metrics['sim_acc']) if metrics['sim_acc'] else 0.0
        lpips_score = np.nanmean(metrics['lpips']) if metrics['lpips'] else 0.0
        clip_score = np.nanmean(metrics['clip']) if metrics['clip'] else 0.0
        count = len(metrics['sim_acc'])
        
        sim_str = f"{sim:.4f}" if metrics['sim_acc'] else "N/A"
        lpips_str = f"{lpips_score:.4f}" if metrics['lpips'] else "N/A"
        clip_str = f"{clip_score:.4f}" if metrics['clip'] else "N/A"
        
        print(f"{setting.capitalize():<10} | {sim_str:<15} | {lpips_str:<18} | {clip_str:<18} | {count}")
    print("="*80 + "\n")

def save_results(output_dir, all_results):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "metrics_summary.csv")
    print(f"\n[Saving] Writing summary to: {csv_path}")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Setting', 'Sim_Acc_Corr', 'LPIPS_LowIsGood', 'CLIP_HighIsGood', 'Sample_Count'])

        for method, settings in all_results.items():
            for setting_name, metrics in settings.items():
                sim = np.mean(metrics['sim_acc']) if metrics['sim_acc'] else 0.0
                lpips_score = np.nanmean(metrics['lpips']) if metrics['lpips'] else 0.0
                clip_score = np.nanmean(metrics['clip']) if metrics['clip'] else 0.0
                count = len(metrics['sim_acc'])

                writer.writerow([
                    method,
                    setting_name,
                    f"{sim:.5f}",
                    f"{lpips_score:.5f}",
                    f"{clip_score:.5f}",
                    count
                ])

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    json_path = os.path.join(output_dir, "metrics_raw.json")
    print(f"[Saving] Writing raw data to: {json_path}")
    
    with open(json_path, 'w') as f:
        json.dump(all_results, f, cls=NumpyEncoder, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation (Simulation Acc, LPIPS, CLIP)")
    parser.add_argument("--root_dir", required=True, help="Experiments folder")
    parser.add_argument("--base_image", required=True, help="Original input image path")
    parser.add_argument("--prompt", required=True, help="Text prompt for CLIP")
    parser.add_argument("--output_dir", default="Unified_Results")
    args = parser.parse_args()

    lpips_eval = LPIPS_Evaluator()
    clip_eval = CLIPEvaluator(args.prompt)
    evaluators = (lpips_eval, clip_eval)

    results = {
        "DDIM": {k: {'sim_acc': [], 'lpips': [], 'clip': []} for k in VALS_SETTINGS.keys()},
        "SDEdit": {k: {'sim_acc': [], 'lpips': [], 'clip': []} for k in VALS_SETTINGS.keys()}
    }

    print(f"\nScanning {args.root_dir}...")
    for root, dirs, files in os.walk(args.root_dir):
        if "DDIM_Inversion" in dirs:
            print(f"Processing DDIM folder: {os.path.basename(root)}")
            process_single_folder(
                os.path.join(root, "DDIM_Inversion"), 
                args.base_image, 
                args.output_dir, 
                f"{os.path.basename(root)}_DDIM",
                results["DDIM"],
                evaluators
            )
        if "SDEdit" in dirs:
            print(f"Processing SDEdit folder: {os.path.basename(root)}")
            process_single_folder(
                os.path.join(root, "SDEdit"), 
                args.base_image, 
                args.output_dir, 
                f"{os.path.basename(root)}_SDEdit",
                results["SDEdit"],
                evaluators
            )

    print_unified_report("FINAL REPORT: DDIM INVERSION", results["DDIM"])
    print_unified_report("FINAL REPORT: SDEdit", results["SDEdit"])

    save_results(args.output_dir, results)

if __name__ == "__main__":
    main()

'''
python evaluation_code/evaluate_DDIM_SDEdit.py \
  --root_dir "Experiments/ablation_3" \
  --base_image "input_image/my_park_photo.jpg" \
  --prompt "A photo of a park with green grass and trees" \
  --output_dir "Evaluation_Results"
'''