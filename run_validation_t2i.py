import json
import subprocess
import argparse
import os
import sys

# ==========================================
# 1. Configuration Map
# ==========================================
# Maps the 'setting_type' to all necessary details:
# - script: The author's python file
# - config: The author's YAML config
# - arg_flag: The command line flag for the parameter list
# - json_key: The key inside validation.json to find the list
SETTINGS_MAP = {
    'bokeh': {
        'script': 'inference_bokehK.py',
        'config': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_bokehK.yaml',
        'arg_flag': '--bokehK_list',
        'json_key': 'bokehK_list'
    },
    'focal': {
        'script': 'inference_focal_length.py',
        'config': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_focal_length.yaml',
        'arg_flag': '--focal_length_list',
        'json_key': 'focal_length_list' # Assumed key based on pattern
    },
    'shutter': {
        'script': 'inference_shutter_speed.py',
        'config': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_shutter_speed.yaml',
        'arg_flag': '--shutter_speed_list',
        'json_key': 'shutter_speed_list' # Assumed key based on pattern
    },
    'color': {
        'script': 'inference_color_temperature.py',
        'config': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_color_temperature.yaml',
        'arg_flag': '--color_temperature_list',
        'json_key': 'color_temperature_list' # Assumed key based on pattern
    }
}

def run_validation(setting_type, json_path):
    if setting_type not in SETTINGS_MAP:
        print(f"Error: Unknown setting type '{setting_type}'. Choose from {list(SETTINGS_MAP.keys())}")
        return

    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    # Load Configuration
    setting_cfg = SETTINGS_MAP[setting_type]
    
    # Load JSON Data
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from {json_path}")
    print(f"Target Script: {setting_cfg['script']}")

    # Iterate through each entry in the JSON
    for i, entry in enumerate(data):
        caption = entry.get('caption')
        
        # Get the parameter list string (e.g., "[1.0, 5.0, ...]")
        # We try to get it using the specific key, fallback to generic if named differently in your json
        param_list_str = entry.get(setting_cfg['json_key'])
        
        if not caption or not param_list_str:
            print(f"Skipping entry {i}: Missing caption or {setting_cfg['json_key']}")
            continue

        print(f"\n[{i+1}/{len(data)}] Processing: {caption[:40]}...")
        print(f"   Params: {param_list_str}")

        # Construct Command
        # Note: We ignore 'base_image_path' because Author's code is T2I (generates from scratch)
        cmd = [
            sys.executable, setting_cfg['script'],
            "--config", setting_cfg['config'],
            "--base_scene", caption,
            setting_cfg['arg_flag'], param_list_str
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"   Error running inference for entry {i}: {e}")
        except KeyboardInterrupt:
            print("\n   Stopped by user.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Author's T2I Inference using validation.json")
    parser.add_argument("--type", type=str, required=True, choices=['bokeh', 'focal', 'shutter', 'color'], 
                        help="The camera setting type (bokeh, focal, shutter, color)")
    parser.add_argument("--json", type=str, required=True, help="Path to the validation.json file")
    
    args = parser.parse_args()
    
    run_validation(args.type, args.json)