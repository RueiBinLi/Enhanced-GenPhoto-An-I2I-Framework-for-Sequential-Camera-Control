import json
import subprocess
import os
import sys

# ==========================================
# 1. Configuration Mapping
# ==========================================
# Maps task type to JSON filename, python script, config, and argument flags.
TASKS = {
    'bokeh': {
        'json_file': './camera_settings/camera_bokehK/annotations',
        'script': 'inference_bokehK.py',
        'config': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_bokehK.yaml',
        'param_arg': '--bokehK_list',
        'json_key': 'bokehK_list'
    },
    'focal': {
        'json_file': './camera_settings/camera_focal_length/annotations',
        'script': 'inference_focal_length.py',
        'config': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_focal_length.yaml',
        'param_arg': '--focal_length_list',
        'json_key': 'focal_length_list'
    },
    'shutter': {
        'json_file': './camera_settings/camera_shutter_speed/annotations',
        'script': 'inference_shutter_speed.py',
        'config': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_shutter_speed.yaml',
        'param_arg': '--shutter_speed_list',
        'json_key': 'shutter_speed_list'
    },
    'color': {
        'json_file': './camera_settings/camera_color_temperature/annotations',
        'script': 'inference_color_temperature.py',
        'config': 'configs/inference_genphoto/adv3_256_384_genphoto_relora_color_temperature.yaml',
        'param_arg': '--color_temperature_list',
        'json_key': 'color_temperature_list'
    }
}

def main():
    print("=== Starting Batch Inference for Author Ground Truth ===\n")

    for task_name, settings in TASKS.items():
        json_path = settings['json_file']
        
        if not os.path.exists(json_path):
            print(f"Skipping '{task_name}': File '{json_path}' not found.")
            continue

        print(f"Found '{task_name}' task. Reading {json_path}...")
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading JSON {json_path}: {e}")
            continue

        for i, entry in enumerate(data):
            caption = entry.get('caption')
            # Extract the parameter list string (e.g., "[1.0, 5.0...]")
            param_values = entry.get(settings['json_key'])

            if not caption or not param_values:
                print(f"   [Skipping Entry {i}] Missing caption or parameter list.")
                continue

            print(f"   Running {i+1}/{len(data)}: {caption[:40]}...")

            # Construct command
            cmd = [
                sys.executable, settings['script'],
                "--config", settings['config'],
                "--base_scene", caption,
                settings['param_arg'], str(param_values)
            ]

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"   Execution failed for prompt: {caption[:20]}")
                print(f"   Error: {e}")
            except Exception as e:
                print(f"   Unexpected error: {e}")

    print("\n=== All tasks completed. ===")

if __name__ == "__main__":
    main()