import os
import json
import argparse

def update_config(config):
    # Modify diffusion_config
    config["diffusion_config"]["T"] = 200

    config['train_config']["max_components"] =0

    config['gen_config']['max_components_gen']=1

    # Modify train_config output directory
    if "output_directory" in config["train_config"]:
        config["train_config"]["output_directory"] = config["train_config"]["output_directory"].replace(
            "./results", "./results_OG"
        )

    # Modify gen_config output directory
    if "output_directory" in config["gen_config"]:
        config["gen_config"]["output_directory"] = config["gen_config"]["output_directory"].replace(
            "./results", "./results_OG"
        )
    if "ckpt_path" in config["gen_config"]:
        config["gen_config"]["ckpt_path"] = config["gen_config"]["ckpt_path"].replace(
            "./results", "./results_OG"
        )

    return config

def process_configs(directory):
    for filename in os.listdir(directory):
        if filename.endswith("FDM.json"):
            filepath = os.path.join(directory, filename)

            with open(filepath, "r") as f:
                config = json.load(f)

            updated_config = update_config(config)

            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_N0{ext}"
            new_filepath = os.path.join(directory, new_filename)

            with open(new_filepath, "w") as f:
                json.dump(updated_config, f, indent=4)

            print(f"[✓] Processed: {filename} → {new_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update config files for baseline variant.")
    parser.add_argument("--config_dir", required=True, help="Directory containing .json config files.")
    args = parser.parse_args()

    process_configs(args.config_dir)