#!/bin/bash

CONFIG_DIR="./config"
OUTPUT_FILE="times_table.csv"

echo "Model,Dataset,OG,Batch Size,Time per batch,Subfolder" > "$OUTPUT_FILE"

for config_file in "$CONFIG_DIR"/config_*.json; do
    output_dir=$(jq -r '.train_config.output_directory' "$config_file")

    # Extract values from diffusion_config
    T=$(jq -r '.diffusion_config.T' "$config_file")
    beta_0=$(jq -r '.diffusion_config.beta_0' "$config_file")
    beta_T=$(jq -r '.diffusion_config.beta_T' "$config_file")
    n=$(jq -r '.train_config.max_components' "$config_file")
    echo n="$n"
    # Format beta_0 and beta_T to remove scientific notation and match folder format
    beta_0_fmt=$(printf "%.4f" "$beta_0")
    #beta_T_fmt=$(printf "%.2f" "$beta_T")

    # Construct the specific subfolder name
    subfolder_name="T${T}_beta0${beta_0_fmt}_betaT${beta_T}_n${n}"
    subfolder="${output_dir}/${subfolder_name}/"

    echo "Processing subfolder: $subfolder"
    results_file="${subfolder}results.txt"
    if [[ ! -f "$results_file" ]]; then
        echo "No results file in $subfolder"
        continue
    fi

    # Extract metrics
    time_epoch=$(grep -m1 "Time per batch:" "$results_file" | awk '{print $4}')
    batch_size=$(grep -m1 "Batch size:" "$results_file" | awk '{print $3}')

    # Parse structure
    model=$(basename "$(dirname "$(dirname "$(dirname "$subfolder")")")")
    dataset=$(basename "$(dirname "$(dirname "$subfolder")")")

    if [[ "$model" == *OG* ]]; then
        og_flag="OG"
    else
        og_flag="No-OG"
    fi

    echo "$model,$dataset,$og_flag,$batch_size,$time_epoch,$subfolder_name" >> "$OUTPUT_FILE"
done