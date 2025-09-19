#!/bin/bash
# Requirements: jq must be installed

CONFIG_FOLDER="config"

for file in "$CONFIG_FOLDER"/*.json; do
    echo "Processing $file..."
    
    # Create temp file
    tmp=$(mktemp)

    jq '
      .train_config.n_iters = ((.train_config.n_iters // 0) + 10000)
    ' "$file" > "$tmp" && mv "$tmp" "$file"

    echo "Updated n_iters in $file."
done

echo "All done."