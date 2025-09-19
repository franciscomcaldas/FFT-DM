#!/bin/bash

CONFIG_DIR=config
SCRIPT=shs/train.sh

LOGFILE=job_submission.log

echo "Job submission started at $(date)" >> "$LOGFILE"
echo "---------------------------------------" >> "$LOGFILE"

for config_file in "$CONFIG_DIR"/config_*.json; do
    echo "Submitting job for config: $config_file"
    # Check if the config file exists
    # Submit job and capture sbatch output (job ID)
    output=$(sbatch "$SCRIPT" "$config_file")
    
    # Log the output with timestamp and config file
    echo "$(date): $config_file --> $output" >> "$LOGFILE"
    
    sleep 0.2
done

echo "Job submission finished at $(date)" >> "$LOGFILE"
echo "=======================================" >> "$LOGFILE"