#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="$SCRIPT_DIR/results.txt"

> "$OUTPUT_FILE"

for py_file in "$SCRIPT_DIR"/*.py; do
    script_name="$(basename "$py_file")"
    echo "========================================" | tee -a "$OUTPUT_FILE"
    echo "Running: $script_name" | tee -a "$OUTPUT_FILE"
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$OUTPUT_FILE"
    echo "----------------------------------------" | tee -a "$OUTPUT_FILE"

    python3 "$py_file" 2>&1 | tee -a "$OUTPUT_FILE"
    exit_code=${PIPESTATUS[0]}

    echo "----------------------------------------" | tee -a "$OUTPUT_FILE"
    echo "Exit code: $exit_code" | tee -a "$OUTPUT_FILE"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
done

echo "========================================" | tee -a "$OUTPUT_FILE"
echo "All scripts finished. Results saved to: $OUTPUT_FILE"
