#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="$SCRIPT_DIR/output.txt"

> "$OUTPUT_FILE"

for py_file in "$SCRIPT_DIR"/*.py; do
    echo "========================================" | tee -a "$OUTPUT_FILE"
    echo "Running: $(basename "$py_file")" | tee -a "$OUTPUT_FILE"
    echo "========================================" | tee -a "$OUTPUT_FILE"
    python3 "$py_file" 2>&1 | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
done

echo "Done. Output saved to $OUTPUT_FILE"
