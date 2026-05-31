#!/bin/bash

echo "Model inference started"

BASE_DIR="./bitstream/base"
ENHANCE_DIR="./bitstream/enhance"
OUTPUT_DIR="results"

PYTHON_SCRIPT="model_inference.py"

BASE_WIDTH=1920
BASE_HEIGHT=1080

ENH_WIDTH=3840
ENH_HEIGHT=2160

mkdir -p "$OUTPUT_DIR"

for base_file in "$BASE_DIR"/odd_*.layer0.yuv
do
    if [ ! -f "$base_file" ]; then
        echo "No base-layer YUV files found in $BASE_DIR"
        exit 1
    fi

    base_name=$(basename "$base_file")

    # odd_xxx.layer0.yuv -> even_xxx.layer1.yuv
    enhance_name="${base_name/odd_/even_}"
    enhance_name="${enhance_name/.layer0.yuv/.layer1.yuv}"

    enhance_file="$ENHANCE_DIR/$enhance_name"

    # output YUV name
    output_name="${base_name/.layer0.yuv/_generated_4k.layer1.yuv}"
    output_file="$OUTPUT_DIR/$output_name"

    # PNG frames directory (derived from YUV output path, strip .yuv)
    png_dir="${output_file%.yuv}_frames"

    echo "----------------------------------------"
    echo "Base       : $base_file"
    echo "Enhancement: $enhance_file"
    echo "Output YUV : $output_file"
    echo "Output PNG : $png_dir"
    echo "----------------------------------------"

    if [ ! -f "$enhance_file" ]; then
        echo "Warning: enhancement file not found, skip:"
        echo "$enhance_file"
        continue
    fi

    python "$PYTHON_SCRIPT" \
        --base "$base_file" \
        --enhancement "$enhance_file" \
        --output "$output_file" \
        --base_width "$BASE_WIDTH" \
        --base_height "$BASE_HEIGHT" \
        --enh_width "$ENH_WIDTH" \
        --enh_height "$ENH_HEIGHT" \
        --png_output_dir "$png_dir"

    echo "Finished: $base_name"
done

echo "All files finished."
