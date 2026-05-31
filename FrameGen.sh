#!/bin/bash

echo "Frame generation started"

BASE_DIR="./bitstream/base"
ENHANCE_DIR="./bitstream/enhance"
OUTPUT_DIR="results"

PYTHON_SCRIPT="main.py"
MODEL_PATH="./YUV_SR/checkpoints_y/best.pth"

BASE_WIDTH=1920
BASE_HEIGHT=1080

ENH_WIDTH=3840
ENH_HEIGHT=2160

ONLY_EDSR=true

naive=false



mkdir -p "$OUTPUT_DIR"
if [ "$naive" = true ]; then
    mkdir -p "warped_hr"
fi

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

    # output name
    output_name="${base_name/.layer0.yuv/_generated_4k.layer1.yuv}"
    if [ "$naive" = true ]; then
        output_file="warped_hr/$output_name"
    else
        output_file="$OUTPUT_DIR/$output_name"
    fi

    echo "----------------------------------------"
    echo "Base       : $base_file"
    echo "Enhancement: $enhance_file"
    echo "Output     : $output_file"
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
        --model_path "$MODEL_PATH" \
        --base_width "$BASE_WIDTH" \
        --base_height "$BASE_HEIGHT" \
        --enh_width "$ENH_WIDTH" \
        --enh_height "$ENH_HEIGHT" \
        $( [ "$ONLY_EDSR" = true ] && echo "--only_edsr" )\
        $( [ "$naive" = true ] && echo "--naive" )

    echo "Finished: $base_name"
done

echo "All files finished."