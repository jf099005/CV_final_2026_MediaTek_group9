#!/bin/bash

echo "Frame generation started"

# input/output folder
INPUT_DIR="bitstream/base"
OUTPUT_DIR="results"

# python script
PYTHON_SCRIPT="main.py"

# input resolution
WIDTH=1920
HEIGHT=1080

# output resolution
OUT_WIDTH=3840
OUT_HEIGHT=2160

# create output folder if not exists
mkdir -p "$OUTPUT_DIR"

# process all .yuv files
for input_file in "$INPUT_DIR"/*.yuv
do
    # avoid error when no .yuv file exists
    if [ ! -f "$input_file" ]; then
        echo "No .yuv files found in $INPUT_DIR"
        exit 1
    fi

    filename=$(basename "$input_file" .yuv)

    output_file="$OUTPUT_DIR/${filename}_4k_10bit.yuv"

    echo "----------------------------------------"
    echo "Input : $input_file"
    echo "Output: $output_file"
    echo "----------------------------------------"

    python "$PYTHON_SCRIPT" \
        --input "$input_file" \
        --output "$output_file" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --out_width "$OUT_WIDTH" \
        --out_height "$OUT_HEIGHT"

    echo "Finished processing $input_file"
done

echo "All YUV files finished."