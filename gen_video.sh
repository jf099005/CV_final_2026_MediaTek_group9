#!/bin/bash

EVEN_DIR="bitstream/enhance"
ODD_DIR="results"
OUTPUT_DIR="mp4"
WIDTH=3840
HEIGHT=2160

mkdir -p "$OUTPUT_DIR"

for even_file in "$EVEN_DIR"/even_*.layer1.yuv; do
    [ -f "$even_file" ] || continue

    base=$(basename "$even_file")
    key="${base#even_}"
    key="${key%.layer1.yuv}"

    odd_file="$ODD_DIR/odd_${key}_generated_4k.layer1.yuv"
    if [ ! -f "$odd_file" ]; then
        echo "No matching odd file for key: $key (expected $odd_file)"
        continue
    fi

    if [[ "$key" == *"ZombieClimbing2"* ]]; then
        fps=24
        crf=28
    else
        fps=60
        crf=25
    fi

    output_file="$OUTPUT_DIR/${key}.mp4"

    echo "----------------------------------------"
    echo "Even : $even_file"
    echo "Odd  : $odd_file"
    echo "Output: $output_file"
    echo "FPS   : $fps  CRF: $crf"
    echo "----------------------------------------"

    python3 - "$even_file" "$odd_file" | ffmpeg -y \
        -f rawvideo -vcodec rawvideo \
        -s "${WIDTH}x${HEIGHT}" \
        -r "$fps" \
        -pix_fmt yuv420p10le \
        -i pipe:0 \
        -c:v libx265 \
        -preset slow \
        -crf "$crf" \
        -pix_fmt yuv420p10le \
        "$output_file" <<'PYEOF'
import sys

FRAME_SIZE = 3840 * 2160 * 3  # yuv420p10le: 1.5 * W * H * 2 bytes

with open(sys.argv[1], 'rb') as ef, open(sys.argv[2], 'rb') as of:
    while True:
        even_frame = ef.read(FRAME_SIZE)
        if not even_frame:
            break
        sys.stdout.buffer.write(even_frame)
        odd_frame = of.read(FRAME_SIZE)
        if not odd_frame:
            break
        sys.stdout.buffer.write(odd_frame)
PYEOF

    if [ $? -eq 0 ]; then
        echo "Done: $output_file"
    else
        echo "Error: $key"
    fi
done

echo "All conversions finished. MP4 files are in $OUTPUT_DIR"
