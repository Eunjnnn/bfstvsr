#!/bin/bash

ROOT_DIR="save_gopro"
TARGET_DIR="gt"
DIS_DIRS=("bfstvsr-w-flow" "bfstvsr" "motif" "videoinr")
FPS=30

for dist_dir in "${DIS_DIRS[@]}"; do
    for dir in "$ROOT_DIR/$TARGET_DIR"/*/; do
        REF_VIDEO_DIR="$dir"
        DIS_VIDEO_DIR="$ROOT_DIR/$dist_dir/$(basename $dir)"

        REF_VIDEO=$(find "$REF_VIDEO_DIR" -maxdepth 1 -type f -name "*.mp4")
        DIS_VIDEO=$(find "$DIS_VIDEO_DIR" -maxdepth 1 -type f -name "*.mp4")

        echo $REF_VIDEO $DIS_VIDEO
        docker run --gpus all --rm -e NVIDIA_DRIVER_CAPABILITIES=compute,video -v $REF_VIDEO:/data/ref.mp4 -v $DIS_VIDEO:/data/dis.mp4 -v $(pwd):/vmaf vmaf_cuda

        mkdir -p "logs_gopro_fps$FPS/$dist_dir"

        yes | mv "vmaf_output.json" "logs_gopro_fps$FPS/$dist_dir/$(basename $dir).json"
    done
done

# docker run --gpus all --rm -e NVIDIA_DRIVER_CAPABILITIES=compute,video -it --entrypoint bash vmaf_cuda:latest