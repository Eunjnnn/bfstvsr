## Usage

### VFIPS, FloLPIPS, tOF
We evaluate VFIPS and FloLPIPS following [VFIBenchmark](https://github.com/mulns/VFIBenchmark.git) respository, and evaluate tOF following [XVFI](https://github.com/JihyongOh/XVFI/blob/1352a13d0fe01226a6d893b6b94181606c8e14f4/utils.py#L603).

To evaluate the video quality metrics, you need to sample the interpolated frames. In the command below, dstDir is the folder containing the frames sampled for the model to be measured, and gtDir is the folder containing the ground-truth frames.

```
# Evalute BF-STVSR using VFIPS on Gopro dataset
python evaluate.py --method bfstvsr --save save_gopro --eval vfips --dataset gopro --dstDir /sampled_frames/bfstvsr_gopro --gtDir path/where/gt/gopro


# Evalute BF-STVSR using FloLPIPS on Gopro dataset
python evaluate.py --method bfstvsr --save save_gopro --eval flolpips --dataset gopro --dstDir /sampled_frames/bfstvsr_gopro --gtDir path/where/gt/gopro

# Evalute BF-STVSR using tOF on Gopro dataset
python evaluate.py --method bfstvsr --save save_gopro --eval tOF --dataset gopro --dstDir /sampled_frames/bfstvsr_gopro --gtDir path/where/gt/gopro

```

### VMAF
We use the videos (.mp4) that created while evaluating above metrics. 
We set fps as 30, and the total number of frames of the target video and predicted video should be same. 

```
cd vmaf

docker build -f Dockerfile.cuda -t vmaf_cuda .

REF_VIDEO=/path/to/target.mp4
DIS_VIDEO=/path/to/predicted.mp4

docker run --gpus all --rm -e NVIDIA_DRIVER_CAPABILITIES=compute,video -v $REF_VIDEO:/data/ref.mp4 -v $DIS_VIDEO:/data/dis.mp4 -v $(pwd):/vmaf vmaf_cuda
```