## Usage
We evaluate VFIPS and FloLPIPS following [VFIBenchmark](https://github.com/mulns/VFIBenchmark.git) respository, and evaluate tOF following [XVFI](https://github.com/JihyongOh/XVFI/blob/1352a13d0fe01226a6d893b6b94181606c8e14f4/utils.py#L603).

```
# Evalute BF-STVSR using VFIPS on Gopro dataset
python evaluate.py --method bfstvsr --save save_gopro --eval vfips --dataset gopro --dstDir /sampled_frames/bfstvsr_gopro --gtDir path/where/gt/gopro


# Evalute BF-STVSR using FloLPIPS on Gopro dataset
python evaluate.py --method bfstvsr --save save_gopro --eval flolpips --dataset gopro --dstDir /sampled_frames/bfstvsr_gopro --gtDir path/where/gt/gopro

# Evalute BF-STVSR using tOF on Gopro dataset
python evaluate.py --method bfstvsr --save save_gopro --eval tOF --dataset gopro --dstDir /sampled_frames/bfstvsr_gopro --gtDir path/where/gt/gopro

```