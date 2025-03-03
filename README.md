# BF-STVSR: B-Splines and Fourier-Best Friends for High Fidelity Spatial-Temporal Video Super-Resolution

Authors : Eunjin Kim*, Hyeonjin Kim*, Kyong Hwan Jin, [Jaejun Yoo](https://scholar.google.co.kr/citations?hl=en&user=7NBlQw4AAAAJ)

(* : equal contribution)

[Paper](https://arxiv.org/abs/2501.11043a)  | [Project Page](https://eunjnnn.github.io/bfstvsr_site/) 
### [News]
* Our **BF-STVSR** is accepted by **CVPR 2025** 🎉!

## Abstract
> Enhancing low-resolution, low-frame-rate videos to high-resolution, high-frame-rate quality is essential for a seamless user experience, motivating advancements in Continuous Spatial-Temporal Video Super Resolution (C-STVSR). While prior methods employ Implicit Neural Representation (INR) for continuous encoding, they often struggle to capture the complexity of video data, relying on simple coordinate concatenation and pre-trained optical flow network for motion representation. Interestingly, we find that adding position encoding, contrary to common observations, does not improve-and even degrade performance. This issue becomes particularly pronounced when combined with pre-trained optical flow networks, which can limit the model's flexibility. To address these issues, we propose BF-STVSR, a C-STVSR framework with two key modules tailored to better represent spatial and temporal characteristics of video: 1) B-spline Mapper for smooth temporal interpolation, and 2) Fourier Mapper for capturing dominant spatial frequencies. Our approach achieves state-of-the-art PSNR and SSIM performance, showing enhanced spatial details and natural temporal consistency.

## Overview of BF-STVSR

<img src = "./asset/overview.png" width="100%" height="100%"/>

## Environmental Setup

We follow the environment setting from [MoTIF](https://github.com/sichun233746/MoTIF) and [VideoINR](https://github.com/Picsart-AI-Research/VideoINR-Continuous-Space-Time-Super-Resolution).


## Citation
If you find this repository useful for your research, please cite the following work.
```
@article{kim2025bf,
  title={BF-STVSR: B-Splines and Fourier-Best Friends for High Fidelity Spatial-Temporal Video Super-Resolution},
  author={Kim, Eunjin and Kim, Hyeonjin and Jin, Kyong Hwan and Yoo, Jaejun},
  journal={arXiv preprint arXiv:2501.11043},
  year={2025}
}
```