# 4K YUV Frame Reconstruction Using RAFT-based Warping and Super-Resolution

## Overview
<p align="center">
    <img src="docs/figure.png" width="500">
</p>
This project aims to reconstruct high-resolution 4K YUV video frames from low-resolution FHD base frames and neighboring 4K reference frames. Instead of directly upscaling the FHD frame, the proposed pipeline uses temporal information from adjacent 4K frames to recover high-frequency details.

The system integrates RAFT-based optical flow estimation, backward warping, valid-mask filtering, adaptive fusion, and optional Y-channel super-resolution to improve the visual quality of generated 4K frames.

[View Report](docs/CV_2025_Final.pdf)


## System Pipeline

```mermaid
flowchart LR
    subgraph Inputs
        A1[FHD Base Layer<br/>YUV420 10-bit]
        A2[Previous 4K Frame]
        A3[Next 4K Frame]
    end

    subgraph Base_Layer_Reconstruction
        B1[Parse YUV420]
        B2[Separate Y / U / V]
        B3[Y-only Super Resolution<br/>or Resize Baseline]
        B4[Upscale U / V]
    end

    subgraph Temporal_Alignment
        C1[RAFT Optical Flow<br/>Previous → Target]
        C2[RAFT Optical Flow<br/>Next → Target]
        C3[Warp Previous Frame]
        C4[Warp Next Frame]
    end

    subgraph Fusion
        D1[Base 4K Estimate]
        D2[Warped Previous]
        D3[Warped Next]
        D4[Weighted Fusion / Adaptive Fusion]
    end

    subgraph Output
        E1[Reconstructed 4K YUV]
    end

    A1 --> B1 --> B2
    B2 --> B3 --> D1
    B2 --> B4

    A2 --> C1 --> C3 --> D2
    A3 --> C2 --> C4 --> D3

    D1 --> D4
    D2 --> D4
    D3 --> D4

    D4 --> E1
    B4 --> E1
```

## Installation 

### 1. Clone the repository

```bash
git clone https://github.com/jf099005/CV_final_2026_MediaTek_group9.git
cd CV_final_2026_MediaTek_group9
```

### 2. Create a virtual environment

```bash
python -3.10 -m venv .venv
```

### 3. Activate the virtual environment
```bash 
.venv\Scripts\activate
```

### 4. Install dependencies
```bash 
pip install -r requirements.txt
```

## Usage
### 1. Download example data
The example bitstream is hosted externally due to its large file size.

- [Download Example Bitstream](https://drive.google.com/file/d/1a--1J5K1HZ9qw1h1OOUiB6ZbXRYik0zY/view?usp=sharing)

After downloading, extract `bitstream.zip` and place the `bitstream/` directory under the project root. The expected structure is:

```text
CV_final_2026_MediaTek_group9/
└── bitstream/
    ├── base/
    ├── enhance/
    └── upscaled/
```
### 2. Upsample the bitstream
```bash
bash ./FrameGen.sh
```
### 3. Generate video 
```bash
bash ./gen_video.sh
```

## Evaluation
You can evaluate the BD-rate of the results from example data using: 
```bash 
cd scriptTest
python testOddFramesAMS05.py
```
