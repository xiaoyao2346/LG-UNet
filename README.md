# LG-UNet: Nested U-Net with Discrepancy-Aware Learning for Enhanced Camouflaged Object Detection

This repository provides the official PyTorch implementation of:

**Yao Xiao, Haotian Wu, Kun Zhu, Dexin Zhao\***  
**"Nested U-Net with Discrepancy-Aware Learning for Enhanced Camouflaged Object Detection"**  
*The Visual Computer*

> 🔗 **Permanent Project Link:** https://github.com/xiaoyao2346/LG-UNet  
> 🔗 **Contact:** xiaoyao@stud.tjut.edu.cn

---

## 🔥 Overview

LG-UNet is a **new discrepancy-aware, nested U-shaped segmentation network** designed to detect camouflaged objects by learning subtle differences between objects and highly similar backgrounds.

Unlike standard U-Net variants, LG-UNet introduces:
- **A Global U-Net (GU)** for hierarchical semantic extraction  
- **Two Local U-Nets**, each processing local discrepancies  
  - **LTDU:** Local Texture Difference-aware U-Net  
  - **LSCU:** Local Spatial Consistency-aware U-Net  
- **DEB:** Discrepancy Enhanced Block  
- **Nested U-shaped architecture** applied both globally and locally  
- **Encoder:** PVTv2-B4  
- **Decoder:** Convolution-based U-Net decoder  

Extensive experiments on **COD10K, CAMO, CHAMELEON, NC4K** and **five polyp datasets** demonstrate that LG-UNet achieves state-of-the-art detection & segmentation performance.

---

---

## 🚀 Installation

### 1. Create environment
```
bash
conda create -n lgunet python=3.8
conda activate lgunet
```


### 2.Required Libraries
```
Python ≥ 3.8
PyTorch ≥ 1.12
torchvision ≥ 0.13
timm
numpy, opencv-python
tqdm, pyyaml
```

## 📂 Dataset Preparation
Camouflaged Object Detection Datasets

Download and place datasets as:

```
datasets/
│── CAMO/
│── CHAMELEON/
│── COD10K/
│── NC4K/
```

Polyp Segmentation Datasets
```
datasets/
│── Kvasir-SEG/
│── CVC-ClinicDB/
│── CVC-ColonDB/
│── CVC-300/
│── ETIS/
```

## 🏋️ Training
```
python LG-UNetTrain_Val.py
```



## 🔍 Testing
```
python LG-UNetTesting.py
```

## 🔍 Evaluation
```
python MyEval.py
```

## 🧠 Key Modules Explanation
### 1. LTDU — Local Texture Difference-aware U-Net

Captures fine-grained texture discrepancies by:

Multi-scale downsampling

TEM-based texture enhancement

U-shaped fusion

Outputs texture-aware feature maps

### 2. LSCU — Local Spatial Consistency-aware U-Net

Captures misalignment of spatial cues using:

Adjacent-layer bidirectional cross-attention

Depth-wise convolutions for local patterns

Multi-stage cross-attention fusion

Upsampling to original scale


## 📗 Citation

Please cite our work if you use the code or results:

@article{Xiao2025LGUNet,
  title   = {Nested U-Net with Discrepancy-Aware Learning for Enhanced Camouflaged Object Detection},
  author  = {Yao Xiao and Haotian Wu and Kun Zhu and Dexin Zhao},
  journal = {The Visual Computer},
  year    = {2025}
}

## 📬 Contact

If you have any other questions, feel free to contact me at xiaoyao@stud.tjut.edu.cn
