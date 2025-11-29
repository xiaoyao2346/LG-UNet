# LG-UNet: Nested U-Net with Discrepancy-Aware Learning for Enhanced Camouflaged Object Detection

This repository provides the official PyTorch implementation of:

**Yao Xiao, Haotian Wu, Kun Zhu, Dexin Zhao\***  
**"Nested U-Net with Discrepancy-Aware Learning for Enhanced Camouflaged Object Detection"**  
*The Visual Computer (Under Review, Revised Version Submitted)*

> 🔗 **Permanent Project Link:** https://github.com/xiaoyao2346/LG-UNet  
> 🔗 **Paper (PDF):** Provided with journal submission  
> 🔗 **Contact:** zhaodexin@email.tjut.edu.cn

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
```bash
conda create -n lgunet python=3.8
conda activate lgunet

### 2.Required Libraries
Python ≥ 3.8
PyTorch ≥ 1.12
torchvision ≥ 0.13
timm
numpy, opencv-python
tqdm, pyyaml

