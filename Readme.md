Phase 1:https://drive.google.com/file/d/1r4O2wAlrC9Y7njH8JJUT78X8f-2FSZeF/view?usp=sharing
Phase 2:https://drive.google.com/file/d/1mHsY_FwquNufExnvFbwwPYaTd3WwuT-b/view?t=7.017
---

# Vulnerability Analysis of Semi-Cycled GANs in Biometric Pipelines

This repository contains the official implementation and experimental framework for a security audit of **Semi-Cycled Generative Adversarial Networks (SCGAN)**. Our research exposes a critical vulnerability in the generative latent space that allows for deterministic identity spoofing in biometric authentication systems.

**Authors:** Devesh Mirchandani, Kartik Sharma, Padmnabh Tewari (IIIT Vadodara)  



---

## 🏗️ System Architecture

The project centers on the **SCGAN** framework, which utilizes a dual-branch cycle to bridge the domain gap between real-world and synthetic image degradations.

### 1. SCGAN Restoration Branch ($\mathcal{G}_{RLS}$)
The primary focus of this audit. This branch is responsible for upscaling $16 \times 16$ low-resolution (LR) inputs to high-resolution (HR) outputs. In a biometric pipeline, this acts as the "trusted gateway" for facial recognition models like FaceNet.

### 2. The Attack: White-Box Pipeline Exploit
We utilize **Projected Gradient Descent (PGD)** to hijack the generative process.
* **Input:** $16 \times 16$ image + $L_\infty$-bounded adversarial noise ($\delta$).
* **Mechanism:** Backpropagating the identity error from a downstream discriminator through the frozen SCGAN layers.
* **Result:** The GAN hallucinations are forced to match a **Target Identity**, bypassing automated biometric judges.

### 3. The Mitigation: Zero-Shot Purification
We evaluate a spatial purification gateway (3x3 Median Filtering) to neutralize the adversarial noise. Our audit quantifies a **Fatal Utility-Robustness Trade-off**, where the defense destroys the GAN's ability to process even benign, unpoisoned images.

---

## 📂 Pretrained Weights

Due to GitHub's file size restrictions, the model weights (`.pth`) are not hosted in this repository. 

**To run this code:**
1. Refer to the **[Original SCGAN Paper Link](https://github.com/HaoHou-98/SCGAN)** (or the specific base paper you are using).
2. Download the baseline `G_l2h` weights.
3. Place the weights file in the `./pretrained_weights/` directory.
4. Ensure the file is named `pretrained_model.pth` to match the configuration in `attack_pipeline.py`.

---

## 📊 Key Results (N=1000)

| Stage | Metric | Value |
| :--- | :--- | :---: |
| **Stealth** | Input SSIM (Orig vs Poison) | **0.9440** |
| **Attack** | Output SSIM (Base vs Spoof) | **0.8227** |
| **Defense** | Defended Output SSIM | **0.5664** |
| **Utility** | Baseline Utility Penalty | **-41.5%** |

