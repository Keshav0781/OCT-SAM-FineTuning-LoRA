## Motivation

Segment Anything Model (SAM) is a powerful vision foundation model, but **zero-shot segmentation** often fails on domain-specific medical images like OCT scans.  
To address this, we applied **LoRA-based fine-tuning** of SAM for **biomarker segmentation** in OCT scans of **Age-related Macular Degeneration (AMD)** and **Macular Hole**.  

This work demonstrates:
- Why SAM alone is insufficient in medical imaging without adaptation.  
- How lightweight **LoRA adapters** enable fine-tuning with limited data.  
- Significant performance gains (Dice / IoU) compared to SAM zero-shot.

## Dataset & Preprocessing

We used the same OCT datasets as in the [OCT-Biomarker-Segmentation](https://github.com/Keshav0781/OCT-Biomarker-Segmentation) project:  
- **Age-related Macular Degeneration (AMD)**  
- **Macular Hole**  

Each dataset contains manually annotated B-scans with pixel-wise masks for clinically relevant biomarkers (e.g., drusen, intra-/sub-retinal fluid, tissue defects).  

**Preprocessing for SAM**  
- OCT B-scans were resized to **1024×1024 RGB** to match SAM’s input requirements.  
- Standard **data augmentation** (random flips, rotations, brightness/contrast) was applied during training.  

> For visual dataset distributions and annotation examples, see the [previous repository](https://github.com/Keshav0781/OCT-Biomarker-Segmentation).

## Methodology — SAM + LoRA Fine-tuning

The **Segment Anything Model (SAM)** provides strong image segmentation priors but struggles in the OCT domain.  
To adapt it, we used **Low-Rank Adaptation (LoRA)**, a lightweight fine-tuning technique that injects trainable rank-decomposition matrices into specific attention layers.

<div align="center">
  <img src="figures/methodology/sam_architecture.png" alt="SAM architecture" width="600"/><br>
  <b>Figure: High-level architecture of SAM (image encoder + prompt encoder + mask decoder)</b>
</div>

### Why LoRA?
- SAM has **hundreds of millions of parameters**, making full fine-tuning impractical.  
- LoRA fine-tunes only a small number of additional parameters, keeping training efficient and memory-friendly.  
- This enables effective adaptation to OCT biomarker segmentation with limited annotated data.

### Training setup
- **Backbone**: Vision Transformer encoder from SAM.  
- **LoRA rank**: 16 (applied to attention projections).  
- **Optimizer**: AdamW with learning rate = 1e-4.  
- **Batch size**: 16.  
- **Epochs**: up to 50 with early stopping (patience = 5).  
- **Loss**: Dice loss + Cross-entropy combination.  
- **Datasets**: AMD and Macular Hole OCT B-scans, resized to 1024×1024.  

> The training was performed separately for AMD and Macular Hole datasets, producing disease-specific SAM + LoRA checkpoints.
