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
