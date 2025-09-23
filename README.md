## Motivation

Segment Anything Model (SAM) is a powerful vision foundation model, but **zero-shot segmentation** often fails on domain-specific medical images like OCT scans.  
To address this, we applied **LoRA-based fine-tuning** of SAM for **biomarker segmentation** in OCT scans of **Age-related Macular Degeneration (AMD)** and **Macular Hole**.  

This work demonstrates:
- Why SAM alone is insufficient in medical imaging without adaptation.  
- How lightweight **LoRA adapters** enable fine-tuning with limited data.  
- Significant performance gains (Dice / IoU) compared to SAM zero-shot.
