Vision Transformer (ViT) on CIFAR-10
====================================

This repository contains a Vision Transformer (ViT) implementation in PyTorch trained on the CIFAR-10 dataset. The project includes optimizations like Mixup, Label Smoothing, AutoAugment, DropPath, Cosine LR Scheduler, and AMP (Automatic Mixed Precision) for faster GPU training.

* * *

Features:

*   Patch-based image embedding with CLS token and learnable positional encoding
    
*   Multi-head Self-Attention (MHSA) transformer blocks with residual connections
    
*   MLP heads for classification
    
*   Mixup data augmentation
    
*   Label smoothing for improved generalization
    
*   AutoAugment for robust image transformations
    
*   DropPath (stochastic depth) for regularization
    
*   CosineAnnealing LR scheduler
    
*   AMP for faster training on GPU
    
*   Fully runnable in Google Colab
    

* * *

Installation:

`# Install dependencies
pip install torch torchvision timm` 

* * *

Dataset:

*   Uses CIFAR-10, which contains 60,000 32x32 color images in 10 classes (50,000 training, 10,000 test).
    
*   Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    

* * *

Model Architecture:

*   Patch size: 4x4
    
*   Embedding dimension: 512
    
*   Depth: 12 transformer blocks
    
*   Heads: 8 per MHSA
    
*   MLP ratio: 4
    
*   DropPath: Linear decay from 0 → 0.1
    
*   CLS token + positional encoding for global representation
    

* * *

Training:

*   Optimizer: AdamW
    
*   Learning Rate: 3e-4
    
*   Scheduler: CosineAnnealingLR
    
*   Batch size: 128
    
*   Epochs: 100
    
*   Loss: Label smoothing + Mixup
    

Example training loop:

`train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, mixup_alpha=0.8)
val_loss, val_acc = evaluate(model, test_loader, criterion)` 

*   Mixed Precision automatically enabled on GPU.
    

* * *

Results:



* * *

How to Run:

1.  Clone the repo:
    

`git clone 
cd ViT-CIFAR10` 

2.  Launch `colab_notebook.ipynb` in Google Colab.
    
3.  Enable GPU runtime (Colab → Runtime → Change runtime type → GPU).
    
4.  Run all cells to train the model end-to-end.
    

* * *

Limitations:

*   Training from scratch is slower; pretraining on ImageNet can further improve accuracy.
    
*   Small CIFAR-10 images (32x32) limit the benefit of very deep ViT models.
    
*   No early stopping implemented yet.
    

* * *

References:

1.  An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT) - https://arxiv.org/abs/2010.11929
    
2.  timm PyTorch library - [https://github.com/rwightman/pytorch-image-models]

* * *
Text-Driven Image Segmentation with SAM 2
=========================================

This repository demonstrates **text-prompted image segmentation** using **Segment Anything Model (SAM 2)**. Given an image and a text prompt (e.g., "red bicycle" or "orange cat"), the pipeline detects relevant regions and produces segmentation masks.

* * *

📌 Features
-----------

*   Accepts **any input image** and a **text prompt** describing the object to segment
    
*   Converts the text prompt to **region seeds** using models like **GroundingDINO / GLIP / CLIPSeg**
    
*   Feeds region seeds to **SAM 2** for mask generation
    
*   Displays **final mask overlay** on the original image
    
*   End-to-end runnable on **Google Colab**
    

* * *

🛠 Installation
---------------

`# Install dependencies
pip install torch torchvision timm transformers opencv-python matplotlib
# Additional SAM 2 / grounding models may require cloning official repos or installing from HuggingFace` 

* * *

📂 Example Images
-----------------

*   Any natural image where the object described in the text prompt is visible.
    
*   Recommended resolution: **moderate size (<= 1024x1024)** for Colab GPU
    
*   Example prompts: `"orange cat"`, `"man with hat"`, `"white dog"`, `"red bicycle"`
    

* * *

🏗 Pipeline Overview
--------------------

1.  **Load Image** – Read the input image.
    
2.  **Text Prompt → Region Seeds** – Use GroundingDINO / GLIP / CLIPSeg to generate bounding boxes or points for the object described in text.
    
3.  **SAM 2 Segmentation** – Feed the seeds to SAM 2 to produce masks.
    
4.  **Display Masks** – Overlay the mask on the original image.
    

* * *

⚡ How to Run
------------

1.  Clone the repo:
    

`git clone 
cd SAM2-TextSeg` 

2.  Launch `sam2_colab.ipynb` in Google Colab.
    
3.  Enable GPU runtime (Colab → Runtime → Change runtime type → GPU).
    
4.  Run all cells and provide an image + text prompt.
    

* * *

🎯 Limitations
--------------

*   Accuracy depends on the text-to-region model; sometimes objects may not be detected if the prompt is ambiguous.
    
*   Very small or partially occluded objects may fail.
    
*   Colab GPU memory limits the maximum image size.
    
*   SAM 2 is large; inference may take a few seconds per image.
    

* * *

📖 References
-------------

1.  [Segment Anything Model 2 (SAM 2)]
    
2.  [GroundingDINO]
    
3.  [CLIPSeg]
