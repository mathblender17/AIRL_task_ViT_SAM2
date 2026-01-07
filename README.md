👁️ Advanced Computer Vision: From Scratch ViT to Zero-Shot SAM 2
=================================================================

This repository demonstrates the spectrum of modern Computer Vision engineering: **Deep Learning Fundamentals** and **Applied Foundation Models**. It contains two distinct projects:

1.  **Vision Transformer (ViT) Implementation**: A custom, ground-up implementation of ViT trained on CIFAR-10 with state-of-the-art regularization techniques.

2.  **Text-Driven Segmentation (VLM)**: A zero-shot pipeline orchestrating **GroundingDINO** and **SAM 2** to segment objects based purely on natural language prompts.

* * * * *

📂 Project Structure
--------------------

| **Notebook** | **Task Description** | **Key Technologies** |
| --- | --- | --- |
| **`q1.ipynb`** | **ViT from Scratch**: Training a Vision Transformer on CIFAR-10. | PyTorch, Mixup, AutoAugment, AMP |
| **`q2.ipynb`** | **Text-to-Mask Pipeline**: Zero-shot segmentation pipeline. | SAM 2, GroundingDINO, Transformers |

* * * * *

🚀 Task 1: Vision Transformer (ViT) on CIFAR-10
-----------------------------------------------

A complete PyTorch implementation of the Vision Transformer architecture, optimized for fast convergence and generalization on small datasets.

### 🧠 Technical Highlights

-   **Architecture from Scratch**: Manual implementation of `PatchEmbed`, `CLS` tokens, and `Multi-Head Self-Attention` (MHSA) blocks.

-   **Modern Regularization Recipe**: Implements the "Bag of Tricks" for training Transformers effectively on small data:

    -   **Mixup & Label Smoothing**: Prevents overfitting by blending images and softening targets.

    -   **AutoAugment**: Learns optimal augmentation policies.

    -   **Stochastic Depth (DropPath)**: Randomly drops residual paths during training.

-   **Performance Optimization**: Uses **Automatic Mixed Precision (AMP)** for accelerated GPU training.

### 📊 Model Config & Training

| **Parameter** | **Value** | **Description** |
| --- | --- | --- |
| **Patch Size** | 4x4 | Optimized for 32x32 CIFAR images |
| **Embed Dim** | 512 | Feature vector size per token |
| **Depth** | 12 | Number of Transformer blocks |
| **Heads** | 8 | Parallel attention heads |
| **Optimizer** | AdamW | `lr=3e-4`, Cosine Decay Scheduler |

* * * * *

🔮 Task 2: Text-Driven Image Segmentation (SAM 2)
-------------------------------------------------

An agentic pipeline that bridges **Language** and **Vision**. Instead of manual prompting, this system allows users to segment objects using natural language (e.g., *"red bicycle"*), leveraging the power of Vision-Language Models.

### 🛠️ The Pipeline

1.  **Prompting**: User inputs a text prompt (e.g., *"orange cat"*).

2.  **Zero-Shot Detection (GroundingDINO)**: The VLM scans the image and generates **bounding box seeds** for the text concept.

3.  **Segmentation (SAM 2)**: These boxes are fed into **Segment Anything Model 2**, which generates pixel-perfect masks.

4.  **Result**: High-quality segmentation without manual clicking or annotations.

### 🌟 Key Capabilities

-   **Zero-Shot Generalization**: Works on any object category without retraining.

-   **Foundation Model Orchestration**: Chains multiple SOTA models (DINO + SAM 2) to solve complex tasks.

-   **Video Ready**: Leveraging SAM 2's architecture, this approach is extensible to video object segmentation.

* * * * *

💻 Installation & Usage
-----------------------

### 1\. Clone the Repository

Bash

```
git clone https://github.com/mathblender17/airl_task_vit_sam2.git
cd airl_task_vit_sam2

```

### 2\. Install Dependencies

Bash

```
# Core DL libraries
pip install torch torchvision timm

# For SAM 2 and GroundingDINO
pip install transformers opencv-python matplotlib

```

### 3\. Run in Colab

Both tasks are designed to be fully runnable in Google Colab (Free Tier compatible).

-   Open `q1.ipynb` for ViT Training.

-   Open `q2.ipynb` for Text-to-Segmentation.

-   *Ensure Runtime is set to **GPU**.*

* * * * *

📜 References
-------------

-   **ViT**: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

-   **SAM 2**: [Segment Anything Model 2](https://github.com/facebookresearch/segment-anything-2)

-   **GroundingDINO**: [Grounding DINO: Marrying DINO with Grounded Pre-Training](https://github.com/IDEA-Research/GroundingDINO)
