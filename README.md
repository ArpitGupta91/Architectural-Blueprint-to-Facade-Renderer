# Architectural Blueprint to Facade Renderer

### Paired Semantic Layout-to-Image Translation Using a Conditional Generative Adversarial Network (cGAN)

**CA-2 Project · Generative AI · PyTorch**

Transforming semantic facade layouts into plausible facade images using a **Pix2Pix-style Conditional Generative Adversarial Network**.

---

## Table of Contents

- [Overview](#overview)
- [Project at a Glance](#project-at-a-glance)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Model and Training Configuration](#model-and-training-configuration)
- [Evaluation](#evaluation)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Running the Notebook](#running-the-notebook)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Team](#team)
- [References](#references)

---

## Overview

**Architectural Blueprint to Facade Renderer** is an academic project that explores **paired image-to-image translation** using a **Pix2Pix-style Conditional Generative Adversarial Network (cGAN)**.

The system takes a **semantic annotation of a building facade** as input and learns to generate a corresponding facade photograph.

The implementation is developed using **Python and PyTorch** and consists of two main neural networks:

- **U-Net Generator** — converts the semantic facade layout into a generated facade image.
- **PatchGAN Discriminator** — evaluates local regions of the conditional input-output pair and encourages realistic generated details.

The generator is trained using a combination of:

- Adversarial loss
- L1 reconstruction loss

A validation-based checkpoint selection approach is used to select the generator for final evaluation.

> **Important scope clarification:** Although the project title uses the word *blueprint*, the implemented experiment uses **semantic facade annotations paired with facade photographs**. It does not currently perform rendering from CAD drawings, floor plans, or construction-ready architectural blueprints.

---

## Project at a Glance

| Item | Details |
|---|---|
| Project | Architectural Blueprint to Facade Renderer |
| Task | Paired semantic layout-to-facade image translation |
| Framework | PyTorch |
| Generative Approach | Pix2Pix-style Conditional GAN |
| Generator | U-Net |
| Discriminator | PatchGAN |
| Dataset | CMP Facade Database and available extended data |
| Input | Semantic facade annotation |
| Output | Generated facade image |
| Resolution | 256 × 256 RGB |
| Training Objective | Adversarial Loss + Weighted L1 Loss |
| L1 Weight | 100 |
| Optimizer | Adam |
| Learning Rate | 0.0002 |
| Adam beta1 | 0.5 |
| Configured Epochs | 30 |
| Evaluation | MAE and SSIM |
| Project Type | Academic CA-2 Project |

---

## Problem Statement

A semantic facade layout can describe the arrangement of important building elements such as:

- Walls
- Windows
- Doors
- Other facade regions

However, a semantic layout does not contain the complete visual appearance of a real building facade, such as:

- Materials
- Colors
- Textures
- Lighting
- Surface appearance
- Visual details

The goal of this project is to investigate whether a **conditional generative model** can learn the relationship between a semantic facade annotation and its corresponding real facade photograph.

The model is trained using **paired examples**, where each semantic annotation is matched with the corresponding facade photograph.

---

## Objectives

The main objectives of the project are:

1. Prepare paired semantic facade annotations and facade photographs.
2. Build a conditional image-to-image generation pipeline using PyTorch.
3. Use a **U-Net Generator** to generate facade images.
4. Use a **PatchGAN Discriminator** to provide adversarial feedback.
5. Train the model using adversarial and L1 reconstruction objectives.
6. Evaluate the generated images using **MAE and SSIM**.
7. Inspect generated outputs qualitatively to understand the model's behavior.

---

## How It Works

The overall workflow can be summarized as:

```text
CMP Facade Dataset
        |
        v
Match Semantic Annotation
with Facade Photograph
        |
        v
Preprocessing
256 × 256 RGB
Normalization
        |
        v
Semantic Facade Layout
        |
        v
   U-Net Generator
        |
        v
Generated Facade Image
        |
        +----------------------+
        |                      |
        v                      v
 Compare with Target     PatchGAN
        |                Discriminator
        v                      |
   L1 Loss              Real / Fake Feedback
        |                      |
        +----------+-----------+
                   |
                   v
               Training
                   |
                   v
       Best Validation Checkpoint
                   |
                   v
          Test Evaluation
             MAE / SSIM
```

During training, the Generator attempts to create a realistic facade image from the semantic layout.

The Discriminator receives the semantic layout together with either:

- the real facade photograph, or
- the generated facade photograph.

It then learns to distinguish between real and generated pairs.

At inference time, only the **Generator** is required to produce a facade image from a semantic layout.

---

## System Architecture

The project follows the basic architecture of a **Pix2Pix conditional GAN**.

### 1. U-Net Generator

The Generator follows an encoder-decoder structure.

#### Encoder

The encoder progressively extracts higher-level features from the semantic input while reducing spatial resolution.

#### Bottleneck

The bottleneck represents deeper and more compact features extracted from the input.

#### Decoder

The decoder progressively reconstructs the output facade image.

#### Skip Connections

U-Net skip connections transfer spatial information from the encoder to corresponding decoder layers.

This is particularly useful for preserving the location and structure of facade elements such as windows and doors.

---

### 2. PatchGAN Discriminator

The Discriminator receives a conditional pair:

```text
Semantic Layout + Real Facade
```

or

```text
Semantic Layout + Generated Facade
```

Instead of judging the entire image with only one decision, PatchGAN evaluates local image regions.

This encourages the Generator to produce realistic local structures, edges, textures, and facade details while remaining consistent with the semantic input.

---

## Methodology

### Data Preparation

The data preparation process consists of:

1. Downloading the CMP Facade dataset and available extended data.
2. Locating facade photographs and corresponding annotation files.
3. Matching each annotation with its corresponding photograph.
4. Inspecting sample pairs to verify correct matching.
5. Resizing images to **256 × 256 pixels**.
6. Converting images to RGB tensors.
7. Normalizing image values to **[-1, 1]**.
8. Splitting the paired data into training, validation, and test subsets.

The configured dataset split is:

```text
80% Training
10% Validation
10% Testing
```

---

## Model Training

The Generator is trained using adversarial loss together with L1 reconstruction loss.

The overall Generator objective is:

```text
L_G = L_GAN + λ_L1 × L_L1
```

where:

```text
λ_L1 = 100
```

### Adversarial Loss

The adversarial component encourages the Generator to produce images that the Discriminator considers realistic.

### L1 Reconstruction Loss

The L1 component measures the pixel-level difference between:

```text
Generated Facade
        vs.
Ground-Truth Facade
```

This encourages the generated image to remain close to the paired target.

### Discriminator Objective

The Discriminator learns to distinguish between:

```text
Real:
Semantic Layout + Real Facade
```

and:

```text
Fake:
Semantic Layout + Generated Facade
```

The Generator and Discriminator are trained together so that the Generator gradually improves its ability to create realistic facade images.

---

## Dataset

This project uses the **CMP Facade Database** and available extended data.

The dataset provides facade photographs together with semantic annotation images that can be matched into input-target pairs.

### Input

Semantic facade annotation.

### Target

Corresponding real facade photograph.

### Image Configuration

```text
Resolution: 256 × 256
Channels: RGB
Normalization: [-1, 1]
```

### Dataset Split

```text
Training:   80%
Validation: 10%
Testing:    10%
```

### Dataset Source

**CMP Facade Database — Center for Machine Perception**

https://cmp.felk.cvut.cz/~tylecr1/facade/

Please follow the dataset provider's terms when downloading or redistributing dataset files. The complete dataset should not be committed to this repository unless redistribution is permitted.

---

## Model and Training Configuration

The configured experiment uses the following settings:

| Setting | Configuration |
|---|---|
| Framework | PyTorch |
| Generator | U-Net |
| Discriminator | PatchGAN |
| Image Resolution | 256 × 256 RGB |
| Normalization | [-1, 1] |
| Optimizer | Adam |
| Learning Rate | 0.0002 |
| Adam beta1 | 0.5 |
| Configured Epochs | 30 |
| Generator Loss | GAN Loss + 100 × L1 |
| Checkpoint Selection | Best Validation L1 Loss |
| Evaluation | MAE and SSIM |

> **Note:** 30 epochs represents the configured training setup. It does not necessarily mean that all 30 epochs have completed in every execution environment.

---

## Evaluation

The selected Generator is evaluated using both qualitative and quantitative analysis.

### Mean Absolute Error (MAE)

MAE measures the average absolute pixel-level difference between the generated image and the target image.

In general:

```text
Lower MAE → smaller pixel-level difference
```

### Structural Similarity (SSIM)

SSIM measures similarity between images based on structural information, luminance, and contrast.

In general:

```text
Higher SSIM → greater structural similarity
```

### Why Use Both?

MAE and SSIM provide complementary information.

MAE focuses more directly on pixel-level differences, while SSIM provides information about structural similarity.

However, these metrics alone cannot fully measure:

- Architectural realism
- Semantic correctness
- Human-perceived visual quality
- Architectural feasibility
- Construction suitability

---

## Results

The final experimental results should be reported only from the **executed notebook**.

The repository should contain the actual notebook outputs, including:

- Sample input semantic layouts
- Ground-truth facade images
- Generated facade images
- Training loss plots
- MAE result
- SSIM result

### Results Reporting Rule

> **Only report values obtained from the actual executed notebook. Do not use fabricated, estimated, or illustrative metric values.**

Once the final experiment has been executed, the actual MAE and SSIM values can be added here.

Example format:

| Metric | Test Result |
|---|---:|
| MAE | Add actual notebook result |
| SSIM | Add actual notebook result |

---

## Repository Structure

The repository currently contains the following main project files:

```text
Architectural-Blueprint-to-Facade-Renderer/
│
├── notebook/
│   └── Architectural_Blueprint_to_Facade_Renderer.ipynb
│
├── paper/
│   └── research_paper.pdf
│
├── presentation/
│   └── Facade_Renderer_Presentation.pptx
│
└── README.md
```

> **Note:** The notebook and presentation filenames shown above should match the actual filenames in the repository.

The repository is intentionally kept simple and contains the main materials required for the CA-2 project:

- Executed Jupyter Notebook
- Research Paper
- Project Presentation
- README documentation

---

## Getting Started

### Prerequisites

The project requires:

- Python
- PyTorch
- Jupyter Notebook or Google Colab
- Required Python libraries used by the notebook
- Internet access for downloading the dataset

A compatible GPU can significantly improve GAN training time, although the actual training performance depends on the available hardware.

---

## Running the Notebook

Open the project notebook located inside the `notebook/` directory.

Run the notebook cells in order:

```text
1. Import dependencies
        ↓
2. Prepare / download dataset
        ↓
3. Match annotation-image pairs
        ↓
4. Inspect paired examples
        ↓
5. Preprocess images
        ↓
6. Create train/validation/test splits
        ↓
7. Define U-Net Generator
        ↓
8. Define PatchGAN Discriminator
        ↓
9. Configure training
        ↓
10. Train the GAN
        ↓
11. Select best validation checkpoint
        ↓
12. Generate test predictions
        ↓
13. Calculate MAE and SSIM
        ↓
14. Inspect generated results
```

Training time depends on the available CPU/GPU hardware and the execution environment.

---

## Limitations

The current project has several limitations:

- The experiment uses **semantic facade annotations**, not complete CAD blueprints or architectural floor plans.
- Generated image quality depends on the quantity and quality of paired training data.
- Correct annotation-photograph pairing is important for supervised training.
- GAN training can be sensitive to hyperparameters.
- Generated images may contain blurry, repetitive, or inconsistent visual details.
- MAE and SSIM alone cannot fully measure architectural quality or human-perceived realism.
- The generated outputs are **conceptual visualizations**, not construction-ready architectural designs.
- Results should only be reported from actual, reproducible notebook executions.

---

## Future Improvements

Possible future improvements include:

1. Expand the variety and quantity of paired facade examples.
2. Experiment with additional training configurations and checkpoint strategies.
3. Compare the Pix2Pix-style approach with suitable image-to-image translation baselines.
4. Include perceptual metrics in addition to MAE and SSIM.
5. Include human evaluation of generated facade quality.
6. Investigate larger and more diverse facade datasets.
7. Explore CAD-to-rendering datasets for a genuine blueprint-to-facade generation system.
8. Incorporate architectural and geometric constraints for more reliable generation.

These are proposed directions for future work and are **not claims about features currently implemented**.

---

## Team

**Arpit Gupta**  
**Aryan Tiwari**

Department of Computer Science  
KIET Deemed to be University  
Ghaziabad, Uttar Pradesh, India

---

## References

1. I. Goodfellow et al., “Generative Adversarial Nets,” *Advances in Neural Information Processing Systems*, vol. 27, 2014.

2. P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-Image Translation with Conditional Adversarial Networks,” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 1125–1134.

   https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html

3. Center for Machine Perception, Czech Technical University in Prague, “CMP Facade Database.”

   https://cmp.felk.cvut.cz/~tylecr1/facade/

4. PyTorch Contributors, “PyTorch Documentation.”

   https://pytorch.org/docs/stable/

---

## Academic Project Note

This project demonstrates the application of **Generative AI and Conditional GANs** to paired semantic layout-to-image translation.

The system learns from paired facade annotations and real facade photographs and attempts to generate a visually plausible facade corresponding to the provided semantic layout.

**Project focus:** Generative AI · Conditional GAN · Pix2Pix · U-Net · PatchGAN · Image-to-Image Translation