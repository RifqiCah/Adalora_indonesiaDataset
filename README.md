# 🇮🇩 AdaLoRA-Indo: Parameter-Efficient Fine-Tuning for Indonesian Tasks

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/Transformers-HuggingFace-yellow?logo=huggingface)
![PEFT](https://img.shields.io/badge/PEFT-AdaLoRA-success)

This repository contains the implementation and experimental results of investigating **Adaptive Low-Rank Adaptation (AdaLoRA)** for fine-tuning compact Transformer models on Indonesian language tasks.

> **Abstract:** The rapid scaling of Pre-trained Language Models (PLMs) has made full fine-tuning computationally prohibitive. While techniques like LoRA are effective, they use a fixed-rank strategy. This project investigates **AdaLoRA** to fine-tune compact models on Indonesian tasks (NLU, QA, NLG). Our experiments demonstrate extreme parameter efficiency, updating as few as **0.17% to 0.31%** of parameters.

## 🔬 Research Focus

We target the underrepresented domain of efficient NLP for the **Indonesian Language** using compact models suitable for edge devices.

### Key Contributions
* **Adaptive Rank Allocation:** Applying AdaLoRA to dynamically distribute the parameter budget to important model modules using Singular Value Decomposition (SVD).
* **Extreme Efficiency:** Achieving competitive performance while updating less than 1% of total parameters.
* **Multi-Domain Evaluation:** Tested across three distinct domains: NLU (NER & Sentiment), Question Answering (SQuAD-id), and NLG (Summarization).

## 🏗️ Methodology

Unlike standard LoRA which uses a fixed rank (e.g., r=8) across all layers, AdaLoRA iteratively prunes less important singular values during training.

```mermaid
graph LR
    subgraph "Standard LoRA"
    A[Input] --> B[Frozen Weights]
    A --> C[Fixed Rank Matrix A]
    C --> D[Fixed Rank Matrix B]
    B --> E[Output]
    D --> E
    end

    subgraph "AdaLoRA (Adaptive)"
    F[Input] --> G[Frozen Weights]
    F --> H[Adaptive Matrix]
    H --> I{SVD-based Pruning}
    I -->|Allocates Higher Rank to Critical Layers| J[Optimized Adapter]
    G --> K[Output]
    J --> K
    end
