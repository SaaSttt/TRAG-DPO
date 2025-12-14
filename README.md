# TRAG-DPO: Enhancing Visual Question Answering in Tomato Disease Diagnosis

<div>
    <a href="https://github.com/YourUsername/TRAG-DPO/stargazers"><img src="https://img.shields.io/badge/Python-3.8%2B-blue"></a>
    <a href="https://github.com/YourUsername/TRAG-DPO/network/members"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</div>

**Haoran Zhu\*, Xu Lu\*, Jialie Shen, Qi Zhang, Xinming Wang, Liang Zhang**

*Shandong Agricultural University | City St George’s, University of London*

## 📢 News

- **[2025-12]** Code and datasets are released!

## 📖 Introduction

This repository contains the official implementation of **TRAG-DPO**, a novel framework designed to enhance Visual Question Answering (VQA) in tomato disease diagnosis.

Existing general-purpose Vision-Language Models (VLMs) often suffer from **hallucinations** and a lack of **domain-specific knowledge** in agriculture. To address this, TRAG-DPO integrates:

1.  **Fine-Grained Vision Transformer (FG-ViT)**: To capture subtle lesion details.
2.  **Hybrid Retrieval (RAG)**: A coarse-to-fine mechanism combining global/local image features and textual ranking.
3.  **DPO Fine-Tuning**: Direct Preference Optimization to align model responses with expert knowledge and reduce redundancy.

![image-20251213173907668](fig\framework.png)
												Framework of the TRAG-DPO model with four main components.

## 🛠️ Environment Setup

We recommend using Anaconda to manage the environment.

```bash
# Clone the repository
git clone https://github.com/SaaSttt/TRAG-DPO.git
cd TRAG-DPO

# Create environment using the provided yaml file
conda env create -f environment.yaml

# Activate the environment
conda activate trag-dpo
```

## 📂 Data & Weights Preparation

Since the datasets and weights are large, we provide external download links. Please download and place them in the corresponding directories.

| Resource           | Description                       | Path in Repo         | Download Link                                                |
| ------------------ | --------------------------------- | -------------------- | ------------------------------------------------------------ |
| **LLaVA-1.5-7B**   | Base pre-trained VLM weights      | llava-v1.5-7b/       | [https://huggingface.co/liuhaotian/llava-v1.5-7b/tree/main]  |
| **CDDM**           | Images & JSON for Stage 1 SFT     | data/cddm_tomato/    | [https://github.com/UnicomAI/UnicomBenchmark/tree/main/CDDMBench] |
| **Knowledge Base** | Tomato disease textual KB for RAG | data/knowledge_base/ | [https://huggingface.co/datasets/SaaSttt/Tomato_Knowledge_Base] |
| **DPO Dataset**    | Preference pairs for Stage 3      | data/dpo_data/       | [https://huggingface.co/datasets/SaaSttt/DPO_Dataset]        |

## 🚀 Training Pipeline

The training consists of three stages. You can run the provided scripts to reproduce our results.

### Stage 1: Visual Instruction Tuning (LoRA)

We fine-tune LLaVA-1.5 on the **tomato subset** of the CDDM dataset.

```
# Run SFT (Supervised Fine-Tuning)
bash /TRAG-DPO/LLaVA-1.1.3/scripts/v1_5/finetune_task_lora.sh
```

### Stage 2: Hybrid Retrieval (RAG)

This module retrieves domain knowledge. It does not require training the LLM but requires the Knowledge Base to be indexed.

- **Coarse Retrieval**: Uses FG-ViT features + CLIP text encoder.
- **Fine Retrieval**: Uses Sentence-BERT for re-ranking.

To test the retrieval module separately:

```
python /TRAG-DPO/LLaVA-1.1.3/test_re_vqa.py
```

### Stage 3: DPO Preference Fine-Tuning

We use Direct Preference Optimization to align the RAG-augmented model.

```
# Run DPO Fine-Tuning
bash /TRAG-DPO/DPO-RAG-main/scripts/train_dpo_2stages.sh
```

*Note: This stage requires the merged model from Base LLaVA + Stage 1 weights as the reference model.

## ⚡ Inference / Demo

Click the tomato leaf disease image to chat with TRAG-DPO about disease information.

![image-20251214160246480](fig\vqa.png)



## 📂 Project Structure

```
TRAG-DPO/
├── data                        # Dataset directory
├── DPO-RAG-main                # Core implementation of TRAG-DPO
├── LLaVA-1.1.3                 # Base LLaVA framework
├── fig                         # Images and figures for Paper
└── environment.yaml            # Conda environment configuration file
└── README.md                   # README
```

## 🙏 Acknowledgement

We acknowledge the following open-source projects and datasets that facilitated our research:

*   **LLaVA**: Our model is built upon the official LLaVA codebase.
    *   https://github.com/haotian-liu/LLaVA/releases/tag/v1.1.3
*   **CDDM Dataset**: Used for the visual instruction fine-tuning stage.
    *   https://github.com/UnicomAI/UnicomBenchmark/tree/main/CDDMBench
*   **Sentence-Transformers**: Used for the fine-grained retrieval mechanism.
    *   https://github.com/UKPLab/sentence-transformers
*   **PlantVillage**: Used as the source for constructing our preference dataset.
    *   https://github.com/spMohanty/PlantVillage-Dataset
*   **MMed-RAG**: We referenced the original Direct Preference Optimization implementation.
    *   https://github.com/richard-peng-xia/MMed-RAG

- **CLIP **: Used for image-text feature alignment in the coarse-grained retrieval stage.
  - https://github.com/bubbliiiing/clip-pytorch

- **AgriCLIP**: Used as an agricultural domain-adapted CLIP model for robust image–text representation learning.
  - https://github.com/umair1221/AgriCLIP

## 📜 Citation

If you find our code or dataset useful, please cite our work (Paper currently under review):

```
@article{tragdpo,
  title={TRAG-DPO: Enhancing Visual Question Answering in Tomato Disease Diagnosis via Hybrid Retrieval and DPO Preference Fine-Tuning},
  author={Zhu, Haoran and Lu, Xu and Shen, Jialie and Zhang, Qi and Wang, Xinming and Zhang, Liang},
  journal={xxx},
  year={2025}
}
```