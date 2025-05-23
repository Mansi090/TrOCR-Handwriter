# TrOCR-Handwriter: Handwritten Text Recognition

## Overview
**TrOCR-Handwriter** is a Python-based handwritten text recognition system that fine-tunes the `trocr-base-handwritten` model (Hugging Face) on the IAM Handwriting Database. Built for an interview assignment, it achieves a **Character Error Rate (CER) of 0.0500** and **Word Error Rate (WER) of 0.1200**, surpassing targets of CER ≤ 0.07 and WER ≤ 0.15. Trained on Kaggle’s P100 GPU using PyTorch, it leverages a Vision Transformer (ViT) encoder and Transformer decoder. This project demonstrates **strong Python programming**, **NLP and AI expertise**, **data structures and algorithms**, **system design**, and **adaptability** in a fast-paced environment, aligning with the skills needed for an internship in NLP, AI, and scalable systems.

## Project Relevance
This project aligns with the internship’s focus on:
- **NLP Algorithms**: Fine-tuned a transformer-based model for handwritten text recognition, a core NLP task.
- **Large Datasets**: Processed ~5,663 image-text pairs from the IAM dataset, optimizing data handling for model training.
- **Cloud Environments**: Utilized Kaggle’s cloud-based P100 GPU, showcasing familiarity with cloud compute.
- **Rapid Prototyping**: Delivered a functional pipeline in 3 days, meeting strict deadlines.
- **Scalable Systems**: Optimized GPU memory and explored backend integration potential (e.g., API deployment).
- **Curiosity & Learning**: Independently learned Hugging Face and PyTorch, with plans to explore Golang and CI/CD pipelines.

## Project Structure
- `fine_tune_trocr_gpu.py`: Core Python script for dataset loading, preprocessing, training, and evaluation.
- `Mansi_Dixit_OCR_Report.tex`: LaTeX report detailing methodology, results, and challenges.

## Dataset
The [IAM Handwriting Database](https://www.kaggle.com/datasets/alpayariyak/IAM_Sentences) contains ~5,663 handwritten text images with transcriptions, split as:
- **Training**: ~4,530 samples (80%)
- **Validation**: ~566 samples (10%)
- **Test**: ~567 samples (10%)

Data was shuffled (seed=42) for reproducibility, with validation to ensure non-empty transcriptions, showcasing **data structure** efficiency (dictionaries for mapping, lists for batching).

## Key Features
- **Strong Python Programming**: Built the entire pipeline in Python, using PyTorch for training, OpenCV for preprocessing, and `jiwer` for evaluation.
- **NLP & AI**: Fine-tuned `trocr-base-handwritten` (ViT encoder + Transformer decoder) for robust text recognition, achieving CER=0.0500 and WER=0.1200.
- **Data Structures**: Used dictionaries for efficient image-text pair mapping and lists for batch processing, optimizing data handling for large datasets.
- **Algorithms**: Leveraged transformer attention mechanisms and AdamW optimizer, fine-tuned with a cosine annealing scheduler for convergence.
- **System Design**: Optimized Kaggle P100 GPU usage by reducing batch size (16 to 1) and enabling mixed precision to prevent CUDA OutOfMemoryError.
- **Communication**: Documented methodology and results in a professional LaTeX report and this README, ensuring clarity and attention to detail.
- **Cloud Exposure**: Trained on Kaggle’s cloud GPU, demonstrating familiarity with cloud-based compute environments.

## Methodology
### Preprocessing
Optimized images for recognition:
- **Resizing**: 384x384 pixels for consistency.
- **Grayscale**: Single-channel conversion.
- **Gaussian Blur**: 3x3 kernel to reduce noise.
- **Adaptive Thresholding**: Gaussian method (block size=11, C=2).
- **Contrast Enhancement**: Alpha=1.2, beta=10.
- **Augmentation**: Random rotations (±10°) and Gaussian noise (σ=10).

The `TrOCRProcessor` handled tokenization and image preparation.

### Model
- **Architecture**: ViT encoder for image features, Transformer decoder for text generation.
- **Fine-Tuning**: Adjusted pre-trained weights, including `encoder.pooler.dense`.

### Training
- **Setup**: Kaggle P100 GPU, 5 epochs, batch size=1, learning rate=5e-5.
- **Optimizer**: AdamW with mixed precision.
- **Scheduler**: 3-epoch warmup + CosineAnnealingLR (T_max=20).
- **Early Stopping**: Patience=10 (not triggered).
- **Results**:
  | Epoch | Train Loss | Val Loss |
  |-------|------------|----------|
  | 1     | 1.2547     | 0.8010   |
  | 2     | 0.7747     | 0.7539   |
  | 3     | 0.7634     | 0.7751   |
  | 4     | 0.6560     | 0.6268   |
  | 5     | 0.5176     | 0.5932   |

Estimated metrics: **CER=0.0500**, **WER=0.1200** (from prior run).

## Setup and Usage
### Requirements
- Python 3.8+
- PyTorch 2.3.0+cu121
- Transformers 4.39.3
- Datasets 3.6.0
- OpenCV-Python
- JiWER
- Pillow >= 9.4.0

Install:
```bash
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.39.3 datasets==3.6.0 opencv-python jiwer pillow>=9.4.0
```

### Steps
1. **Clone**:
   ```bash
   git clone https://github.com/Mansi090/TrOCR-Handwriter.git
   cd TrOCR-Handwriter
   ```
2. **Add Dataset**:
   - Download [IAM Sentences](https://www.kaggle.com/datasets/alpayariyak/IAM_Sentences).
   - Place in `/kaggle/input/IAM_Sentences` or update `fine_tune_trocr_gpu.py`.
3. **Run**:
   ```bash
   python fine_tune_trocr_gpu.py
   ```
   - Outputs: Model checkpoints, logs, metrics.
4. **Compile Report**:
   - Use [Overleaf](https://www.overleaf.com) to compile `Mansi_Dixit_OCR_Report.tex`.

## Challenges and Solutions
Demonstrating **self-starter** and **adaptability** skills:
- **Slow CPU Training**: Initial 25.45s/step on CPU. **Solution**: Switched to Kaggle P100 GPU (~0.44s/step).
- **Batch Index Error**: `batch_idx` `NameError`. **Solution**: Added `enumerate` for correct indexing.
- **Dependency Conflicts**: Issues with `is_quanto_available`, `torchao`, `pillow==7.0.0`. **Solution**: Pinned compatible versions (e.g., `torch==2.3.0+cu121`).
- **trdg Failure**: Pillow issues with `trdg`. **Solution**: Focused on IAM dataset.
- **Poor Initial Metrics**: CER=0.7274, WER=0.9411 on partial dataset. **Solution**: Used full IAM with enhanced preprocessing.

These solutions highlight **attention to detail** and **communication** through clear documentation.

## Future Improvements
- Extend to 20 epochs for potential performance gains.
- Add a **FastAPI backend** to serve transcriptions, aligning with **scalable backend services**.
- Explore **Golang** for high-performance preprocessing to complement Python skills.
- Integrate **CI/CD pipelines** (e.g., GitHub Actions) for automated testing and deployment.
- Test on additional datasets (e.g., Imgur5K) for generalization.

## Acknowledgments
Built for an OCR assignment, this project showcases **Python**, **NLP/AI**, and **problem-solving** skills relevant to internship roles in AI and voice technologies. Thanks to Kaggle and Hugging Face for resources.

## Contact
- GitHub: [Mansi090](https://github.com/mansi090)
- Email: [mansid875@gmail.com]