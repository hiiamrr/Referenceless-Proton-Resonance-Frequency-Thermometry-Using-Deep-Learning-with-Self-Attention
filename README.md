# Referenceless Proton Resonance Frequency Thermometry Using Deep Learning with Self-Attention

Official implementation of "Referenceless Proton Resonance Frequency Thermometry Using Deep Learning with Self-Attention" - a deep learning framework for accurate, referenceless temperature mapping in MR-guided focused ultrasound therapy, featuring the C-SANet (Complex-valued Self-Attention Network) architecture.

## Abstract

This repository contains the implementation of C-SANet (Complex-valued Self-Attention Network), a novel deep learning framework for referenceless PRF thermometry. The method accurately reconstructs background phase and yields robust temperature estimates by modeling multi-scale spatial structure and contextual dependencies directly in the complex domain, eliminating the need for baseline reference scans.

## Installation

### Prerequisties
- Python 3.12.0 or higher
- PyTorch 2.4.1 or higher
- CUDA 12.4+ (for GPU acceleration)

### Install dependencies:
```bash
pip install -r requirements.txt
```

## Running
Run main.py can launch the training processing.
```bash
python main.py --gpus $GPUS
```
where $GPUS is the GPU ranks like --gpus 0 or using multiple GPUs as --gpus 0,1.
Other hyperparameters about models and datasets/dataloader can be modified at ./configs.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
