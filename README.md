# Fineflow3D
This is the corresponding code for a two-stage prediction of autonomous driving scene flow.
# FineFlow3D Project

This repository contains the implementation of `fineflow3d` for scene flow estimation. The code is built upon [OpenSceneFlow](https://github.com/optimization-ai/opensceneflow).

## 🛠️ Installation & Environment

The environment setup largely follows the original OpenSceneFlow configuration. Please refer to their repository for the base requirements.

**Important Additional Requirements:**
In addition to the base environment, you must install `torch_cluster` and `torch_scatter`. **Crucially**, these must be compatible with your specific PyTorch and CUDA versions.

You can find the installation instructions for these packages here: [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

Example installation (ensure you replace with your versions):
```bash
# Example only - check your specific CUDA/Torch version!
pip install torch-scatter -f [https://data.pyg.org/whl/torch-1.12.0+cu113.html](https://data.pyg.org/whl/torch-1.12.0+cu113.html)
pip install torch-cluster -f [https://data.pyg.org/whl/torch-1.12.0+cu113.html](https://data.pyg.org/whl/torch-1.12.0+cu113.html)
Data Preparation
Please organize your datasets (e.g., NuScenes) inside the data directory.

Bash

Project/
├── data/
│   ├── nuscenes/
│   └── ...
├── result/
├── train.py
└── ...
🚀 Training
To train the model on the NuScenes dataset, run the following command:

Bash

python train.py \
    model=fineflow3d \
    lr=2e-4 \
    epochs=15 \
    batch_size=4 \
    accumulate_grad_batches=4 \
    num_frames=5 \
    loss_fn=deflowLoss
📊 Results
Training and validation results on the NuScenes dataset are stored in the result/ folder. You can check the logs and checkpoints there.
