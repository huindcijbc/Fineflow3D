# Fineflow3D
This is the corresponding code for a two-stage prediction of autonomous driving scene flow.
# FineFlow3D Project

This repository contains the implementation of `fineflow3d` for scene flow estimation. The code is built upon [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow).

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
```

## 📂 Data Preparation

To train the full dataset, please refer to the https://github.com/KTH-RPL/OpenSceneFlow/blob/main/dataprocess/README.md for raw data download and h5py files preparation.

Please organize your datasets (e.g., NuScenes) inside the data directory.
```bash
Project/
├── data/
│   ├── nuscenes/
│   └── ...
├── result/
├── train.py
└── ...
```
## 🚀 Train

To train a model using the NuScenes dataset, run the following command, You can also modify `batch_size` and `accumulate_grad_batches` according to your computer's configuration:
```bash
python train.py \
    model=fineflow3d \
    lr=2e-4 \
    epochs=15 \
    batch_size=4 \
    accumulate_grad_batches=4 \
    num_frames=5 \
    loss_fn=deflowLoss
```

## ⚖️ Evaluation

You can also evaluate the model on the NuScenes validation set using the following command:
```bash
python eval.py checkpoint=path_to_checkpoint dataset_path=/your_path/data/nuscenes/h5py/full
```
The evaluation results obtained on this dataset are shown below:
<img width="2553" height="1113" alt="val" src="https://github.com/user-attachments/assets/907268ba-8bdb-48bb-ab3b-ce1622395349" />

For additional visualization results, please refer to OpenSceneFlow.

## 🙏 Acknowledgements

This code is based on the DeFlow code by Qingwen Zhang. I would like to express my deepest gratitude to her for this excellent work.

Additionally, I would like to thank [Insert Name Here] for their support with the NuScenes dataset for this task.
📊 Results
Training and validation results on the NuScenes dataset are stored in the result/ folder. You can check the logs and checkpoints there.
