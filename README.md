# ESNet: A Resource-Aware Stereo Matching Network for Disparity Estimation on Edge Devices

This repository provides the official implementation of **ESNet**, a resource-aware stereo matching network designed for disparity estimation on resource-constrained edge devices.

ESNet adopts an efficient **2D/1D cascaded architecture**. It avoids explicit high-dimensional 3D cost-volume construction and 3D convolution-based cost aggregation. Instead, it combines recurrent correlation matching, progressive cascade refinement, and confidence-aware gated refinement to balance disparity quality, inference latency, and system-level memory usage.

The method is designed for low-power edge visual perception and depth-perception display systems, where stable disparity prediction, boundary preservation, and deployment feasibility are important.

## Highlights

* A resource-aware stereo matching network for edge-device disparity estimation.
* A compact 1D epipolar correlation representation without explicit 3D cost volumes.
* Progressive cascade refinement and confidence-aware gated refinement for local detail recovery.
* Multi-scale OHEM-L1 supervision with auxiliary physical regularization terms.
* TensorRT FP16 deployment analysis on NVIDIA Jetson Nano 4GB.

## Repository Structure

```text
ESNet/
├── configs/                 # YAML configuration files for training, evaluation, and ablation
├── data/                    # Dataset loaders and data transforms
├── engines/                 # Training and evaluation engines
├── losses/                  # Multi-scale loss and physical constraint losses
├── models/                  # ESNet model architecture
├── scripts/                 # Training, evaluation, submission, and ONNX export scripts
├── tests/                   # Analysis scripts for ablation, zero-shot, depth, and complexity
├── utils/                   # Utility functions
├── requirements.txt         # Python dependencies
└── README.md
```

## Installation

Create a new Python environment and install the required dependencies:

```bash
conda create -n esnet python=3.8 -y
conda activate esnet

pip install -r requirements.txt
```

The main dependencies include:

```text
torch
torchvision
numpy
opencv-python
Pillow
PyYAML
matplotlib
tensorboard
onnx
thop
tqdm
```

For TensorRT deployment, please install NVIDIA TensorRT separately according to your device and CUDA environment.

## Datasets

This project uses **Scene Flow** for pre-training and **KITTI 2015** for fine-tuning, evaluation, and submission.

Please download the datasets from their official websites and arrange them as follows.

### KITTI 2015

```text
KITTI2015/
├── training/
│   ├── image_2/
│   ├── image_3/
│   ├── disp_occ_0/
│   └── disp_noc_0/
└── testing/
    ├── image_2/
    └── image_3/
```

Then update the dataset path in the corresponding configuration files, for example:

```yaml
data:
  root_dir: /path/to/KITTI2015
```

### Scene Flow

Please prepare the Scene Flow dataset according to the dataloader settings in `data/` and update the path in:

```text
configs/train_sceneflow.yaml
configs/sceneflow_config.yaml
```

## Pretrained Weights

We provide the best-performing checkpoint of ESNet, corresponding to the **Accurate / Full ESNet** setting reported in the paper.

The Fast, Balanced, and Accurate inference modes are derived from the same trained model by enabling or disabling the PRN and Gate modules during inference or ONNX export.

The pretrained checkpoint can be downloaded from: sha256:68da797d9b6f814b7b134f42d0456d3f1469c5e26ef46bbe4ea20f38ada21b9f

After downloading the checkpoint, place it under:

```text
pretrained/esnet_full_kitti2015_best.pth
```

Recommended checkpoint name:

```text
esnet_full_kitti2015_best.pth
```

## Training

### Scene Flow Pre-training

```bash
python scripts/train.py \
  --config configs/train_sceneflow.yaml \
  --ohem_ratio 0.3 \
  --gpu 0
```

### KITTI 2015 Fine-tuning

```bash
python scripts/train.py \
  --config configs/kitti_finetune_final.yaml \
  --resume pretrained/esnet_full_kitti2015_best.pth \
  --ohem_ratio 0.3 \
  --gpu 0
```

If training from a Scene Flow pre-trained checkpoint, replace the `--resume` path with the corresponding pre-trained weight.

## Evaluation

### KITTI Validation Evaluation

```bash
python scripts/eval_ohem.py \
  --config configs/kitti_finetune_final.yaml \
  --resume pretrained/esnet_full_kitti2015_best.pth \
  --gpu 0
```

### Official KITTI 2015 Submission

To generate KITTI 2015 test-set disparity maps:

```bash
python scripts/submit_kitti.py
```

The generated disparity maps should be saved under:

```text
kitti_submission/disp_0/
```

Please make sure the dataset path and checkpoint path in `scripts/submit_kitti.py` or the corresponding config file are correctly set before running.

## Ablation Experiments

The following configuration files are provided for ablation studies:

```text
configs/ablation_1_baseline.yaml
configs/ablation_2_edge.yaml
configs/ablation_3_edge_ohem.yaml
configs/ablation_4_full.yaml
configs/ohem_ablation.yaml
```

Example command:

```bash
python scripts/train.py \
  --config configs/ohem_ablation.yaml \
  --resume pretrained/esnet_full_kitti2015_best.pth \
  --ohem_ratio 0.3 \
  --gpu 0
```

To evaluate an ablation checkpoint:

```bash
python scripts/eval_ohem.py \
  --config configs/ohem_ablation.yaml \
  --resume path/to/ablation_checkpoint.pth \
  --gpu 0
```

## Zero-shot Evaluation

Zero-shot evaluation is used to test the transfer ability from Scene Flow to KITTI 2015 without KITTI fine-tuning.

```bash
python tests/zero_shot.py
```

Before running, please check the dataset path and checkpoint path in the script or adapt them to your local environment.

## Boundary-region Analysis

Boundary-region analysis evaluates disparity quality around image-gradient-based object contours.

```bash
python tests/boundary_region_analysis_six.py \
  --config configs/boundary_kitti.yaml
```

This analysis computes metrics such as Boundary D1 and Boundary EPE within boundary regions generated by Sobel gradients and dilation.

## Depth Recovery Error Analysis

Depth recovery analysis converts predicted disparity maps into depth values using KITTI stereo calibration parameters and reports errors across different distance ranges.

```bash
python tests/depth_ranging_analysis.py \
  --config configs/depth_kitti.yaml
```

The reported metrics include:

```text
Depth MAE
Depth RMSE
AbsRel
DispEPE
```

## Model Complexity and Inference Profiling

To compute parameter count, MACs, and PyTorch CUDA inference latency for the three inference modes:

```bash
python tests/profile_complexity.py
```

The three inference modes are:

```text
Fast / Baseline:          use_prn=False, use_gate=False
Balanced / PRN:           use_prn=True,  use_gate=False
Accurate / Full ESNet:    use_prn=True,  use_gate=True
```

## ONNX Export and TensorRT Deployment

To export the three deployment modes to ONNX:

```bash
python scripts/export_three_modes_onnx.py \
  --mode all \
  --weight pretrained/esnet_full_kitti2015_best.pth \
  --output-dir onnx_three_modes
```

To export a single mode:

```bash
python scripts/export_three_modes_onnx.py \
  --mode fast \
  --weight pretrained/esnet_full_kitti2015_best.pth \
  --output-dir onnx_three_modes
```

Supported modes:

```text
fast
balanced
accurate
all
```

The exported ONNX files are intended for TensorRT FP16 deployment. For operators such as GridSample, TensorRT plugin support may be required.

## Main Results

### KITTI 2015 Official Test Set

| Method | D1-bg (%) | D1-fg (%) | D1-all (%) |
| ------ | --------: | --------: | ---------: |
| ESNet  |      2.93 |      5.63 |       3.38 |

### Jetson Nano Deployment

Input resolution: `320 × 640`
Precision: `TensorRT FP16`
Batch size: `1`

| Mode     | Architecture          |  Latency |  FPS | D1-all |
| -------- | --------------------- | -------: | ---: | -----: |
| Fast     | Baseline              | 150.6 ms | 6.64 |   3.94 |
| Balanced | Baseline + PRN        |   885 ms | 1.13 |   2.99 |
| Accurate | Baseline + PRN + Gate |  1063 ms | 0.94 |   2.88 |

Note: The Jetson Nano D1-all values are used only for mode-level deployment comparison and should not be directly compared with the official KITTI 2015 test-set accuracy.

## Notes

* Dataset files and trained checkpoints are not included in the repository.
* Please update dataset paths in the YAML configuration files before training or evaluation.
* TensorRT deployment results may vary depending on CUDA, TensorRT version, plugins, and hardware configuration.
* The provided scripts are intended for research reproduction and may need path adaptation for different environments.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{esnet2026,
  title={ESNet: A Resource-Aware Stereo Matching Network for Disparity Estimation on Edge Devices},
  author={Keke Shang and Song Wang },
  journal={Displays},
  year={2026}
}
```

The citation information will be updated after publication.

## License

This project is released for academic research purposes only. Please check the license file for detailed terms.

## Acknowledgements

This project uses public stereo matching datasets, including Scene Flow and KITTI 2015. We also thank the open-source PyTorch community and prior stereo matching works that inspired this research.
