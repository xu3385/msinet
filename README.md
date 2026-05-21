
# Multi-Scale Interaction Networkfor Stereo Image Super Resolution

MSTNet / MSINet — Stereo Image Super-Resolution (Quick Start)

This repository contains an implementation of a Multi-Scale Interaction Network (MSINet / MSTNet) adapted from the BasicSR framework for stereo (binocular) image super-resolution (x4).

Current status: Code is organized and includes training and testing scripts, example configuration files, and example weights (see experiments/).

Directory overview
- `basicsr/`: core code (data loaders, models, training/testing pipelines, utilities).
- `options/`: example YAML configurations for training and testing (e.g. `options/train/MSINet_x4.yml`, `options/test/MSINet_x4.yml`).
- `experiments/`: trained checkpoints and example weights (e.g. `MSINet_x4.pth`).
- `data/`: dataset and preprocessing utilities.
MSTNet / MSINet — Stereo Image Super-Resolution (Quick Start)

This repository contains an implementation of a Multi-Scale Interaction Network (MSINet / MSTNet) adapted from the BasicSR framework for stereo (binocular) image super-resolution (x4).

Current status: Code is organized and includes training and testing scripts, example configuration files, and example weights (see experiments/).

Directory overview
- `basicsr/`: core code (data loaders, models, training/testing pipelines, utilities).
- `options/`: example YAML configurations for training and testing (e.g. `options/train/MSINet_x4.yml`, `options/test/MSINet_x4.yml`).
- `experiments/`: trained checkpoints and example weights (e.g. `MSINet_x4.pth`).
- `data/`: dataset and preprocessing utilities.

Key features
- Built on top of BasicSR; supports distributed training, TensorBoard and optional Weights & Biases integration.
- Supports `PairedStereoImageDataset` for stereo/bilateral image SR tasks.

Quick start (macOS / Linux)

1) Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# To install in editable mode and compile CUDA extensions when needed:
# python setup.py develop
```

2) Prepare data
- Follow `dataroot_gt` and `dataroot_lq` fields in the YAML files under `options/` to organize datasets on disk.
- If you prefer LMDB, see `basicsr/utils/create_lmdb.py`.

3) Training example

```bash
# Single-process single-GPU (change `num_gpu` in the YAML if needed)
python basicsr/train.py -opt options/train/MSINet_x4.yml --launcher none

# The training script also accepts single-image input/output for quick inference:
python basicsr/train.py -opt options/train/MSINet_x4.yml --input_path /path/in.png --output_path /path/out.png
```

4) Testing / Inference example

```bash
python basicsr/test.py -opt options/test/MSINet_x4.yml --launcher none
```

5) Using pretrained weights
- Place checkpoint files under `experiments/` or set `path.pretrain_network_g` in the YAML to point to a checkpoint.

Common configuration items
- `options/*.yml`: `num_gpu`, `datasets.*.dataroot_*`, `train.total_iter`, `logger.print_freq`, and similar fields are commonly adjusted for experiments.
- For distributed training, use `--launcher pytorch` and configure `dist_params` in the YAML.

Acknowledgements
- This project is adapted from BasicSR.

If you want a more detailed README (training hyperparameters explanation, data conversion examples, or CI/test scripts), I can expand it further.
