## Install

This project follows the environment design of the original [**FlashOcc**](https://github.com/Yzichen/FlashOCC) repository, with additional dependencies verified on our side. The following instructions are tested on **Ubuntu 20.04 / 22.04**, **CUDA 11.3–11.6**, and **PyTorch ≥1.10**.

### Step 1. Create and Activate Environment

```bash
conda create --name MambaOcc python=3.8.5 -y
conda activate MambaOcc
```

### Step 2. Install PyTorch and Core Dependencies

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
    -f https://download.pytorch.org/whl/torch_stable.html
```


### Step 3. Install OpenMMLab Libraries

```bash
pip install -U openmim
mim install mmcv-full==1.5.3
pip install mmdet==2.25.1 mmsegmentation==0.25.0
```

### Step 4. System and CUDA Setup

```bash
sudo apt-get install -y python3-dev libevent-dev
sudo apt-get install -y build-essential
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_ROOT=/usr/local/cuda
pip install pycuda
```


### Step 5. Install Dataset & Utility Packages

```bash
pip install lyft_dataset_sdk nuscenes-devkit
pip install plyfile scikit-image tensorboard trimesh==2.35.39
pip install networkx==2.2 numba==0.53.0 numpy==1.23.5
pip install setuptools==59.5.0 yapf==0.40.1
pip install einops
pip install triton timm==0.4.12 chardet yacs submitit tensorboardX fvcore seaborn
pip install ipdb
pip install mmengine==0.10.1
cd VMamba
cd kernels/selective_scan && pip install .
cd ../../..
cd ops_dcnv3
python setup.py install
cd ..
```


### Step 6. Clone and Install Repositories

```bash
git clone https://github.com/Hub-Tian/MambaOcc.git
cd MambaOCC

# Clone mmdetection3d
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc4
pip install -v -e .
cd ..

git submodule add https://github.com/MzeroMiko/VMamba.git VMamba git submodule update --init --recursive

# Install this project's modules
cd projects
pip install -v -e .
cd ..
```


## Datasets and Models

### Step 1. Prepare nuScenes Dataset

You can download nuScenes dataset [HERE](https://www.nuscenes.org/download) and create the pkl by running (following FlashOcc):
```shell
python tools/create_data_bevdet.py
```
thus, the folder will be ranged as following:
```shell script
MambaOcc/
└── data
    └── nuscenes
        ├── v1.0-trainval (existing)
        ├── sweeps  (existing)
        ├── samples (existing)
        ├── bevdetv2-nuscenes_infos_train.pkl (new)
        └── bevdetv2-nuscenes_infos_val.pkl (new)
```

### Step 2. Prepare Occupancy GT

Download Occ3D-nuScenes occupancy GT from [gdrive](https://drive.google.com/file/d/1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY/view?usp=drive_link), unzip it, and save it to `data/nuscenes/gts`.

The folder will be ranged as following:

```
MambaOcc/
└── data
    └── nuscenes
        ├── v1.0-trainval
        ├── sweeps
        ├── samples
        ├── bevdetv2-nuscenes_infos_train.pkl
        ├── bevdetv2-nuscenes_infos_val.pkl
        └── gt
```

### Step 3. Prepare Pretrained Models

Download the pretrained model:

```
flashocc-r50-256x704.pth
```

from [Google Drive](https://drive.google.com/file/d/1k9BzXB2nRyvXhqf7GQx3XNSej6Oq6I-B/view) and place it in `pretrained_model/`.

Download the pretrained model:

```
vssm_base_0229_ckpt_epoch_237.pth 
vssm_small_0229_ckpt_epoch_222.pth 
vssm_tiny_0230_ckpt_epoch_262.pth
```

from [VMamba](https://github.com/MzeroMiko/VMamba/blob/main/assets/performance.md) and place it in `pretrained_model/`.

## Train & Val

```bash
cd MambaOcc

# for train
./tools/dist_train.sh $config $gpu_num
# e.g.
./tools/dist_train.sh projects/configs/mambaocc/mambaocc-stbase-4d-stereo-512x1408_4x4_1e-4_dcnv3_3x3_large.py 8

# for test
./tools/dist_test.sh $config $ckpt_path $gpu_num --eval mAP
# e.g.
./tools/dist_test.sh projects/configs/mambaocc/mambaocc-stbase-4d-stereo-512x1408_4x4_1e-4_dcnv3_3x3_large.py work_dirs/mambaocc-stbase-4d-stereo-512x1408_4x4_1e-4_dcnv3_3x3_large/epoch_24_ema.pth 8 --eval mAP

```

