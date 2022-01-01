# Category Level Articulated Object Pose Estimation Pytorch

Implementation of the paper *[Category-Level Articulated Object Pose Estimation](https://arxiv.org/abs/1912.11913)*  
Checkout the official code release for the paper, pretrained model, data at [dragonlong/articulated-pose](https://github.com/dragonlong/articulated-pose).

## Installation Requirements
* [Pytorch](https://pytorch.org/) (tested with v1.7.1)

Our implementation is based on the Pointnet2/Pointnet++ PyTorch Implemention from [erikwijmans/Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)

### Installation
- git clone the repo
```bash
git clone https://github.com/3dlg-hcvc/ANCSH-pytorch.git
cd ANCSH-pytorch
```
- create a python environment
```bash
conda create -n ancsh python=3.7
conda activate ancsh
pip install -e .
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install h5py
pip install scipy
```

## Running Experiments
- train the model
```bash
# Train the two models for SAPIEN drawer dataset
python train.py network=ancsh_sapien test=False paths=shawn_lab
python train.py network=npcs_sapien test=False paths=shawn_lab
# Inference on the trained models
python train.py network=ancsh_sapien test=True paths=shawn_lab inference_model=<PATH_TO_MODEL>
python train.py network=npcs_sapien test=True paths=shawn_lab inference_model=<PATH_TO_MODEL>
# Use kinematic constrained optimization to infer the pred part pose 
python optimize.py 
```




