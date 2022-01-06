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
```

## Running Experiments
- Train the Networks
```bash
# Train ANCSH model
python train.py network=ancsh
python train.py network=npcs
optional arguments:
- train.input_data=<PATH_TO_TRAIN_INPUT>
- paths.result_dir=<PATH_TO_RESULT_FOLDER>
- network.max_epochs=<MAX_EPOCHS>
- network.num_workers=<NUM_WORKERS>
- network.lr=<LEARNING_RATE>
```
- Evaluate the Networks
```bash
python train.py network=ancsh eval_only=true inference_model=<PATH_TO_MODEL>
python train.py network=npcs eval_only=true inference_model=<PATH_TO_MODEL>
optional arguments:
- test.split=val (or test)
- test.input_data=<PATH_TO_TRAIN_INPUT>
- paths.result_dir=<PATH_TO_RESULT_FOLDER>
- network.num_workers=<NUM_WORKERS>
```
- Kinematic Constrained Optimization
```bash
# Use kinematic constrained optimization to infer the pred part pose 
python optimize.py optimization=sapien_urdf paths=shawn_lab ancsh_results_path=<PATH_TO_ANCSH_RESULTS> npcs_results_path=<PATH_TO_NPCS_RESULTS>
```
- Evalaution
```bash
# Do evalaution
python evaluate.py evaluation=sapien_urdf paths=shawn_lab combined_result_path=<PATH_TO_COMBINED_REUSLTS>
```




