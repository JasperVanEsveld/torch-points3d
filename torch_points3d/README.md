# Torch Points 3D

Contains many different algorithms to run on different datasets.
From segmentation to classification and object detection.
Code is similar to what is found in https://github.com/torch-points3d/torch-points3d

Main difference being the added models:
- gmm:MoNet
- harmonics:Harmonic

## How to run

Running train.py you can use the models and datasets defined here.

Example:

``python train.py task=segmentation models=segmentation/harmonics model_name=Harmonic data=segmentation/shapenet-small training.batch_size=1``