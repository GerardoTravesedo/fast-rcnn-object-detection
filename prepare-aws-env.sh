#!/usr/bin/env bash

# This script is intended to be used with the Deep Learning Base AMI (Amazon Linux 2) Version 18.0 (ami-05603f16d12d9996f)
# This version comes with the following:
#   - Python3
#   - Git
#   - Nvidia driver version: 418.40.04
#   - CUDA versions available: cuda-10.0 (default), cuda-8.0, cuda-9.0, cuda-9.2
#   - Libraries: cuDNN, NCCL, Intel MKL-DNN
#
# More info about the different Deep Learning AMIs: https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html


# The requirements for the version of TF that we use are the following:
#       Version			          Python version   Compiler   Build tools	    cuDNN	   CUDA
# tensorflow_gpu-1.13.1	       2.7, 3.3-3.6	    GCC 4.8	  Bazel 0.19.2	   7.4	   10.0

# tensorflow: Latest stable release for CPU-only (Ubuntu and Windows)
# tensorflow-gpu: Latest stable release with GPU support (Ubuntu and Windows) -> will only work with cuda + cudnn installed

# Clone projects
git clone https://github.com/GerardoTravesedo/fast-rcnn-object-detection.git
git clone https://github.com/GerardoTravesedo/ml-common.git

# Install python dependencies
cd /home/ec2-user/fast-rcnn-object-detection
sudo pip3 install -r requirements-gpu.txt
# Uncomment next line of there are libraries missing for opencv-python
# sudo pip install opencv-python-headless

# Set up PYTHONPATH env variable
export PYTHONPATH=$PYTHONPATH:/home/ec2-user/ml-common
export PYTHONPATH=$PYTHONPATH:/home/ec2-user/fast-rcnn-object-detection/object_detection
echo $PYTHONPATH

# Getting dataset
mkdir -p /home/ec2-user/datasets/images/pascal-voc/transformed/
# We need to set credentials before or set up a role
aws s3 cp s3://gerardo-ml-datasets/images/pascal-voc/transformed/ /home/ec2-user/datasets/images/pascal-voc/transformed/ --recursive

# Run network
# Update config file if necessary
cd /home/ec2-user/fast-rcnn-object-detection/object_detection/detection
python3 rcnn_detection.py