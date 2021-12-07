# SPADES: Official Pytorch implementation (work in progress)

This repository is an implementation of algorithms for training sparse networks and unstructured pruning.

## Training 

We have an implementation of global magnitude pruning, as a baseline representation of dense-to-sparse methods for unstructured pruning.

### CIFAR10/100

- To train ResNet32 with SPADES on CIFAR100, open the file "main_GMP.ipynb" and run the command:

prune_gmp(dataset="cifar100", network="resnet")

- To train VGG19 use network="vgg" instead.
- For CIFAR10 experiments, use dataset="cifar10" instead. 

### Tiny ImageNet200

The commands are the same, but use dataset="tiny_imagenet" instead. The data should be downloaded and stored under a folder "tiny_imagenet" in the same directory, with subfolders "train" and "val" consisting of the training and testing data respectively.

## Runtime

The code was run on a Google GCP instance using Python 3.7.12 and PyTorch v1.10, and a Tesla V100 GPU.

## ACKNOWLEDGEMENTS

This library is based on the following repositories. 

- Picking Winning Tickets Before Training by Preserving Gradient Flow, GRASP (ICLR'20): https://github.com/alecwangcq/GraSP
- Layer-adaptive sparsities, LAMP (ICLR'21): https://github.com/jaeho-lee/layer-adaptive-sparsity


