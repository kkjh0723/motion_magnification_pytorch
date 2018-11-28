# motion_magnification_pytorch
Reproducing Learning-based Video Motion Magnification in pytorch

We write the code with reference to the original [tensorflow implementation](https://github.com/12dmodel/deep_motion_mag) from the authors

This code is tested on python 3.5 and pytorch 0.4.1.

# Data
We use the dataset opened by the authors.
Please refer to the [authors repository](https://github.com/12dmodel/deep_motion_mag).

# Train the network
    python train.py [--additional option]

ex) to run training on gpu number 0

    python train.py --gpu 0

# reference
1) https://github.com/12dmodel/deep_motion_mag
2) Oh, Tae-Hyun, et al. "Learning-based Video Motion Magnification." arXiv preprint arXiv:1804.02684 (2018).
