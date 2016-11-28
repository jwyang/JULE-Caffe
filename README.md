# JULE-Caffe
Caffe code for our CVPR 2016 paper "Joint unsupervised learning of deep representations and image clusters". The Torch version code can be found [here](https://github.com/jwyang/JULE-Torch). 

### NOTE

**I have not yet finished cleaning up this code since I have to do it remotely. But it will be finished in several days. Once it is finished, this NOTE will be gone.**

### Overview

This project is a Caffe implementation for our CVPR 2016 [paper](https://arxiv.org/abs/1604.03628), which performs jointly unsupervised learning of deep CNN and image clusters.

### Acknowledgement

A great thanks to [happynear](https://github.com/happynear) for providing an awesome windows version [Caffe-Windows](https://github.com/happynear/caffe-windows). It is based on his code that we developed our JULE algorithm.

### License

This code is released under the MIT License (refer to the LICENSE file for details).

### Citation
If you find our code is useful in your researches, please cite:

    @inproceedings{yangCVPR2016joint,
        Author = {Yang, Jianwei and Parikh, Devi and Batra, Dhruv},
        Title = {Joint Unsupervised Learning of Deep Representations and Image Clusters},
        Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        Year = {2016}
    }

### Dependencies

1. [CUDA](https://developer.nvidia.com/cuda-toolkit). Install CUDA on your PC. I used CUDA 7.5, but it should also work to use new versions.

2. [Visual Studio](https://www.visualstudio.com/downloads/). It is flexible to use various version of VS. I used VS2013 in my experiments.

3. [Third Party](). Caffe depends on several third-party libraries, including hdf5, boost, gflag, opencv, glog, to name a few. [happynear](https://github.com/happynear) has provided the compiled libraties at [Caffe-Windows] (https://github.com/happynear/caffe-windows). Download those libraries and place them in the root folder, then add the ./3rdparty/bin folder to your environment variable PATH. Please ensure that these libraries has the same dependency on CUDA to your project.

### Steps to run the code

1. Open ./buildVS2013/MainBuilder.sln using Visual Studio. Ideally, you will see 11 projects in one solution. Among them, you will mainly use caffelib and caffe_unsupervised to reproduce the results in our paper. However, the projects might crash because of different version of CUDA you are using. In this case, change the CUDA version in vcxproj file of each project.

2. 

### Datasets

We upload six small datasets: COIL-20, USPS, MNIST-test, CMU-PIE, FRGC, UMist. The other large datasets, COIL-100, MNIST-full and YTF can be found in my google drive [here](https://drive.google.com/folderview?id=0B9J-9A2jotGRT25vSDhUWTQxVWs&usp=sharing).

### Compared Approaches

We upload the code for the compared approaches in matlab folder. Please refer to the original paper for details and cite them properly. In this foler, we also attach the evaluation code for two metric: normalized mutual information (NMI) and clustering accuracy (AC).

### Q&A

You are welcome to send message to (jw2yang at vt.edu) if you have any issue on this code.

