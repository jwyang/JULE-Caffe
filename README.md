# JULE-Caffe
Caffe code for our JULE paper "Joint unsupervised learning of deep representations and image clusters". The Torch version code can be found [here](https://github.com/jwyang/JULE-Torch). 

### NOTE

**I have not yet finished cleaning up this code since I have to do it remotely. But it will be finished in several days. Once it is finished, this NOTE will be gone.**

### Overview

This project is a Caffe implementation for our CVPR 2016 [paper](https://arxiv.org/abs/1604.03628), which performs jointly unsupervised learning of deep CNN and image clusters.

### Acknowledgement

A great thanks to [happynear](https://github.com/happynear) for providing an awesome windows version [Caffe](https://github.com/happynear/caffe-windows). It is based on his code that we developed our JULE algorithm.

### License

This code is released under the MIT License (refer to the LICENSE file for details).

### Citation
If you find our code is useful in your researches, please consider citing:

    @inproceedings{yangCVPR2016joint,
        Author = {Yang, Jianwei and Parikh, Devi and Batra, Dhruv},
        Title = {Joint Unsupervised Learning of Deep Representations and Image Clusters},
        Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        Year = {2016}
    }

### Dependencies

### Train model

### Datasets

We upload six small datasets: COIL-20, USPS, MNIST-test, CMU-PIE, FRGC, UMist. The other large datasets, COIL-100, MNIST-full and YTF can be found in my google drive [here](https://drive.google.com/folderview?id=0B9J-9A2jotGRT25vSDhUWTQxVWs&usp=sharing).

### Compared Approaches

We upload the code for the compared approaches in matlab folder. Please refer to the original paper for details and cite them properly. In this foler, we also attach the evaluation code for two metric: normalized mutual information (NMI) and clustering accuracy (AC).

### Q&A

You are welcome to send message to (jw2yang at vt.edu) if you have any issue on this code.

