# LDDP: Learning Detection with Diverse Proposals

By Samaneh Azadi, Jiashi Feng, Trevor Darrell at UC Berkeley.

### Introduction: LDDP is proposed to predict a set of diverse and informative proposals with enriched representations which is able to augment object detection architectures. 
LDDP considers both label-level contextual information and spatial layout relationships between object proposals without increasing the number of parameters of the network, and thus improves location and category specifications of final detected bounding boxes substantially during both training and inference schemes.
This implementation is built based on [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn) framework but can be modified for other detection architectures.
For more information on LDDP, please refer to the [arxiv preprint](https://arxiv.org/pdf/1704.03533.pdf) which will be published at CVPR 2017. 

### License
LDDP is released under the MIT License (refer to the LICENSE file for details).

### Citing LDDP 
If you find LDDP useful in your research, please cite:
	@article{azadi2017learning,
	  title={Learning Detection with Diverse Proposals},
	  author={Azadi, Samaneh and Feng, Jiashi and Darrell, Trevor},
	  journal={arXiv preprint arXiv:1704.03533},
	  year={2017}
	} 

Requirements and installation instructions are similar to [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn), but we mention them again for your convenience.
	
### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```
  You can download my [Makefile.config](https://people.eecs.berkeley.edu/~sazadi/LDDP/Makefile.config) for reference.
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`

### Requirements: hardware
Hardware requirements are similar to the those for running [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/blob/96dc9f1dea3087474d6da5a98879072901ee9bf9/README.md#requirements-hardware).

### Installation

1. Clone the LDDP repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/azadis/LDDP.git
  ```

2. We'll call the directory that you cloned LDDP into `LDDP_ROOT`


3. Build the Cython modules
    ```Shell
    cd $LDDP_ROOT/py-faster-rcnn/lib
    make
    ```

4. Build Caffe and pycaffe
    ```Shell
    cd $LDDP_ROOT/py-faster-rcnn/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```
### Installation for training and testing models
1. Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```

2. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	```

3. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

4. Create symlinks for the PASCAL VOC dataset

	```Shell
    cd $LDDP_ROOT/py-faster-rcnn/data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.
5. [Optional] follow similar steps to get PASCAL VOC 2010 and 2012.
6. [Optional] If you want to use COCO, please see the notes [here](https://github.com/rbgirshick/py-faster-rcnn/blob/96dc9f1dea3087474d6da5a98879072901ee9bf9/data/README.md).
7. Follow the next sections to download pre-trained ImageNet models.

### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: ZF and VGG16.

```Shell
cd $LDDP_ROOT/py-faster-rcnn
./data/scripts/fetch_imagenet_models.sh
```
VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but is provided here for your convenience.
ZF was trained at MSRA.

### Usage
To train and test a LDDP end-to-end detection framework:
```Shell
cd $LDDP_ROOT/py-faster-rcnn
./experiments/scripts/LDDP_end2end.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701 TRAIN.SCALES [400,500,600,700]
```

Trained LDDP networks are saved under:

```
output/<experiment directory>/<dataset name>/
```

Test outputs are saved under:

```
output/<experiment directory>/<dataset name>/<network snapshot name>/
```

Semantic Similarity matrices used in the [paper](https://arxiv.org/pdf/1704.03533.pdf) are stored as pickle files at:
```Shell
$LDDP_ROOT/data
```
An example ipython script to generate semantic similarity matrices for PASCAL VOC and COCO data sets is located at:

```Shell
$LDDP_ROOT/tools/Semantic_Similarity.ipynb
```











 
