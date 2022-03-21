## Requirements and Dependency
* Install [PyTorch](http://torch.ch) with CUDA (for GPU). (Experiments are validated on python 3.8.11 and pytorch 1.7.0)
* (For visualization if needed), install the dependency [visdom](https://github.com/facebookresearch/visdom) by:
```Bash
pip install visdom
```



## Experiments

Here, we provide the code for reproducing the main experiments on ImageNet datasets. 

#### 1. Prepare the dataset:

Download the ImageNet-1K datasets, and put it in the dir: `./data/imageNet/`   or you can specify your datapath by changing `--dataset-root=/your-data-path`



#### 2. Run scripts of experiments: 

We provide the scripts in `./experiments/`, including the experiments on the ResNet, ResNeXt,  Mobilenet-V2  and ShuffleNet-V2  . 



