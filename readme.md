## Requirements and Dependency
* Install [PyTorch](http://torch.ch) with CUDA (for GPU). (Experiments are validated on python 3.8.11 and pytorch 1.7.0)
* (For visualization if needed), install the dependency [visdom](https://github.com/facebookresearch/visdom) by:
```Bash
pip install visdom
```
===========================================================================

Note: I noted the pre-trained models are requested, here is the link of [ResNet-XBNBlock-standard_train](https://drive.google.com/file/d/1xqGsCDD4Y_rv6PjTtACAs8D918fjUT55/view?usp=sharing) and [ResNet-XBNBlock-advanced_train](https://drive.google.com/file/d/1pHRhjc67M5wki5fWy44SG5lj06q71gat/view?usp=sharing). I will upload our pre-trained ResNeXt model, if I can access the machine in my lab. (work at home due to COVID-19)

===========================================================================




## Experiments

Here, we provide the code for reproducing the main experiments on ImageNet datasets. 

#### 1. Prepare the dataset:

Download the ImageNet-1K datasets, and put it in the dir: `./data/imageNet/`   or you can specify your datapath by changing `--dataset-root=/your-data-path`



#### 2. Run scripts of experiments: 

We provide the scripts in `./experiments/`, including the experiments on the ResNet, ResNeXt,  Mobilenet-V2  and ShuffleNet-V2  . 



