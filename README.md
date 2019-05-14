# U-net with multiple classification using Keras

This is a modified project from the 2-class [zhixuhao/unet](https://github.com/zhixuhao/unet.git) here. 

The orinigal thesis is [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).


# My result

![image]("img/pic_modified.png")

# You have to know : 
### Data.py

The input data should be 512*512 images for U-net in model.py. I have the sample images of cats and dogs from internet.  
Besides, I don't use dataPrepare.ipynb so just ingnore it.


### Model



### Training

The model is trained for 20 epochs, 100 steps per epoch and 6 per batch size.


---

## How to use

### Dependencies

My dependency:

* Tensorflow : 1.4.0
* Keras >= 1.0
* Python 3.5.2
* cuda 8.0 (for my Nvidia GTX980ti)


(it's optional, but I recommend that):
docker 18.09.5


### Run main.py



## About Keras


Read the documentation [Keras.io](http://keras.io/)

Keras is compatible with: Python 2.7-3.5.
