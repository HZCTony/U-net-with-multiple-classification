# U-net with multiple classification using Keras

This is a modified project from the 2-class [zhixuhao/unet](https://github.com/zhixuhao/unet.git) here. The main purpose of this project is establishing a correct process of colorful classification. Ha ha :)

The orinigal thesis is [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).


# My result

![image](img/pic_modified.png)
loss = 0.2976

accuracy = 0.8786

Actually, the result is not good enough. To improve that, it may need more src data, larger batch size, etc. 

# You have to know what I modification: 
### Data.py

The input data should be 512*512 images for U-net in model2.py. I have the sample images of cats and dogs from internet.  
Besides, I don't use dataPrepare.ipynb so just ingnore it. 


### model2.py

I slightly rectified the strucure of U-net and saved it as model2.py . What I've done is:

* set activation = None in every conv2D and add LeakyReLU after every conv2D. 
* conv10 is the last layer for classification. set activation = "softmax" in conv10
* Adam optimizer with learing rate = 1e-5 (just try it)
* set 'categorical_crossentropy' as loss function 



### Training

The model is trained for 20 epochs, 100 steps per epoch and 6 per batch size.




## How to use

### Dependencies

My dependency:

* Tensorflow : 1.4.0
* Keras >= 1.0
* Python 3.5.2
* cuda 8.0 (for my Nvidia GTX980ti)


(it's optional, but I recommend that):

* docker 18.09.5


### Run main.py

```
python3 main.py
```

## About Keras


Read the documentation [Keras.io](http://keras.io/)

Keras is compatible with: Python 2.7-3.5.
