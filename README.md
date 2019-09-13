# Unet : multiple classification using Keras

This is a modified project from the two-class(cell and background) [zhixuhao/unet](https://github.com/zhixuhao/unet.git) here. The main purpose of this project is establishing a process of multiple classification. Here are 3 classes, dog, cat and background and I open the labelled images. Try it!

--------------------------------------------------------------------------------
### 2019/09/13 update : Quick Start

I simplified my code and now make training much easier.
Once you want to run training, you can just pass some parameters in command line like below:
```
python3 main.py -n 001 -lr 0.00004 -ldr 0.000008 -b 16 -s 60 -e 80
```

-n   = A number helps save different .h5 and directories of infered images.  
-lr  = learning rate  
-ldr = learning decay rate  
-b   = batch size  
-s   = steps  
-e   = epochs  
(check more params in mode/config.py)

After you build up your own dev environment, you can run with the command immediately.

--------------------------------------------------------------------------------


### You have to know:
The structure of this project is:

/data/catndog : my sample collection of cat and dog with the [required catalog](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d). 

All you have see are defined as below:
* data.py : prepare the related images you want to train and predict.
* model.py : define the U-net structure
* main.py : run the program


### data.py

My original size of images is 512x512. However, they'll be resized to 256x256 for U-net architecture defined in model.py. I collect the sample images of cats and dogs from internet. You can find my sample data in /data/catndog/. However, /catndog/ just show how to put your data. You can prepare your data by your own.

My modifications are summerized below:

* in def trainGenerator(), at first, comment "classes". Second, set the target directories as train_path+"image" in image_datagen.flow_from_directory() and train_path+"label" in mask_datagen.flow_from_directory(). Keras will detect the classes from your training data automatically.

* Let all the "flag_multi_class = True"

* in def adjustdata(), reshape the mask with (batch_size, width, height, classes). Every channel in fourth dimemsion corresponds to a certain class with one-hot format. This repo only written for 3 classes(cat, dog, background).

* in labelVsiualize(), pick up the max value in one-hot vector and draw the corresponding colors to every gnenerated all-zero array. You can define the color in clolor_dict.

### model.py

All the U-net architecture is defined in model.py .


### main.py

Training and test steps are defined in main.py .


### My dependencies

* Tensorflow : 1.4.0
* Keras >= 1.0
* Python 3.5.2
* cuda 8.0 (for my Nvidia GTX980ti)


(it's optional, but I recommend that):

* docker 18.09.5



If there is any other suggestion, do not hesitate to tell me.





The orinigal thesis：[U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/). 

