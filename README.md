# U-net with multiple classification using Keras

This is a modified project from the 2-class [zhixuhao/unet](https://github.com/zhixuhao/unet.git) here. The main purpose of this project is establishing a correct process of colorful classification. Ha ha :)

The orinigal thesis is [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).


# My result

![image](img/pic_modified.png)
loss = 0.2976

accuracy = 0.8786

Actually, the result is not good enough. To improve that, it may need more src data, larger batch size, etc. 


### You have to know:
The structure of this project is:

/data/catndog : my sample collection of cat and dog with the [required catalog](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d). 
* data.py : prepare the related images you want to train and predict.
* model2.py : define the U-net structure
* main.py : run the program


### data.py

The original size of images is 512x512. However, they'll be resized to 256x256 for U-net in model2.py. I collect the sample images of cats and dogs from internet. You can find them in /data/catndog/ . Besides, I don't use dataPrepare.ipynb so just ingnore it. My modifications are summerized below:

* in def trainGenerator(), at first, comment "classes". Second, set the target directories as train_path+"image" in image_datagen.flow_from_directory() and train_path+"label" in mask_datagen.flow_from_directory(). Keras will detect the classes from your training data automatically.

* Let all the "flag_multi_class = True"

* in def adjustdata(), reshape the mask with (batch_size, width, height, classes). Every channel in fourth dimemsion corresponds to a certain class with one-hot format. The code here only written for 3 classes(cat, dog, background).

* in labelVsiualize(), pick up the max value in one-hot vector and draw the corresponding colors to every gnenerated all-zero array. You can define the color in clolor_dict.


### model2.py

* Set activation = None in every conv2D and add LeakyReLU after every conv2D. It helps prevent the training process from not updating weights. 
* Set activation = "softmax" in last layer, conv10, for classification. 
* Adam optimizer with learing rate = 1e-5 (I just try it)
* Set 'categorical_crossentropy' as loss function rather than 'binary_crossentropy'



### Training

The model is trained for 20 epochs, 100 steps per epoch and 6 per batch size. You can test more hyperparameters and let me know something amazing from you.



### My dependencies

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



