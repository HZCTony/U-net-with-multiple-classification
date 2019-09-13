from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import sys
from mode.config import *
np.set_printoptions(threshold=sys.maxsize, precision=5, suppress=True)

arg = command_arguments()
#########################configuration########################
cat = [120,0,0]
dog = [0,255,0]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([ cat, dog, Unlabelled])
class_name = [ 'cat', 'dog', 'None']  # You must define by yourself

color = 'grayscale'

num_classes = 3 # include cat, dog and None.
num_of_test_img = arg.img_num

test_img_size = 256 * 256

img_size = (256,256)
###############################################################



def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255.
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        mask[(mask!=0.)&(mask!=255.)&(mask!=128.)] = 0.
        new_mask = np.zeros(mask.shape + (num_class,))
        ########################################################################
        #You should define the value of your labelled gray imgs
        #For example,the imgs in /data/catndog/train/label/cat is labelled white
        #you got to define new_mask[mask == 255, 0] = 1
        #it equals to the one-hot array [1,0,0].
        ########################################################################
        new_mask[mask == 255.,   0] = 1
        new_mask[mask == 128.,   1] = 1
        new_mask[mask == 0.,   2] = 1
        mask = new_mask

    elif(np.max(img) > 1):
        img = img / 255.
        mask = mask /255.
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)



def trainGenerator( batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode = "grayscale",
                    mask_color_mode = "grayscale", image_save_prefix  = "image", mask_save_prefix  = "mask",
                    flag_multi_class = True, num_class = num_classes , save_to_dir = None, target_size = img_size, seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path+"image",
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path+"label",
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    print('classes:',image_generator.class_indices, mask_generator.class_indices)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

### You have to prepare validation data by your own while training
### If you prepared, add validation_data= "your own val path" in fit_generator in main.py
def validationGenerator( batch_size, val_path, image_folder, mask_folder, aug_dict, image_color_mode = "grayscale",
                         mask_color_mode = "grayscale", image_save_prefix  = "image", mask_save_prefix  = "mask",
                         flag_multi_class = True, num_class = num_classes , save_to_dir = None, target_size = img_size, seed = 1):

    image_datagen = ImageDataGenerator()
    mask_datagen  = ImageDataGenerator()
    val_image_generator = image_datagen.flow_from_directory(
        val_path+"image",
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        )
    val_mask_generator = mask_datagen.flow_from_directory(
        val_path+"label",
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        )
    val_generator = zip(val_image_generator, val_mask_generator)
    for (img,mask) in val_generator:
        img ,mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img,mask)

def testGenerator(test_path,num_image = num_of_test_img, target_size = img_size, flag_multi_class=True, as_gray=True):
    for i in range(num_image):
        i = i + 1
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        #img = img / 255.
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


### You have to prepare test data by your own after training
def testGenerator_for_evaluation(test_path, mask_path, num_image=num_of_test_img, num_class=num_classes ,target_size=(256,256), flag_multi_class = True, as_gray = True):
    for i in range(num_image):
        i = i + 1
        # read test images
        img = io.imread(os.path.join(test_path,"%d.png"%i), as_gray = as_gray)
        img = trans.resize(img, target_size)
        img = np.reshape(img,img.shape+(1,)) if (flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        # read mask images
        mask = io.imread(os.path.join(mask_path,"%d.png"%i), as_gray = as_gray)
        mask = trans.resize(mask, target_size)
        mask = np.expand_dims(mask,0)
        mask = np.expand_dims(mask,-1)
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        ###### filter noise points not related to the classes ######
        mask[(mask!=0.)&(mask!=255.)&(mask!=128.)] = 0.
        new_mask = np.zeros(mask.shape + (num_class,))
        new_mask[(mask == 255.),   0] = 1
        new_mask[(mask == 128.),   1] = 1
        new_mask[(mask ==   0.),   2] = 1
        mask = new_mask
        yield (img,mask)



### draw imgs in labelVisualize and save results in saveResult
def labelVisualize(num_class,  color_dict, img):
    img_out = np.zeros(img[:,:,0].shape + (3,))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            index_of_class = np.argmax(img[i,j])
            img_out[i,j] = color_dict[index_of_class]
    return img_out

def saveResult(save_path,npyfile,flag_multi_class = True,num_class = num_classes ):
    count = 1
    for i,item in enumerate(npyfile):
        if flag_multi_class:
            img = labelVisualize(num_class,COLOR_DICT,item)
            img = img.astype(np.uint8)
            io.imsave(os.path.join(save_path,"%d.png"%count),img)
        else:
            img=item[:,:,0]
            print(np.max(img),np.min(img))
            img[img>0.5]=1
            img[img<=0.5]=0
            print(np.max(img),np.min(img))
            img = img * 255.
            io.imsave(os.path.join(save_path,"%d.png"%count),img)
        count += 1

