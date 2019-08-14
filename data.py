from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import sys
<<<<<<< HEAD
from mode.config import *
np.set_printoptions(threshold=sys.maxsize, precision=5, suppress=True)

arg = command_arguments()
#########################configuration#########################
adult = [120,0,0]
egg = [0,255,0]
kidtwo = [0,0,200]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([ adult, egg, kidtwo, Unlabelled])
class_name = [ 'className1', 'className2', 'className3', 'None']  # You must define by yourself

color = 'grayscale'

num_classes = 4
num_of_test_img = arg.img_num

probability_threshold = 0.1
portion_threshold = 0.03

test_img_size = 256 * 256

img_size = (256,256)
###############################################################

=======
np.set_printoptions(threshold=sys.maxsize, precision=5, suppress=True)

cat = [128,0,0]
dog = [0,128,0]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([ cat, dog, Unlabelled])
>>>>>>> b9d9b86016be6d998071ccdfe8bf5924e7894700


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        #print(type(img))
        img = img / 255.
<<<<<<< HEAD
        #print(mask.shape)
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        mask[(mask!=0.)&(mask!=255.)&(mask!=120.)&(mask!=140.)] = 0.
        #print(mask.shape)
        new_mask = np.zeros(mask.shape + (num_class,))
        new_mask[mask == 120.  , 0] = 1
        new_mask[mask == 255.,   1] = 1
        #new_mask[mask == 180.,   2] = 1
        new_mask[mask == 140.,   2] = 1
        new_mask[mask == 0.,     3] = 1
=======
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        mask[(mask!=0.)&(mask!=255.)&(mask!=128.)] = 0.
        new_mask = np.zeros(mask.shape + (num_class,))
        new_mask[mask == 255.  , 0] = 1
        new_mask[mask == 128.,   1] = 1
        new_mask[mask == 0.,     2] = 1
>>>>>>> b9d9b86016be6d998071ccdfe8bf5924e7894700
        mask = new_mask

    elif(np.max(img) > 1):
        img = img / 255.
        mask = mask /255.
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)



def trainGenerator( batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode = "grayscale",
                    mask_color_mode = "grayscale", image_save_prefix  = "image", mask_save_prefix  = "mask",
<<<<<<< HEAD
                    flag_multi_class = True, num_class = num_classes , save_to_dir = None, target_size = img_size, seed = 1):
=======
                    flag_multi_class = True, num_class = 3, save_to_dir = None, target_size = (256,256), seed = 1):
>>>>>>> b9d9b86016be6d998071ccdfe8bf5924e7894700
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path+"image",
        #classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path+"label",
        #classes = [mask_folder],
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



<<<<<<< HEAD
def validationGenerator( batch_size, val_path, image_folder, mask_folder, aug_dict, image_color_mode = "grayscale",
                         mask_color_mode = "grayscale", image_save_prefix  = "image", mask_save_prefix  = "mask",
                         flag_multi_class = True, num_class = num_classes , save_to_dir = None, target_size = img_size, seed = 1):

    image_datagen = ImageDataGenerator()
    mask_datagen  = ImageDataGenerator()
    val_image_generator = image_datagen.flow_from_directory(
        val_path+"image",
        #classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        )
    val_mask_generator = mask_datagen.flow_from_directory(
        val_path+"label",
        #classes = [mask_folder],
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
=======
def testGenerator(test_path,num_image = 28,target_size = (256,256),flag_multi_class = True,as_gray = True):
>>>>>>> b9d9b86016be6d998071ccdfe8bf5924e7894700
    for i in range(num_image):
        i = i + 1
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        #img = img / 255.
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        #print("shape of test img:",img)
        yield img

<<<<<<< HEAD
def testGenerator_for_evaluation(test_path, mask_path, num_image=num_of_test_img, num_class=num_classes ,target_size=(256,256), flag_multi_class = True, as_gray = True):
    for i in range(num_image):
        i = i + 1
        # read test images
        img = io.imread(os.path.join(test_path,"%d.png"%i), as_gray = as_gray)
        img = trans.resize(img, target_size)
        img = np.reshape(img,img.shape+(1,)) if (flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        #print("img shape:",img.shape)
        # read mask images
        mask = io.imread(os.path.join(mask_path,"%d.png"%i), as_gray = as_gray)
        mask = trans.resize(mask, target_size)
        #print("mask shape:",mask.shape)
        mask = np.expand_dims(mask,0)
        mask = np.expand_dims(mask,-1)
        #print(mask)
        #print(mask.shape)
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
          ###### filter noise points not related to the classes ######
        mask[(mask!=0.)&(mask!=255.)&(mask!=120.)&(mask!=140.)] = 0.
        new_mask = np.zeros(mask.shape + (num_class,))
        new_mask[(mask == 120.),   0] = 1
        new_mask[(mask == 255.),   1] = 1
        #new_mask[(mask == 180.),   2] = 1
        new_mask[(mask == 140.),   2] = 1
        new_mask[(mask == 0.),     3] = 1
        mask = new_mask
               
        yield (img,mask)


def labelVisualize(num_class,  color_dict, img):
=======

def geneTrainNpy(image_path,mask_path,flag_multi_class = True,num_class = 3,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class, color_dict, img):
>>>>>>> b9d9b86016be6d998071ccdfe8bf5924e7894700
    img_out = np.zeros(img[:,:,0].shape + (3,))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            index_of_class = np.argmax(img[i,j])
<<<<<<< HEAD
            if img[i,j,index_of_class] < probability_threshold:
               index_of_class = -1
            img_out[i,j] = color_dict[index_of_class]
    return img_out

def recognition_text(num_class, img):
    counting = np.zeros(num_class, dtype=int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #temp = np.array([img[i,j,0], img[i,j,1], img[i,j,2]])
            temp = np.array(img[i,j])
            index_of_class = np.argmax(temp)
            if temp[index_of_class] < probability_threshold:
               index_of_class = -1
            counting[index_of_class] += 1
    
    predicted_class = np.argmax([counting[0], counting[1], counting[2]])
    portion = counting[predicted_class]/(test_img_size)
    if portion < portion_threshold:
       predicted_class = -1       

    pred_dict = {
                 'class': class_name[predicted_class],
                 'img_size' : img_size,
                 'portion': portion,
                 'pixel count': counting,
                 'probability threshold': probability_threshold,
                 'portion threshold': portion_threshold
                }
    print(pred_dict)

    return pred_dict

def saveResult(save_path,npyfile,flag_multi_class = True,num_class = num_classes ):
    count = 1
    for i,item in enumerate(npyfile):
        if flag_multi_class:
            pred_result = recognition_text(num_class, item)
            img = labelVisualize(num_class,COLOR_DICT,item)
            img = img.astype(np.uint8)
            io.imsave(os.path.join(save_path,"%d_%s.png"%(count,pred_result['class'])),img)
=======
            img_out[i,j] = color_dict[index_of_class]
    return img_out
          


#def labelVisualize(num_class,color_dict,img):
#    #for i in range(10):
#    #  for j in range(10):
#    #      print(np.argmax(img[i,j,:]))
#    img = img[:,:,0] if len(img.shape) == 3 else img
#    img_out = np.zeros(img.shape + (3,))
#    img = img * 255.
#    
#    for i in range(num_class):
#        img_out[img == i,:] = color_dict[i]
#        #print(img_out,i)
#    return img_out / 255.

def saveResult(save_path,npyfile,flag_multi_class = True,num_class = 3):
    for i,item in enumerate(npyfile):
        if flag_multi_class:
            img = labelVisualize(num_class,COLOR_DICT,item)
            img = img.astype(np.uint8)
            io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
>>>>>>> b9d9b86016be6d998071ccdfe8bf5924e7894700
        else:
            img=item[:,:,0]
            print(np.max(img),np.min(img))
            img[img>0.5]=1
            img[img<=0.5]=0
            print(np.max(img),np.min(img))
            img = img * 255.
<<<<<<< HEAD
            io.imsave(os.path.join(save_path,"%d_%s.png"%(count,pred_result['class'])),img)
        count += 1

=======
            io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)


#def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
#    for i,item in enumerate(npyfile):
#        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
#        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
>>>>>>> b9d9b86016be6d998071ccdfe8bf5924e7894700
