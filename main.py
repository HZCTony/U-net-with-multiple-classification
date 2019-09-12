import os
import os.path
from model import *
from data import *
from keras.models import load_model
from keras.callbacks import History
import tensorflow as tf
import matplotlib.pyplot as plt 
from keras import backend as K
from mode.config import *
from csvrecord import * 
from pathlib import Path
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

arg = command_arguments()

batch_size = arg.batchsize
train_path = arg.train_path
train_img_folder = arg.train_img_folder
train_label_folder = arg.train_label_folder
test_img_path = arg.test_img_path
steps_per_epoch = arg.steps_per_epoch
epochs = arg.epochs
save_result_folder = arg.save_result_folder
csvfilename = arg.csvfilename
model_name = arg.model_name
plt_save_name = arg.plt_save_name
val_plt_name = arg.val_plt_name
img_num = arg.img_num
filenum = arg.filenum

#augs 

rotation_range = arg.rotation_range
width_shift_range = arg.width_shift_range
height_shift_range = arg.height_shift_range
zoom_range = arg.zoom_range
horizontal_flip = arg.horizontal_flip
fill_mode = arg.fill_mode

data_gen_args = dict(
                    featurewise_center=True,
                    featurewise_std_normalization=True,
                    rotation_range=rotation_range,
                    width_shift_range=width_shift_range,
                    height_shift_range=height_shift_range,
                    #shear_range=0.05,
                    #zoom_range=zoom_range,
                    horizontal_flip=horizontal_flip,
                    fill_mode=fill_mode,
                    cval=0)



def show_train_history(train_history, train, loss, plt_save_name=plt_save_name):
    plt.plot(train_history.history['acc'])
    plt.plot(train_history.history['loss'])
    plt.title('Train hist')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['acc','loss'], loc='upper left')
    plt.savefig(plt_save_name)

myGene = trainGenerator(batch_size, train_path, train_img_folder, train_label_folder, data_gen_args, save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint(model_name, monitor='loss',verbose=1, save_best_only=True)
training = model.fit_generator(myGene, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_steps=10, callbacks=[model_checkpoint])
show_train_history(training, 'acc', 'loss')
model = load_model(model_name)

testGene = testGenerator(test_img_path)
#testGene_for_eval = testGenerator_for_evaluation(test_img_path)
results = model.predict_generator(testGene, img_num, verbose=1)
#loss, acc = model.evaluate_generator(testGene_for_eval, steps=img_num, verbose=1)
#print("test loss:",loss,"  test accuracy:", acc)

if not os.path.exists(save_result_folder):
    os.makedirs(save_result_folder)

saveResult( save_result_folder, results)


if (os.path.isfile(csvfilename)!=True):
    csv_create(csvfilename, filenum, batch_size, steps_per_epoch, epochs, learning_rate, learning_decay_rate, rotation_range)
else:
    csv_append(csvfilename, filenum, batch_size, steps_per_epoch, epochs, learning_rate, learning_decay_rate, rotation_range)



K.clear_session()
