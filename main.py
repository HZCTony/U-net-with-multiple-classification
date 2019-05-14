from model2 import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(#rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    #shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(6,'/unet/data/catndog/train/','image','label',data_gen_args,save_to_dir = 'data/catndog/train/aug/')

model = unet()
model_checkpoint = ModelCheckpoint('catndog.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=100,epochs=20,callbacks=[model_checkpoint])

testGene = testGenerator("data/catndog/test")
results = model.predict_generator(testGene,28,verbose=1)
saveResult("data/catndog/predict/",results)
