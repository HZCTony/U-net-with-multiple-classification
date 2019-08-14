import csv
from mode.config import *

def csv_create(csvfilename,orders,batch_size,steps_per_epoch,epochs, learning_rate,learning_decay_rate, rotation_range):
    with open(csvfilename, 'w') as csvfile:
        fieldnames = ['orders','batch_size','steps_per_epoch','epochs','learning_rate','learning_decay_rate','rotation_range']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'orders':orders,'batch_size': batch_size, 'steps_per_epoch': steps_per_epoch, 'epochs':epochs, 'learning_rate':learning_rate,'learning_decay_rate':learning_decay_rate, 'rotation_range': rotation_range})


def csv_append(csvfilename,orders,batch_size,steps_per_epoch,epochs,learning_rate,learning_decay_rate, rotation_range):
    with open(csvfilename,'a') as fd:
        fieldnames = ['orders','batch_size','steps_per_epoch','epochs','learning_rate','learning_decay_rate','rotation_range']
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        writer.writerow({'orders':orders,'batch_size': batch_size, 'steps_per_epoch': steps_per_epoch, 'epochs':epochs ,'learning_rate':learning_rate,'learning_decay_rate':learning_decay_rate, 'rotation_range': rotation_range})
