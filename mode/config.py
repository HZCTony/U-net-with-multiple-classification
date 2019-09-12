#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse


def command_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenum','-n', default=1 ,type=int,help='file numbers')
    #training
    parser.add_argument('--learning_rate','-lr', default=0.00005, type=float, help='learning_rate')
    parser.add_argument('--learning_decay_rate','-ldr', default=0.00001, type=float,help='learning_decay_rate')
    parser.add_argument('--batchsize','-b', default=5, type=int, help='batch size')
    parser.add_argument('--steps_per_epoch','-s', default=70,type=int, help='steps_per_epoch')
    parser.add_argument('--epochs','-e', default=70, type=int, help='epochs')
    parser.add_argument('--train_path','-tp', default='./data/catndog/train/', help='train_path')
    parser.add_argument('--train_img_folder','-tif', default='image', help='train_img_folder')
    parser.add_argument('--train_label_folder','-tlf', default='label', help='train_label_folder')
    parser.add_argument('--test_img_path','-tip',default="./data/catndog/test/", help='test_img_path')    
    parser.add_argument('--img_num','-tm', default=27, type=int, help='test img num')
    
    save_result_folder = "./data/catndog/results/%s_mrsu/" % parser.parse_args().filenum
    model_name = '%s_mrsu_lichi.hdf5' % parser.parse_args().filenum
    plt_save_name = '%s_fig_mrsu.png' % parser.parse_args().filenum
    val_plt_name = '%s_val.png' % parser.parse_args().filenum

    parser.add_argument('--save_result_folder', '-save', default=save_result_folder, help='save_result_folder')
    parser.add_argument('--csvfilename', '-csv', default="training_record.csv", help='csv file name')
    parser.add_argument('--model_name', '-m', default=model_name, help='model_name')
    parser.add_argument('--plt_save_name', '-plt', default=plt_save_name, help='plt_save_name')
    parser.add_argument('--val_plt_name','-vplt', default=val_plt_name, help='val_plt_name')
    #data aug params
    parser.add_argument('--rotation_range', '-rot', default=180, help='rotation_range')
    parser.add_argument('--width_shift_range', '-wid', default=0.9, help='width_shift_range')
    parser.add_argument('--height_shift_range', '-hei', default=0.9, help='height_shift_range')
    parser.add_argument('--zoom_range', '-zoo', default=0, help='zoom_range')
    parser.add_argument('--vertical_flip', '-ver', default=True, help='vertical_flip')
    parser.add_argument('--horizontal_flip', '-hor', default=True, help='horizontal_flip')
    parser.add_argument('--fill_mode', '-fil', default='constant', help='fill_mode')
    

    return parser.parse_args()

if __name__ == "__main__":
    arg = command_arguments()
    print(arg)
