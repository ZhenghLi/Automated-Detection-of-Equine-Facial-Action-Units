import os
import pandas as pd

cuda_num = 0

class_number = 1
height = 64
width = 64

lr = 0.0001
lr_decay_every_epoch = 15
lr_decay_rate = 0.1
weight_decay = 0.00001

epoch = 50
train_batch_size = 64
val_batch_size = 64
test_batch_size = 64
thresh = 0.5
val_every_epoch = 1

# model = "DRML"
model = "AlexNet"

au = "AU101"
region = "eye"
# region = "lower_face"
val_horse = "A"
test_horse = "B"

data_root = '/Midgard/Data/zhenghli/FACS'
train_info = os.path.join(data_root, 'dataset_info_' + str(au).lower() + '/train_' + val_horse + '_' + test_horse + '_binary_' + str(au).lower() + '_' + region + '_balance.xlsx')
val_info = os.path.join(data_root, 'dataset_info_' + str(au).lower() + '/val_' + val_horse + '_' + test_horse + '_binary_' + str(au).lower() + '_' + region + '_balance.xlsx')
test_info = os.path.join(data_root, 'dataset_info_' + str(au).lower() + '/test_' + val_horse + '_' + test_horse + '_binary_' + str(au).lower() + '_' + region + '_balance.xlsx')

image_dir = os.path.join(data_root, 'Image_' + region +'_pad')