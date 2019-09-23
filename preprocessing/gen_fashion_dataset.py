# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 06:06:17 2019

@author: lee
"""

import numpy as np
import pandas as pd
import os,io,sys,glob
from time import time
from tqdm import tqdm_notebook,trange,tqdm
import h5py
from collections import Counter
from PIL import Image

from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input

##########################
# Load Dataset
##########################
t1=time()
dataset = np.load('../dataset/AmazonFashion6ImgPartitioned.npy',encoding='bytes',allow_pickle=True)
train, val, test, meta, usernum, itemnum = dataset
print('data load complete[%.2fs]'%(time()-t1))

##########################
# Save Item images
##########################
os.mkdir('../dataset/meta_img')

for idx in trange(len(meta)):
    try:
        c=io.BytesIO(meta[idx][b'imgs'])
        im=Image.open(c)
        im.save('../dataset/meta_img/%s.jpg'%idx)
        
    except OSError: # png, but useless
        os.remove(('../dataset/meta_img/%s.jpg'%idx))
        
##########################
# Save image features
##########################        
def feature_extractor():
    t1=time()
    avg_pool_features={}
    base_model = ResNet50(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    print('model load complete[%0.2fs]'%(time()-t1))
    
    # generate image batch
    path = '../dataset/meta_img/*'
    file_list = glob.glob(path)
    file_list= [i for i in file_list if i.endswith('jpg')]
    
    for index, img_path in tqdm_notebook(enumerate(file_list)):
        meta_index = img_path.split('\\')[1].split('.')[0] # meta_index는 이미지파일 번호로 딕셔너리 key용으로 사용
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        avg_pool_features[meta_index] = model.predict(x)[0]

    return avg_pool_features

avg_pool_features = feature_extractor()


file = '../dataset/amazonfashion6_imgfeature.hdf5'

with h5py.File(file, 'w') as f:
    f.create_dataset('imgs', (len(meta),2048,), dtype='float32')
    img_set=f['imgs']
    
    for n,i in tqdm_notebook(avg_pool_features.items()):
        img_set[int(n)]= i

##########################################
# Convert deepfashion type to keras-yolo3
##########################################
df=pd.read_csv('../dataset/In-shop_Clothes_Retrieval_Benchmark/Anno/list_bbox_inshop.txt',header=None,sep='\s+',
               names=['path','label','pose','x_1','y_1','x_2','y_2'])


train=[]
for index,f,class_id,pose,x_1,y_1,x_2,y_2 in df.itertuples():
    train.append('../dataset/In-shop_Clothes_Retrieval_Benchmark/%s %s,%s,%s,%s,%s'%(f,x_1,y_1,x_2,y_2,int(class_id)+79))

# Save train.txt
f = open("../keras-yolo3-detection/train.txt", 'w')
for i in train:
    data = i+'\n'
    f.write(data)
f.close()
    