#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:03:38 2019

@author: lee
"""

import numpy as np
import keras
from keras import regularizers
from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten
from time import time
import argparse
import json
import sys
from keras.models import Sequential, Model, load_model, save_model
import heapq

"""
Argument
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--mlp_pretrain', nargs='?', default='../dataset/pretrain/amazon_MLP_[16,8]_1569250599.h5',
                        help='pretrain path')
    
    parser.add_argument('--dataset', nargs='?', default='amazon',
                        help='dataset')
    
    parser.add_argument('--topk', type=int, default=10, 
                        help='topk')
    
    parser.add_argument('--user', type=int, default=0,
                        help='user index')
    
    parser.add_argument('--layers', nargs='?', default='[16,8]', 
                        help="embedding size, layer[0]/2")
    
    parser.add_argument('--reg_layers', nargs='?', default='[0,0]',
                        help="regularization")
    
    return parser.parse_args()



def get_model(num_users, num_items, layers = [32,16], reg_layers=[0,0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)
    
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input') 
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input') 
    
    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = 'user_embedding',
                                   embeddings_initializer = 'random_normal', 
                                   embeddings_regularizer = regularizers.l2(reg_layers[0]), 
                                   input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'item_embedding',
                                  embeddings_initializer = 'random_normal', 
                                  embeddings_regularizer = regularizers.l2(reg_layers[0]), 
                                  input_length=1)

    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))

    vector = keras.layers.concatenate([user_latent, item_latent])

    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer= regularizers.l2(reg_layers[idx]), kernel_initializer = 'he_normal',activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)

    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(vector)
    
    model = Model(inputs=[user_input, item_input],
                  outputs=prediction)

    return model


# if user already purchsed items, extract!
    
def predict_item_score(user):
    map_item_score = {}
    user_index = np.full(itemnum, user, dtype = 'int32')
    items_index = np.arange(0, itemnum, 1, np.int) 
    predictions = model.predict([user_index, items_index],batch_size=itemnum, verbose=0)
    
    for i in range(len(items_index)):
        map_item_score[i] = predictions[i]
    
    # top-k 
    rank_cosmetic_list = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
    
    # asin name
    real_rank_cosmetic_list=[]
    for i in rank_cosmetic_list:
        real_rank_cosmetic_list.append(product_name[str(i)])
    
    return real_rank_cosmetic_list


if __name__ == '__main__':
    args = parse_args()
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    mlp_pretrain = args.mlp_pretrain
    user = args.user
    topK = args.topk
    
    evaluation_threads = 1


    """ Load data """
    
    t1 = time()
    dataset = open('../dataset/amazon_raw2inner_dict.json',encoding='utf-8-sig').read()
    js=json.loads(dataset)
    usernum,itemnum,product_name = len(js['user_dict']), len(js['product_dict']), js['product_dict']
    print("Load dict done [%.1f s]. #user=%d, #item=%d"
          % (time() - t1, usernum, itemnum))
    

    """ Load pretrain model"""
    
    model = get_model(usernum, itemnum, layers, reg_layers)
    model.load_weights(mlp_pretrain)
    print("Load pretrained mlp(%s) models done. " %(mlp_pretrain))

    
    print('\t')
    print("User %s's top-%s recommendation" %(user,topK))
    print('\t')
    for i in range(topK):
        print("%s. %s" %(i+1,predict_item_score(user)[i]))

    
    
