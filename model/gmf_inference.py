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
    parser = argparse.ArgumentParser(description="Run GMF.")

    parser.add_argument('--dataset', nargs='?', default='amazon',
                        help='dataset')
    
    parser.add_argument('--GMF_pretrain', nargs='?', default='../dataset/pretrain/amazon_GMF_4_1569244262.h5',
                        help='pretrain model path')

    parser.add_argument('--num_factors', type=int, default=4,
                        help='user,item embedding size')

    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="regularization")

    parser.add_argument('--user', type=int, default= 0,
                        help='user index')
    
    parser.add_argument('--topk', type=int, default= 10,
                    help='top k')
    
    return parser.parse_args()



def get_model(num_users, num_items, latent_dim, regs=[0, 0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=latent_dim, name='user_embedding',
                                  embeddings_initializer='random_normal', embeddings_regularizer=regularizers.l2(regs[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=latent_dim, name='item_embedding',
                                  embeddings_initializer='random_normal', embeddings_regularizer=regularizers.l2(regs[1]), input_length=1)

    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))    

    predict_vector = keras.layers.multiply([user_latent,item_latent])

    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(predict_vector)

    model = Model(inputs=[user_input, item_input],
                  outputs=prediction)

    return model


""" 유저별 사용자 예측 """

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
    num_factors = args.num_factors
    regs = eval(args.regs)
    gmf_pretrain = args.GMF_pretrain
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
    
    ### dict 거꾸로 바꿔서 다시저장하자. 그리고 삭제
    #product_name={}
    #for k,v in product_dict.items():
    #    product_name[v]=int(k)
    ###
    
    """ Load pretrain model"""
    
    model = get_model(usernum, itemnum, num_factors, regs)
    model.load_weights(gmf_pretrain)
    print("Load pretrained gmf (%s) models done. " %(gmf_pretrain))

    
    print('\t')
    print("User %s's top-%s recommendation" %(user,topK))
    print('\t')
    for i in range(topK):
        print("%s. %s" %(i+1,predict_item_score(user)[i]))

    
    
