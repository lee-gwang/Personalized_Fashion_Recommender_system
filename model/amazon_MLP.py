#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 04:32:28 2019

@author: lee
"""

import numpy as np
import keras
from keras import backend as K
from keras import initializers
from keras import regularizers
from keras.models import Sequential, Model

from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten, Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from fast_evaluation import evaluate_model
from time import time

import argparse
import multiprocessing as mp
import json


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='',
                        help='dataset path')
    
    parser.add_argument('--dataset', nargs='?', default='amazon',
                        help='dataset')
    
    parser.add_argument('--epochs', type=int, default=10, 
                        help='epochs')
    
    parser.add_argument('--batch_size', type=int, default=2**14,
                        help='batch size')
    
    parser.add_argument('--layers', nargs='?', default='[16,8]', 
                        help="embedding size, layer[0]/2")
    
    parser.add_argument('--reg_layers', nargs='?', default='[0,0]',
                        help="regularization")
    
    parser.add_argument('--num_neg', type=int, default=4,
                        help='number of negative instances')
    
    parser.add_argument('--lr', type=float, default=0.03,
                        help='Learning rate.')
    
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='optimizers : adagrad, adam, rmsprop, sgd')
    
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    
    parser.add_argument('--out', type=int, default=1,
                        help='if 1, save train model')
    
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


def get_train_instances(train,num_negatives):
    user_input, item_input, labels = [], [], []
    for u in train:
        # positive instance
        for i in train[u]:
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
        # negative instances
            for t in range(num_negatives):
                j = np.random.randint(itemnum)
                while j== i:
                    j = np.random.randint(itemnum)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose
    
    topK = 10
    print("MLP arguments: %s" % (args))
    model_out_file = '../dataset/pretrain/%s_MLP_%s_%d.h5' %(args.dataset, args.layers, time())


    """ Load data """
    
    t1 = time()
    dataset = open('../dataset/amazon_clothing_fast_implicit.json',encoding='utf-8-sig').read()
    js=json.loads(dataset)
    train, valRatings, valNegatives,val_full_user, val_sum_Ratings, flatten_val_sum_Ratings, usernum,itemnum = js['train'],js['valRatings'],js['valNegatives'],js['val_full_user'],js['val_sum_Ratings'],js['flatten_val_sum_Ratings'], js['usernum'],js['itemnum']
    print("Load data done [%.1f s]. #user=%d, #item=%d"
          % (time() - t1, usernum, itemnum))
    
      
    model = get_model(usernum,itemnum, layers, reg_layers)
    
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')    


    
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, valRatings, val_full_user,flatten_val_sum_Ratings, val_sum_Ratings, topK)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f s]' %(hr, ndcg, time()-t1))
    
       
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        user_input, item_input, labels = get_train_instances(train,num_negatives)
    
        hist = model.fit([np.array(user_input), np.array(item_input)], np.array(labels), 
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)

        t2 = time()

        # Evaluation
        if epoch %verbose == 0:
            (hits, ndcgs) = evaluate_model(model, valRatings, val_full_user,flatten_val_sum_Ratings, val_sum_Ratings, topK)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best MLP model is saved to %s" %(model_out_file))
