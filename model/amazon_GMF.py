#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 18:22:23 2019

@author: lee
"""

import numpy as np
import keras
from keras import backend as K
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten
from fast_evaluation import evaluate_model
from time import time
import argparse
import json
import sys
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation

from keras import initializers, regularizers
import multiprocessing as mp

"""
Argument
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path', nargs='?', default='',
                        help='dataset path')

    parser.add_argument('--dataset', nargs='?', default='amazon',
                        help='dataset')

    parser.add_argument('--epochs', type=int, default=50,
                        help='epochs')

    parser.add_argument('--batch_size', type=int, default=2**14,
                        help='batch size')

    parser.add_argument('--num_factors', type=int, default=4,
                        help='user, item embedding size')

    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="user, item embedding regularization")

    parser.add_argument('--num_neg', type=int, default=4,
                        help='number of negative instances')

    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate.')

    parser.add_argument('--learner', nargs='?', default='adam',
                        help='optimizers : adagrad, adam, rmsprop, sgd')

    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')

    parser.add_argument('--out', type=int, default=1,
                        help='if 1, save train model')

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

    model = Model(inputs=[user_input, item_input], outputs=prediction)

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
                item_input.append(str(j))
                labels.append(0)
    return user_input, item_input, labels


if __name__ == '__main__':
    args = parse_args()
    num_factors = args.num_factors
    regs = eval(args.regs)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose

    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("GMF arguments: %s" % (args))
    model_out_file = '../dataset/pretrain/%s_GMF_%d_%d.h5' % (args.dataset, num_factors, time())
   
    t1 = time()
    dataset = open('../dataset/amazon_clothing_fast_implicit.json',encoding='utf-8-sig').read()
    js=json.loads(dataset)
    train, valRatings, valNegatives,val_full_user, val_sum_Ratings, flatten_val_sum_Ratings, usernum,itemnum = js['train'],js['valRatings'],js['valNegatives'],js['val_full_user'],js['val_sum_Ratings'],js['flatten_val_sum_Ratings'], js['usernum'],js['itemnum']
    print("Load data done [%.1f s]. #user=%d, #item=%d"
          % (time() - t1, usernum, itemnum))
    
    model = get_model(usernum, itemnum, num_factors, regs)
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
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time() - t1))


    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        
        user_input, item_input, labels = get_train_instances(train,num_negatives)

        hist = model.fit([np.array(user_input), np.array(item_input)],
                         np.array(labels),  # labels
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, valRatings, val_full_user,flatten_val_sum_Ratings, val_sum_Ratings, topK)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best GMF model is saved to %s" % (model_out_file))