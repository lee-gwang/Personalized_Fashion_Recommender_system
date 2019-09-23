import math
import heapq
import multiprocessing
import numpy as np
from time import time
import random
from copy import copy
import json

_testRatings = None
_testNegatives = None
_K = None


def evaluate_model(testRatings, testNegatives, K, num_thread):

    global _testRatings
    global _testNegatives
    global _K
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
    

    hits, ndcgs = [],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in range(len(_testRatings)):
        (hr,ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs) 

def eval_one_rating(idx):
    rating = _testRatings[idx] 
    items = _testNegatives[idx]   
    gtItem = rating[1]       
    items.append(gtItem)
    rank = copy(items)
    items.pop() 
    
    # Evaluate top rank list
    ranklist=random.sample(rank,_K)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

topK = 10
evaluation_threads = 1 #mp.cpu_count()

dataset = open('../dataset/amazon_clothing_fast_implicit.json',encoding='utf-8-sig').read()
js=json.loads(dataset)

train, valRatings, valNegatives,usernum,itemnum = js['train'],js['valRatings'],js['valNegatives'],js['usernum'],js['itemnum']


print('Data load complete')
print('random model evaluation...')
t1=time()
(hits, ndcgs) = evaluate_model(valRatings, valNegatives,topK, evaluation_threads)
hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
t2=time()
print('HR = %.4f, NDCG = %.4f' % (hr, ndcg))