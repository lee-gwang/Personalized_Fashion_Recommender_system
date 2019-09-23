import math
import heapq 
import multiprocessing
import numpy as np
from time import time
import random
from copy import copy
from collections import Counter
from itertools import chain
import json

_testRatings = None
_testNegatives = None
_K = None
_result = None

def evaluate_model(testRatings, testNegatives, K, num_thread, result):

    global _testRatings
    global _testNegatives
    global _K
    global _result
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
    _result = result

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
    random.shuffle(rank)
    map_item_score = {}
    for i in rank:
        map_item_score[i] = _result[i]
          
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get) 
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

allitems = list(chain.from_iterable(train.values()))
result = Counter(allitems)


print('Data load complete')
print('item popularity model evaluation...')
t1=time()
(hits, ndcgs) = evaluate_model(valRatings, valNegatives, topK, evaluation_threads,result)
hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
t2=time()
print('HR = %.4f, NDCG = %.4f' % (hr, ndcg))