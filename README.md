# Personalized Fashion Recommender system

Fast fashion is a contemporary term used by fashion retailers to express that designs move from catwalk quickly to capture current fashion trends. Nowadays, a lot of new clohtes are released every day. We look into the monitor all day long to find a favorite clothes among those numerous clothes. It's really time-consuming!! Recommener system could solve this challenge. Recommender system provide users with perosnalized recommendation for clothes, which consider their preference and need.
  
But if you look more closely, there are many more things to consider. As fashion trends change rapidly, people's preferences for clothes change as well. There is also a need to consider seasonality. For example, when winter starts, problems arise because you need to use data from the previous winter to recommend clothes to users. Finally, aesthetics, social consciousness and context must also be considered. Imagine you are wearing red tops, bottoms and shoes, which is really terrible!! And imagine wearing short sleeves, shorts and slippers as a wedding guest dress. This is also terrible.
  
- Will users be happy if they recommended a similar style of clothing? Few people buy clothes of similar style or similar color or similar pattern that they have already purchased.
  
- Will users be happy if they recommend items using the 'knn' model? Because the users are all different, the model may not work well.  
  
- Will users be satisfied if they recommends items by creating a preference-based model that considers the image of the product? It will solve the cold start problem, but it does less to help the user discover different types of items that he might like as much  
  
Through the defined evaluation method, it is possible to predict that the user will be satisfied, but it is not known whether the user is actually satisfied. It is very difficult to create a model to satisfy human emotions. However, this project will satisfy users by recommending items using various reasonable models.



## Similarity based Recommender system

### Similarity Model Architecture
![ex_screenshot](./img/example3.jpg)
##
### Item Recommendation Example
![ex_screenshot](./img/example.JPG)

## Model based Recommender System

### GMF and MLP Model Architecture
![ex_screenshot](./img/example5.JPG)
##
![ex_screenshot](./img/example4.jpg)
comming soon
##

### Item Recommendation Example
![ex_screenshot](./img/example2.JPG)


## Dataset

#### Amazon Datasets-1
- Data for Similarity-based Rs
- AmazonFashion6ImgPartitioned.npy
-https://cseweb.ucsd.edu//~wckang/DVBPR/AmazonFashion6ImgPartitioned.npy click and get 
- Download from https://github.com/kang205/DVBPR

#### Amazon Datasets-2 
- Data for Model-based Rs
- reviews_Clothing_Shoes_and_Jewelry_5.json.gz
- 39387 Users, 23033 Items, 96.92% Sparsity
- Users who rated at least 5 items
- Download from http://jmcauley.ucsd.edu/data/amazon/

#### Amazon Datasets-3
- meta_Clothing_Shoes_and_Jewelry.json.gz
- Download from http://jmcauley.ucsd.edu/data/amazon/

#### Deepfashion dataset
- Data for training yolov3 model
- Download from http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html 

## Example Usage

#### Generate data
```
cd preprocessing
python gem_fashion_dataset.py
```
#### Train yolov3, Run Similarity based model

```
cd keras-yolo3-detection
python train.py
```
```
cd model
python similarity_model.py
```

#### Train Model based Rs(you should tuning hyperparameter)
```
cd model
python amazon_MLP.py
python amazon_GMF.py
```
#### Recommendation Items for Users
```
cd model
python mlp_inference.py
python gmf_inference.py
```
```
# output (if you want real product name, use metadata)

User 0's top-10 recommendation
1. B0000C321X
2. B0008EOEPK
3. B000T9VK56
4. B0002TOZ1E
5. B004L7J7IO
6. B005LERHD8
7. B0000WLSCW
8. B000J6ZYL0
9. B000MXIMHI
10. B0013KBX7K
```
## Requirements
```
pip install -r requirements.txt
```

## Path
```

├── Fashion-Recommender-system
    |
    ├── dataset
    |   ├── In-shop_Clothes_Retrieval_Benchmark
    |   |   ├── Anno
    |   |   └── Img
    |   |       ├── MEN
    |   |       └── WOMEN
    |   ├── amazon_clothing_explicit.json
    |   ├── amazon_clothing_fast_implicit.json
    |   ├── reviews_Clothing_Shoes_and_Jewelry_5.json
    |   ├── amazonfashion6_imgfeature.hdf5
    |   ├── AmazonFashion6ImgPartitioned.npy
    |   └── pretrain
    |    
    ├── keras-yolo3-detection
    |
    ├── preprocessing
    |   └── gen_fashion_dataset.py
    |
    ├── AmazonFashion6ImgPartitioned.npy
    |
    ├── img
    |
    ├── jupyter_notebook
    |
    ├── model
    |   ├── similarity_model.py
    |   ├── ...
    |   ├── ...
    |   └── Image_based_MLP.py(comming soon)
    |
    ├── preprocessing
    |   └── gen_fashion_dataset.py
    |
    └── README.md
    
```

## Reference
- https://en.wikipedia.org/wiki/Fast_fashion 
- https://dressipi.com/blog/building-fashion-recommendation-systems/
- https://github.com/kang205/DVBPR
- http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
- http://jmcauley.ucsd.edu/data/amazon/
- https://github.com/hexiangnan/neural_collaborative_filtering
- Hines, Tony, and M. Bruce. 2001. Fashion marketing - Contemporary issues. Oxford: Butterworth-Heinemann.
- Xiangnan He. Neural Collaborative Filtering. In Proceedings of WWW '17, Perth, Australia, April 03-07
- Julian McAuley. Image-based recommendations on styles and substitutes. SIGIR
- Wang-Cheng Kang. Visually-Aware Fashion Recommendation and Design with Generative Image Models. ICDM'17
- Yifan Hu. Collaborative Filtering for Implicit Feedback Datasets.
