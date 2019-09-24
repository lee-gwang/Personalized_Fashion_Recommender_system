# Fashion-Recommender-system
This project is fashion recommender system
it is simiple recommender system
오늘날 새로운 옷들이 쏟아진다(fast fashion 언급), 자라언급
사용자는 수많은 원피스중에 원하는 원피스를 찾기위해 끊임없이 여러 쇼핑몰을 드나든다
이건 너무나 시간낭비!
추천시스템은 사용자의 선호도를 파악하여 원하는 옷을 추천해줄 수 있다.
similar based rs는 street fashion 사진에서 하의,상의, fullbody 를 detect하고 비슷한 스타일의 상품을 찾아줄 수 있다.
대부분의 사람들은 이러한 유사한 상품을 찾아주는 rs가 최선이라 생각하지만 아니다.
어느 누가 자신이 구매한 원피스와 비슷한 스타일의 원피스를 또 구매할거란 건가?
similar based rs는 옷을 detect하고 street fashion에서 구매한 원피스와 어울리는 style을 찾을 수 있다.
이 프로젝트는 feed back 기반의 추천시스템 모델과 단순한 similarity based Rs를 만들었으며
향후에 이사람이 구매한 상품과 어룰리는 다른 상품을 추천하기 위한 추천시스템을 구현할 것이다.


## Similarity based Recommender system
![ex_screenshot](./img/example.JPG)

##
### Similarity Model Architecture
![ex_screenshot](./img/example3.jpg)





## Model based Recommender System
![ex_screenshot](./img/example2.JPG)
##
### GMF and MLP Model Architecture
![ex_screenshot](./img/example5.JPG)
##





![ex_screenshot](./img/example4.jpg)
comming soon





## Dataset
#### Amazon Datasets Used for Model-based Rs
- reviews_Clothing_Shoes_and_Jewelry_5.json.gz
- 39387 Users, 23033 Items, 96.92% Sparsity
- Users who rated at least 5 items
- Download from http://jmcauley.ucsd.edu/data/amazon/

#### Amazon Metadatasets
- meta_Clothing_Shoes_and_Jewelry.json.gz
- Download from http://jmcauley.ucsd.edu/data/amazon/

#### Amazon Datasets Used for Similarity-based Rs
- AmazonFashion6ImgPartitioned.npy
- Download from https://github.com/kang205/DVBPR

#### Deepfashion dataset
- Data for training yolov3 model
- Download from http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html 

## Example Usage

#### Generate data
```
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
python amazon_MLP.py
python amazon_GMF.py
```
#### Recommendation Items for Users
```
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
- https://github.com/kang205/DVBPR
- http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
- http://jmcauley.ucsd.edu/data/amazon/
- https://github.com/hexiangnan/neural_collaborative_filtering
- Neural Collaborative Filtering. Xiangnan He. In Proceedings of WWW '17, Perth, Australia, April 03-07
- Image-based recommendations on styles and substitutes. Julian McAuley. SIGIR
