# Adaptive Dynamic Graph Convolutional Network (ADGraphNet)
A graph convolutional network for MCI identification.

this is my master's thesis research project

**Adaptive Dynamic Graph Convolutional Network for MCI identification**

## Prerequisites
`pip install -r requirements.txt`

### Installation
```
cd torchlight; python setup.py install; cd ..
```

## Data Preparation
Divide training set and test set. (5 fold by  default.)
```
python3 processor/gendata.py --data_path <path to origin MCI data> --out_folder <path to save MCI data>
```

## Training
To train a new ADGraphNet model, run
```
python main.py recognition -c config/train.yaml 
```
or
```
sh config/run.sh
```


# Test
```
python main.py recognition -c config/test.yaml 
```

# Citation
```
@inproceedings{2sagcn2019cvpr,  
      title     = {Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition},  
      author    = {Lei Shi and Yifan Zhang and Jian Cheng and Hanqing Lu},  
      booktitle = {CVPR},  
      year      = {2019},  
}

@article{shi_skeleton-based_2019,
    title = {Skeleton-{Based} {Action} {Recognition} with {Multi}-{Stream} {Adaptive} {Graph} {Convolutional} {Networks}},
    journal = {arXiv:1912.06971 [cs]},
    author = {Shi, Lei and Zhang, Yifan and Cheng, Jian and LU, Hanqing},
    month = dec,
    year = {2019},
}
```
