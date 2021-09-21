from __future__ import division
import torch
import random
import numpy as np
from sklearn.cluster import DBSCAN, KMeans

import cv2

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _init_fn():
    np.random.seed(0)

def clustering(features, num_cluster=20, method='kmeans', eps = 0.6,):
    if method =='DBSCAN':
        cluster = DBSCAN(eps, min_samples=5, metric='euclidean', n_jobs=-1)
        cluster_ids = cluster.fit_predict(features)
    elif method =='kmeans':
        cluster = KMeans(num_cluster)
        cluster_ids = cluster.fit_predict(features)
    return cluster_ids

def pretty_dict_string(d, indent=0):
    string = ''
    for key, value in d.items():
        string += '\t' * indent + str(key)
        if isinstance(value, dict):
            string += pretty_dict_string(value, indent + 1)
        else:
            string +=('\t' * (indent + 1) + str(value) + '\n')
    return string


def generate_confident_target(pred, confidence_threshold = 0):

    prob,target = torch.max(pred,1)
    target[prob<confidence_threshold] = 255
    return target

def generate_ignore_region_cutmix(mask):

    kernel = np.ones((5, 5), np.uint8)
    mask  =  1- mask.cpu().numpy()

    batch = []
    for m in mask:
        erosion = cv2.erode(m[0], kernel, iterations=1)
        dilation = cv2.dilate(m[0], kernel, iterations=1)
        ignore_mask = torch.from_numpy(1 - (dilation - erosion))
        batch.append(ignore_mask[None, ...])
    batch = torch.cat(batch,dim=0)
    return batch