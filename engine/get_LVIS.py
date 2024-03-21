from PIL import Image
import requests
import torch

import json
import random
import pycocotools.mask as mask_utils
import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm
import os


def ann_to_rle(h,w,segm):
    """Convert annotation which can be polygons, uncompressed RLE to RLE.
    Args:
        ann (dict) : annotation object

    Returns:
        ann (rle)
    """
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask_utils.frPyObjects(segm, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(segm["counts"], list):
        # uncompressed RLE
        rle = mask_utils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann["segmentation"]
    return rle



def ann_to_mask(imgs):
    """Convert annotation which can be polygons, uncompressed RLE, or RLE
    to binary mask.
    Args:
        ann (dict) : annotation object

    Returns:
        binary mask (numpy 2D array)
    """

    h=imgs['height']
    w=imgs['width']

    segmentations=imgs['segmentations']
    segmentation_mask=[]
    for i in range(len(segmentations)):
        rle = ann_to_rle(h,w,segmentations[i])
        mask=mask_utils.decode(rle)
        segmentation_mask.append(mask)

    segmentation_mask=torch.tensor(np.array(segmentation_mask))

    return segmentation_mask


def get_LVIS(path = '/home/zhangxintong/AA_CODE/VL/visprog/gpt3.5_imageediting_allupdate_11.2/LAVIS/'):

    with open(path+'lvis_v1_train.json', 'r') as f:  
        data = json.load(f)


    categories=data['categories']
    category_num=len(categories)
    category_dict={}
    for i in range (category_num):
        category_name=categories[i]['name']
        category_id=categories[i]['id']
        category_dict[category_name]=category_id


    category_id_dict={}
    for i in range (category_num):
        category_name=categories[i]['name']
        category_id=categories[i]['id']
        category_id_dict[category_id]=category_name
    print ('-------------category_id_dict finish------------------')



    images=data['images']
    image_dict={}
    image_num=len(images)
    for i in range (image_num):
        image_id=images[i]['id']
        image_info=images[i]
        coco_url=images[i]['coco_url']
        coco_url=coco_url.split('/')
        coco_url=coco_url[-1]
        neg_category_ids=images[i]['neg_category_ids']
        image_info['path']=coco_url
        image_info['neg_category_ids']=neg_category_ids
        image_info['categories']=[]
        image_info['category_ids']=[]
        image_info['bboxes']=[]
        image_info['segmentations']=[]
        image_dict[image_id]=image_info

    print ('-------------image_dict finish1------------------')


    annotations=data['annotations']
    annotation_num=len(annotations)
    for i in range (annotation_num):
        image_id=annotations[i]["image_id"]
        category_id=annotations[i]["category_id"]
        bbox=annotations[i]["bbox"]
        segmentation=annotations[i]["segmentation"]
        # segmentation=ann_to_mask(image_dict,annotations[i])
        category=category_id_dict[category_id]
        image_dict[image_id]['categories'].append(category)
        image_dict[image_id]['category_ids'].append(category_id)
        image_dict[image_id]['bboxes'].append(bbox)
        image_dict[image_id]['segmentations'].append(segmentation)

    print ('-------------image_dict finish2------------------')


    image_category_list=[]
    for i in range (category_num):
        image_category_list.append([])
    annotations=data['annotations']
    annotation_num=len(annotations)
    for i in range (annotation_num):
        annotation_category_id=annotations[i]['category_id']
        image_id=annotations[i]['image_id']
        category_id=annotations[i]["category_id"]
        image_category_list[category_id-1].append(image_dict[image_id])

    print ('-------------image_category_list finish------------------')

    return category_dict,image_category_list



def color_palette():
    """Color palette that maps each class to RGB values.
    
    This one is actually taken from ADE20k.
    """
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]