import json
import os
import random


def get_LVIS(path = None):

    with open(os.path.join(path, 'lvis_v1_train.json'), 'r') as f:  
        data = json.load(f)

    image_path=os.path.join(path,'LVIS_train2017')

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
        image_info={}
        coco_url=images[i]['coco_url']
        coco_url=coco_url.split('/')
        coco_url=coco_url[-1]
        neg_category_ids=images[i]['neg_category_ids']
        image_info['path']=coco_url
        image_info['neg_category_ids']=neg_category_ids
        image_info['categories']=[]
        image_info['category_ids']=[]
        image_info['bboxes']=[]
        image_info['image_id'] = image_id
        image_dict[image_id]=image_info



    annotations=data['annotations']
    annotation_num=len(annotations)
    for i in range (annotation_num):
        image_id=annotations[i]["image_id"]
        category_id=annotations[i]["category_id"]
        bbox=annotations[i]["bbox"]
        category=category_id_dict[category_id]
        image_dict[image_id]['categories'].append(category)
        image_dict[image_id]['category_ids'].append(category_id)
        image_dict[image_id]['bboxes'].append(bbox)

    print ('-------------image_dict finish------------------')


    image_category_list=[]
    for i in range (category_num):
        image_category_list.append([])
    annotations=data['annotations']
    annotation_num=len(annotations)
    for i in range (annotation_num):
        annotation_category_id=annotations[i]['category_id']
        image_id=annotations[i]['image_id']
        category_id=annotations[i]["category_id"]
        # print ('category_id',category_id)
        image_category_list[category_id-1].append(image_dict[image_id])

    print ('-------------image_category_list finish------------------')
    
    return category_dict, image_category_list, image_path, data

    # my_data = {
    #     "category_dict" : category_dict, 
    #     "image_category_list": image_category_list, 
    #     "image_path": image_path
    # }
    # with open("LVIS_temp.json", 'w') as json_file:
    #     json.dump(my_data, json_file)  # 存储字典




# def get_LVIS():
#     with open("LVIS_temp.json", 'r') as json_file:
#         loaded_data = json.load(json_file)  # 读取字典
#     category_dict = loaded_data['category_dict']
#     image_category_list = loaded_data['image_category_list']
#     image_path = loaded_data['image_path']
#     return category_dict, image_category_list, image_path