# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import os
import torch
from util_LOC import box_ops
from util_LOC.vis import vis
from PIL import Image
import torch.nn.functional as F
from util_LOC.vis import show_lvis_ori, show_lvis_predict
import numpy as np
import matplotlib.pyplot as plt



@torch.no_grad()
def evaluate_real(model, processor, criterion, postprocessors, image_real, class_name, device, save_image, save_prompt):
    model.eval()
    criterion.eval()
    images = [Image.open(image_real).convert("RGB")] ###[3, 334, 500]
    inputs = processor(text=[["a photo of " + class_name]], images=images, return_tensors="pt").to(device) #### [pixel_value] [1, 3, 768, 768] [input_ids] [1204, 16]
    ###
    inputs["prompt_use"] = False
    inputs["outer_prompt"] = "ori"
    with torch.no_grad():
        outputs = model(**inputs)
    show_lvis_predict_new(model.config.vision_config.image_size, outputs, images, class_name + "_real_ori")
    cur_image_feature = outputs.vision_model_output.last_hidden_state
    count_pseudo_ori = count_pseudo_right(outputs)

    for k,v in outputs.items():
        if v is not None:
            outputs[k] = v.to('cpu') if isinstance(v, torch.Tensor) else v
        
    target_sizes = torch.Tensor([images[0].size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs,threshold=0.1,target_sizes=target_sizes)
    boxes, scores = results[0]["boxes"], results[0]["scores"]
    boxes = boxes.cpu().detach().numpy().tolist()
    scores = scores.cpu().detach().numpy().tolist()
    if len(boxes)==0:
        # return []
        pass
    else:
        boxes, scores = zip(*sorted(zip(boxes,scores),key=lambda x: x[1],reverse=True))
        selected_boxes = []
        selected_scores = []
        for i in range(len(scores)):
            if scores[i] > 0.1:
                coord = normalize_coord(boxes[i],images[0].size)
                selected_boxes.append(coord)
                selected_scores.append(scores[i])

        selected_boxes, selected_scores = nms(
            selected_boxes,selected_scores, 0.5)
        for i in range(len(selected_boxes)):
            selected_boxes[i] = normalize_coord_new(selected_boxes[i],images[0].size)

        # return selected_boxes
        show_nms(images, selected_boxes, selected_scores, class_name + "_real_ori_nms_")


    res = torch.mean(torch.cat(save_prompt), dim=0).unsqueeze(0)
    inputs["prompt_use"] = True
    inputs["outer_prompt"] = res
    with torch.no_grad():
        outputs = model(**inputs)
    show_lvis_predict_new(model.config.vision_config.image_size, outputs, images, class_name + "_real_mean_")



    ###
    pseudo_sel =[]
    pseudo_weight =[]
    inputs = processor(text=[["a photo of " + class_name]], images=images, return_tensors="pt").to(device) #### [pixel_value] [1, 3, 768, 768] [input_ids] [1204, 16]
    save_prompt_tensor = torch.cat(save_prompt) ### [5, 768]
    inputs["prompt_use"] = True
    for i in range(len(save_prompt_tensor)):
        inputs["outer_prompt"] = save_prompt_tensor[i].unsqueeze(0)
        with torch.no_grad():
            outputs = model(**inputs)
        #show_lvis_predict_new(model.config.vision_config.image_size, outputs, images, class_name  +  "_real_check_"+str(i)+"_")
        count_pseudo_cur = count_pseudo_right(outputs)
        if count_pseudo_cur >= 1 and count_pseudo_cur >= count_pseudo_ori:
            pseudo_sel.append(True)
            pseudo_weight.append(count_pseudo_cur)
        else:
            pseudo_sel.append(False)
            # pseudo_weight.append(0)


    inputs = processor(text=[["a photo of " + class_name]], images=images, return_tensors="pt").to(device) #### [pixel_value] [1, 3, 768, 768] [input_ids] [1204, 16]
    # save_prompt_tensor = torch.cat(save_prompt) ### [5, 768]
    inputs["prompt_use"] = True
    pseudo_weight =torch.tensor(pseudo_weight)
    # pseudo_weight = pseudo_weight / torch.sum(pseudo_weight)
    inputs["outer_prompt"], sim =  get_prompt_new(torch.cat(save_image)[pseudo_sel], save_prompt_tensor[pseudo_sel], cur_image_feature, pseudo_weight) #torch.mean(save_prompt_tensor).unsqueeze(0)
    # inputs["outer_prompt"], sim =  get_prompt_new(torch.cat(save_image), save_prompt_tensor, cur_image_feature, pseudo_weight) #torch.mean(save_prompt_tensor).unsqueeze(0)
    with torch.no_grad():
        outputs = model(**inputs)
    show_lvis_predict_new(model.config.vision_config.image_size, outputs, images, class_name +   "_real_check_softmax_")

    for k,v in outputs.items():
        if v is not None:
            outputs[k] = v.to('cpu') if isinstance(v, torch.Tensor) else v
        
    target_sizes = torch.Tensor([images[0].size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs,threshold=0.1,target_sizes=target_sizes)
    boxes, scores = results[0]["boxes"], results[0]["scores"]
    boxes = boxes.cpu().detach().numpy().tolist()
    scores = scores.cpu().detach().numpy().tolist()
    if len(boxes)==0:
        return []

    boxes, scores = zip(*sorted(zip(boxes,scores),key=lambda x: x[1],reverse=True))
    selected_boxes = []
    selected_scores = []
    for i in range(len(scores)):
        if scores[i] > 0.1:
            coord = normalize_coord(boxes[i],images[0].size)
            selected_boxes.append(coord)
            selected_scores.append(scores[i])

    selected_boxes, selected_scores = nms(
        selected_boxes,selected_scores, 0.5)
    for i in range(len(selected_boxes)):
        selected_boxes[i] = normalize_coord_new(selected_boxes[i],images[0].size)

    # return selected_boxes
    show_nms(images, selected_boxes, selected_scores, class_name + "real_softmax_nms_")

    

def count_pseudo_right(outputs):

    logits = torch.max(outputs["logits"][0], dim=-1)
    scores = torch.sigmoid(logits.values).cpu().detach().numpy()

    labels = logits.indices.cpu().detach().numpy()
    score_threshold = 0.1
    count = 0
    for score in scores:
        if score < score_threshold:
            continue
        count = count + 1
    return count
      

def count_right(outputs, given_label, targets):

    logits = torch.max(outputs["logits"][0], dim=-1)
    scores = torch.sigmoid(logits.values).cpu().detach().numpy()
    boxes = outputs["pred_boxes"][0]
    real_box = targets[0]["boxes"]

    labels = logits.indices.cpu().detach().numpy()
    score_threshold = 0.1
    count = 0
    for score, label, box in zip(scores, labels, boxes):
        if score < score_threshold:
            continue
        if label == given_label:
            iou,_ = box_iou_self(box_ops.box_cxcywh_to_xyxy(box.unsqueeze(0)), box_ops.box_cxcywh_to_xyxy(real_box))
            if torch.max(iou) >= 0.5:
                count = count + 1
    return count
      


# modified from torchvision to also return the union
from torchvision.ops.boxes import box_area
def box_iou_self(boxes1, boxes2):
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union









def show_nms(images, boxes, scores, class_name):
    # image_size = model.config.vision_config.image_size
    image = images[0]
    # image = mixin.resize(image, image_size)
    input_image = np.asarray(image).astype(np.float32) / 255.0

    # Threshold to eliminate low probability predictions
    # score_threshold = 0.5

    # Get prediction logits
    # logits = torch.max(outputs["logits"][0][:,:-1], dim=-1)
    # scores = torch.sigmoid(logits.values).cpu().detach().numpy()

    # # Get prediction labels and boundary boxes
    # labels = logits.indices.cpu().detach().numpy()
    # boxes = outputs["pred_boxes"][0].cpu().detach().numpy()

    plot_nms(input_image, boxes, scores, class_name)

def plot_nms(input_image, boxes, scores, class_name):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(input_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()
    for box, label_data in zip(boxes, scores):
        box, score = box, label_data
        cx, cy, w, h = box.cpu().numpy()
        ax.plot([cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
                [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], "r")
        ax.text(
            cx - w / 2,
            cy + h / 2 + 0.015,
            f"{class_name}: {score:1.2f}",
            ha="left",
            va="top",
            color="red",
            bbox={
                "facecolor": "white",
                "edgecolor": "red",
                "boxstyle": "square,pad=.3"
            })
    fig.savefig(os.path.join("failed_class_prompt/out_fig", 'test_' + class_name + "_nms.jpg"))
    # fig.savefig('test_' + class_name + "_nms.jpg")
    plt.close()


        
def normalize_coord(bbox,img_size):
    w,h = img_size
    x1,y1,x2,y2 = [int(v) for v in bbox]
    x1 = max(0,x1)
    y1 = max(0,y1)
    x2 = min(x2,w-1)
    y2 = min(y2,h-1)
    return [x1,y1,x2,y2]


from src.util import BoxUtil       
def normalize_coord_new(bbox,img_size):
    # w,h = img_size
    w, h = img_size
    boxes = torch.tensor(bbox) / torch.tensor([w, h, w, h], dtype=torch.float32)
    boxes = BoxUtil.box_convert(boxes, "xyxy", "cxcywh")
    return boxes

def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score

def get_prompt_new(save_image_tensor, save_prompt_tensor, cur_image_feature, pseudo_weight):
    save_image_tensor = save_image_tensor[:,0:1,:]
    cur_image_feature =cur_image_feature[:,0:1,:]

    # save_image_tensor = save_image_tensor / (torch.linalg.norm(save_image_tensor, dim=-1, keepdim=True) + 1e-6)
    # cur_image_feature = cur_image_feature / (torch.linalg.norm(cur_image_feature, dim=-1, keepdim=True) + 1e-6)
    if len(pseudo_weight) == 0:
        return torch.zeros(1, 1, 768).cuda(), None
    cur_image_feature = cur_image_feature.expand(len(save_image_tensor), -1, -1)
    euclidean_distances = torch.norm(cur_image_feature - save_image_tensor, dim=(1, 2))
    sim = F.softmax(-euclidean_distances / 5, dim=0)  # 注意使用负号将距离转换为相似性
    sim = (sim + pseudo_weight.cuda() / torch.sum(pseudo_weight).cuda() ) / 2
    res  = torch.sum(sim.view(sim.shape[0], 1, 1) * save_prompt_tensor, dim =0).unsqueeze(0)
    return res, sim


def get_prompt_ori(save_image_tensor, save_prompt_tensor, cur_image_feature):
    # save_image_tensor = torch.cat(save_image) ### [5, 768]
    # save_prompt_tensor = torch.cat(save_prompt) ### [5, 10, 768]
    save_image_tensor = save_image_tensor[:,0:1,:]
    cur_image_feature =cur_image_feature[:,0:1,:]

    # save_image_tensor = save_image_tensor / (torch.linalg.norm(save_image_tensor, dim=-1, keepdim=True) + 1e-6)
    # cur_image_feature = cur_image_feature / (torch.linalg.norm(cur_image_feature, dim=-1, keepdim=True) + 1e-6)
    cur_image_feature = cur_image_feature.expand(len(save_image_tensor), -1, -1)
    euclidean_distances = torch.norm(cur_image_feature - save_image_tensor, dim=(1, 2))
    # tensor([1192.7373, 1151.4248, 1187.8894, 1202.9275, 1206.1204, 1225.3323,
    #     1176.7173, 1143.4907, 1100.2905, 1177.4397], device='cuda:0')
    # cur_image_feature = cur_image_feature.view(len(cur_image_feature), -1)
    # save_image_tensor = save_image_tensor.view(len(save_image_tensor), -1)
    # dot_products = torch.mm(cur_image_feature, save_image_tensor.t())
    # normalized_vector1 = F.normalize(cur_image_feature, p=2, dim=1)  # 使用L2范数进行归一化
    # normalized_vector2 = F.normalize(save_image_tensor, p=2, dim=1)  # 使用L2范数进行归一化
    # dot_products = torch.mm(cur_image_feature, save_image_tensor.t())  # .t()进行转置以匹配维度
    # euclidean_distance = torch.norm(cur_image_feature - save_image_tensor, dim=1)
    # euclidean_distances = torch.nn.functional.pairwise_distance(cur_image_feature, save_image_tensor)
    # values, indices = torch.topk(euclidean_distances, min(top_k, len(save_prompt_tensor)), largest=False)
    values, indices = torch.topk(euclidean_distances, len(save_prompt_tensor), largest=False)
    # m = nn.Softmax(dim=1)
    # sim = m(values)
    sim = F.softmax(-values / 5, dim=0)  # 注意使用负号将距离转换为相似性
    sel_prompt = save_prompt_tensor[indices.squeeze(0)]
    # res  = torch.sum(sim.view(sim.shape[0], 1, 1) * sel_prompt, dim =0).unsqueeze(0)
    res  = torch.sum(sim.view(sim.shape[0], 1, 1) * sel_prompt, dim =0).unsqueeze(0)
    # return res, values, indices, save_prompt_tensor
    return res, values, indices, sim


from transformers.image_utils import ImageFeatureExtractionMixin
mixin = ImageFeatureExtractionMixin() 
def show_lvis_predict_new(image_size, outputs, images, name):
    # image_size = model.config.vision_config.image_size
    image = images[0]
    image = mixin.resize(image, image_size)
    input_image = np.asarray(image).astype(np.float32) / 255.0

    # Threshold to eliminate low probability predictions
    # score_threshold = 0.5

    # Get prediction logits
    logits = torch.max(outputs["logits"][0], dim=-1)
    scores = torch.sigmoid(logits.values).cpu().detach().numpy()

    # Get prediction labels and boundary boxes
    labels = logits.indices.cpu().detach().numpy()
    boxes = outputs["pred_boxes"][0].cpu().detach().numpy()

    plot_predictions_lvis(input_image, scores, boxes, labels, name)

def plot_predictions_lvis(input_image, scores, boxes, labels, name):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(input_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()
    score_threshold = 0.05
    for score, box, label in zip(scores, boxes, labels):
      if score < score_threshold:
        continue

      cx, cy, w, h = box
      ax.plot([cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
              [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], "r")
      ax.text(
          cx - w / 2,
          cy + h / 2 + 0.015,
          f"{label}: {score:1.2f}",
          ha="left",
          va="top",
          color="red",
          bbox={
              "facecolor": "white",
              "edgecolor": "red",
              "boxstyle": "square,pad=.3"
          })
    # fig.savefig('test_' + name + "_pred.jpg")
    fig.savefig(os.path.join("failed_class_prompt/out_fig", 'test_' + name + "_pred.jpg"))
    plt.close()