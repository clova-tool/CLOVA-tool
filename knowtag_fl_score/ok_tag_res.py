
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# import tqdm

# from PIL import Image
# from IPython.core.display import HTML

# from engine.utils import ProgramGenerator, ProgramInterpreter
# from prompts.knowtag import PROMPT
import matplotlib.pyplot as plt
import json
# from tool_eval import VQAEval

def normalize_bbox(bbox, image_width, image_height):
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2 / image_width
    cy = (y_min + y_max) / 2 / image_height
    w = (x_max - x_min) / image_width
    h = (y_max - y_min) / image_height
    return cx, cy, w, h


def plot_predictions(input_image, real_boxes):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(input_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()

    width, height = input_image.size

    for box in real_boxes:
        new_box = normalize_bbox(box, width, height)
        cx, cy, w, h = new_box
        ax.plot([cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
                [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], "r")



def calculate_iou(boxA, boxB):
    # 计算两个边界框的交集面积
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # 计算两个边界框的并集面积
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou


def calculate_f1_score_loc(predictions, ground_truths, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred_box in predictions:
        matched = False
        for gt_box in ground_truths:
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                matched = True
                break
        if matched:
            true_positives += 1
        else:
            false_positives += 1
    
    false_negatives = len(ground_truths) - true_positives   
    return true_positives, false_positives, false_negatives


def calculate_f1_score_single(predictions, ground_truths, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred_box, pred_label in predictions:
        matched = False
        for gt_box, gt_label in ground_truths:
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold and pred_label == gt_label:
                matched = True
                break
        if matched:
            true_positives += 1
        else:
            false_positives += 1
    
    for gt_box, gt_label in ground_truths:
        matched = False
        for pred_box, pred_label in predictions:
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold and pred_label == gt_label:
                matched = True
                break
        if not matched:
            false_negatives += 1

    # return true_positives, false_positives, false_negatives
    if true_positives + false_positives==0:
        precision=0
    else:
        precision = true_positives / (true_positives + false_positives)
        
    if true_positives + false_negatives==0:
        recall=0
    else:
        recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score

def calculate_f1_score_tag(predictions, ground_truths, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred_box, pred_label in predictions:
        matched = False
        for gt_box, gt_label in ground_truths:
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold and pred_label == gt_label:
                matched = True
                break
        if matched:
            true_positives += 1
        else:
            false_positives += 1
    
    for gt_box, gt_label in ground_truths:
        matched = False
        for pred_box, pred_label in predictions:
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold and pred_label == gt_label:
                matched = True
                break
        if not matched:
            false_negatives += 1

    return true_positives, false_positives, false_negatives



def judge_all_right(predictions, ground_truths, iou_threshold=0.5):

    if len(predictions) != len(ground_truths):
        return False
    
    for pred_box, pred_label in predictions:
        matched = False
        for gt_box, gt_label in ground_truths:
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold and pred_label == gt_label:
                matched = True
                break
        if matched:
            continue
        else:
            return False
    return True



# with open('fl_score/ok_tag_box_real.json', 'r') as file:
#     data_real = file.readlines()
# with open('fl_score/result_after_0.03_0.4.json', 'r') as file:
#     data_pred = file.readlines()


# # Loc precision:0.603, recall:0.409, f1_score:0.487
# # Tagging precision:0.476, recall:0.323, f1_score:0.385
# ###
# total_true_positives = 0
# total_false_positives = 0
# total_false_negatives = 0
# for real, pred in zip(data_real, data_pred):
#     real_json = json.loads(real.strip())
#     pred_json = json.loads(pred.strip())

#     if real_json['image'] != pred_json['image']:
#         print("wrong order")
#         break

#     real_box_all = real_json['real']
#     pred_box_all = pred_json['res']
#     for data in pred_box_all:
#         if 'category' in data.keys():
#             del data['category']
#         if 'class_score' in data.keys():
#             del data['class_score']
    
#     new_format_pred =[]
#     new_format_real = []
#     for data in pred_box_all:
#         new_format_pred.append((data['box']))
#     for data in real_box_all:
#         new_format_real.append((data['box']))
                     
#     true_positives, false_positives, false_negatives = calculate_f1_score_loc(new_format_pred, new_format_real)
#     total_true_positives += true_positives
#     total_false_positives += false_positives
#     total_false_negatives += false_negatives

# precision = total_true_positives / (total_true_positives + total_false_positives)
# recall = total_true_positives / (total_true_positives + total_false_negatives)

# if precision + recall == 0:
#     f1_score = 0
# else:
#     f1_score = 2 * (precision * recall) / (precision + recall)
# print("Loc precision:{:.3f}, recall:{:.3f}, f1_score:{:.3f}".format(precision, recall, f1_score))



# ###
# total_true_positives = 0
# total_false_positives = 0
# total_false_negatives = 0
# for real, pred in zip(data_real, data_pred):
#     real_json = json.loads(real.strip())
#     pred_json = json.loads(pred.strip())

#     if real_json['image'] != pred_json['image']:
#         print("wrong order")
#         break

#     real_box_all = real_json['real']
#     pred_box_all = pred_json['res']
#     for data in pred_box_all:
#         if 'category' in data.keys():
#             del data['category']
#         if 'class_score' in data.keys():
#             del data['class_score']
    
#     new_format_pred =[]
#     new_format_real = []
#     for data in pred_box_all:
#         new_format_pred.append((data['box'], data['class']))
#     for data in real_box_all:
#         new_format_real.append((data['box'], data['class']))

#     is_all_right = judge_all_right(new_format_pred, new_format_real) ### 判断每张图是否完全正确   
#     print("signle image predictiobn: ", is_all_right)   
#     f1_single = calculate_f1_score_single(new_format_pred, new_format_real) ### 计算单张图f1 score   
#     print("single image f1_socre", f1_score)                             
#     true_positives, false_positives, false_negatives = calculate_f1_score_tag(new_format_pred, new_format_real)
#     total_true_positives += true_positives
#     total_false_positives += false_positives
#     total_false_negatives += false_negatives

# precision = total_true_positives / (total_true_positives + total_false_positives)
# recall = total_true_positives / (total_true_positives + total_false_negatives)
# # acc = (total_true_positives +)
# if precision + recall == 0:
#     f1_score = 0
# else:
#     f1_score = 2 * (precision * recall) / (precision + recall)
# print("Tagging precision:{:.3f}, recall:{:.3f}, f1_score:{:.3f}".format(precision, recall, f1_score))

