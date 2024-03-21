import matplotlib.pyplot as plt
import torch
from transformers.image_utils import ImageFeatureExtractionMixin
mixin = ImageFeatureExtractionMixin()
import numpy as np
# Load example image

def plot_predictions(input_image, text_queries, scores, boxes, labels, prompt_use, outer_prompt, num):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(input_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()
    score_threshold = 0.1
    for score, box, label in zip(scores, boxes, labels):
      if score < score_threshold:
        continue

      cx, cy, w, h = box
      ax.plot([cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
              [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], "r")
      ax.text(
          cx - w / 2,
          cy + h / 2 + 0.015,
          f"{text_queries[label]}: {score:1.2f}",
          ha="left",
          va="top",
          color="red",
          bbox={
              "facecolor": "white",
              "edgecolor": "red",
              "boxstyle": "square,pad=.3"
          })
    fig.savefig(str(num) + '_' +str(prompt_use)+ '_' +str(outer_prompt)+".jpg")
    


def vis(model, outputs, text_queries, image, prompt_use, outer_prompt, num):
    image_size = model.config.vision_config.image_size
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

    plot_predictions(input_image, text_queries, scores, boxes, labels, prompt_use, outer_prompt, num)


def show_lvis_predict_new(images, image_size, boxes, labels, new_class_list, num_data, epoch):
    # image_size = model.config.vision_config.image_size
    image = images[0]
    image = mixin.resize(image, image_size)
    input_image = np.asarray(image).astype(np.float32) / 255.0

    # Threshold to eliminate low probability predictions
    # score_threshold = 0.5

    # Get prediction logits
    # logits = torch.max(outputs["logits"][0][:,:-1], dim=-1)
    # scores = torch.sigmoid(logits.values).cpu().detach().numpy()

    # # Get prediction labels and boundary boxes
    # labels = logits.indices.cpu().detach().numpy()
    # boxes = outputs["pred_boxes"][0].cpu().detach().numpy()

    plot_predictions_name_new(input_image, new_class_list, boxes, labels, num_data, epoch)

def plot_predictions_name_new(input_image, new_class_list, boxes, labels, num, epoch):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(input_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()
    score_threshold = 0.1
    for box, label_data in zip(boxes, labels):
        label, logit = label_data[0].cpu(), label_data[1].cpu()
        score = torch.sigmoid(logit).cpu().detach().numpy()
    #   if score < score_threshold:
    #     continue

        cx, cy, w, h = box.cpu().detach().numpy()
        ax.plot([cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
                [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], "r")
        ax.text(
            cx - w / 2,
            cy + h / 2 + 0.015,
            f"{new_class_list[label]}: {score:1.2f}",
            ha="left",
            va="top",
            color="red",
            bbox={
                "facecolor": "white",
                "edgecolor": "red",
                "boxstyle": "square,pad=.3"
            })
    fig.savefig('lvis_' + str(num) + "for_debug_epoch_" + str(epoch) + "_pred.jpg")
    plt.close()


def show_lvis_predict(image_size, outputs, new_class_list, images, num_data, epoch):
    # image_size = model.config.vision_config.image_size
    image = images[0]
    image = mixin.resize(image, image_size)
    input_image = np.asarray(image).astype(np.float32) / 255.0

    # Threshold to eliminate low probability predictions
    # score_threshold = 0.5

    # Get prediction logits
    logits = torch.max(outputs["logits"][0][:,:-1], dim=-1)
    scores = torch.sigmoid(logits.values).cpu().detach().numpy()

    # Get prediction labels and boundary boxes
    labels = logits.indices.cpu().detach().numpy()
    boxes = outputs["pred_boxes"][0].cpu().detach().numpy()

    plot_predictions_lvis(input_image, new_class_list, scores, boxes, labels, num_data, epoch)

def plot_predictions_lvis(input_image, new_class_list, scores, boxes, labels, num, epoch):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(input_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()
    score_threshold = 0.1
    for score, box, label in zip(scores, boxes, labels):
      if score < score_threshold:
        continue

      cx, cy, w, h = box
      ax.plot([cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
              [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], "r")
      ax.text(
          cx - w / 2,
          cy + h / 2 + 0.015,
          f"{new_class_list[label]}: {score:1.2f}",
          ha="left",
          va="top",
          color="red",
          bbox={
              "facecolor": "white",
              "edgecolor": "red",
              "boxstyle": "square,pad=.3"
          })
    fig.savefig('lvis_' + str(num) + "_epoch_" + str(epoch) + "_pred.jpg")
    plt.close()


def show_lvis_ori(images, targets, class_name_list, num):
    image = images[0]
    plt.ioff()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image, extent=(0, 1, 1, 0))
    ax.set_axis_off()
    boxes = targets[0]['boxes'].cpu()
    labels = targets[0]['labels'].cpu()

    for box, label in zip(boxes, labels):

      cx, cy, w, h = box
      ax.plot([cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
              [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], "r")
      ax.text(
          cx - w / 2,
          cy + h / 2 + 0.015,
          f"{class_name_list[label.item()]}", ## 加减1需要对应清楚
          ha="left",
          va="top",
          color="red",
          bbox={
              "facecolor": "white",
              "edgecolor": "red",
              "boxstyle": "square,pad=.3"
          })
    fig.savefig('lvis_' +str(num)+ '_ori' +".jpg")
    plt.close()