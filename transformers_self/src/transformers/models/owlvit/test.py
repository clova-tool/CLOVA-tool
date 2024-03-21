# from transformers import OwlViTProcessor, OwlViTForObjectDetection
from modeling_owlvit import OwlViTForObjectDetection
# from processing_owlvit import OwlViTProcessor
from transformers import OwlViTProcessor
import torch
import os
# import skimage
import numpy as np
from PIL import Image
# from src.dataset import get_dataloaders
from tqdm import tqdm
# from src.train_util import coco_to_model_input

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")


# Download sample image
# image = skimage.data.astronaut()
image = Image.open("assets/a.png").convert("RGB")

# Text queries to search the image for
text_queries = [["human face", "rocket", "nasa badge", "star-spangled banner", 'cat']]
inputs = processor(text=text_queries, images=image, return_tensors="pt")

# Print input names and shapes
for key, val in inputs.items():
    print(f"{key}: {val.shape}")


# model = model.to(device)
model.eval()

# Get predictions
with torch.no_grad():
  outputs = model(**inputs)
## output.logits [1, 576,5], outputs.pred_boxex [1, 576, 4]
# print(outputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to COCO API
results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = text_queries[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
#[576, 4] #[576] # [576]
# Print detected objects and rescaled box coordinates
score_threshold = 0.1
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    if score >= score_threshold:
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
