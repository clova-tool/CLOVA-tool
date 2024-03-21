import cv2
import os
import torch
import openai
import functools
import numpy as np
import face_detection
import io, tokenize
from augly.utils.base_paths import EMOJI_DIR
import augly.image as imaugs
import PIL
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from transformers import (ViltProcessor, ViltForQuestionAnswering, 
    OwlViTProcessor, OwlViTForObjectDetection,
    CLIPProcessor, CLIPModel, AutoProcessor, BlipForQuestionAnswering)
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline
# import ruamel.yaml as yaml
from tools import clip
import yaml
import random
import numpy as np


from engine.nms import nms

from tools.blip_vqa.meta_distance import MultiHeadAttention


from engine.data_utils import pre_question
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torch.utils.data import Dataset
from diffusers.optimization import get_scheduler
from engine.image2text import I2T_function
from tqdm import tqdm

# # from blip_vqa.config import config as vqa_config
all_updated_model_config_path='configs/all_updated_model_config.yaml'
all_model_config=  yaml.load(open(all_updated_model_config_path, 'r'), Loader=yaml.Loader)


from engine.data_utils import pre_question
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from engine.get_LVIS import get_LVIS, color_palette, ann_to_mask


#######################################

from transformers_self.src.transformers.models.maskformer.modeling_maskformer import MaskFormerForInstanceSegmentation
from transformers_self.src.transformers.models.maskformer.feature_extraction_maskformer import MaskFormerFeatureExtractor



def parse_step(step_str,partial=False):

    tokens = list(tokenize.generate_tokens(io.StringIO(step_str).readline))
    output_var = tokens[0].string
    step_name = tokens[2].string
    parsed_result = dict(
        output_var=output_var,
        step_name=step_name)
    if partial:
        return parsed_result

    arg_tokens = [token for token in tokens[4:-3] if token.string not in [',','=']]
    num_tokens = len(arg_tokens) // 2
    args = dict()
    for i in range(num_tokens):
        args[arg_tokens[2*i].string] = arg_tokens[2*i+1].string
    parsed_result['args'] = args
    return parsed_result



class SegmentInterpreter():
    step_name = 'SEG'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.config = all_model_config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = MaskFormerFeatureExtractor.from_pretrained(
            self.config["SEG"]['init']['pretrained_fe'])
        self.model = MaskFormerForInstanceSegmentation.from_pretrained(
            self.config["SEG"]['init']['pretrained_model']).to(self.device)
        self.model.eval()

        self.data_num= self.config["SEG"]['init']['data_num']
        self.iterative_num= self.config["SEG"]['init']['iterative_num']
        self.batch_num= self.config["SEG"]['init']['batch_num']
        self.categpry_name_pool={}
        self.GLOBAL_TOKEN_H= self.config["SEG"]['init']['GLOBAL_TOKEN_H']
        self.GLOBAL_TOKEN_W= self.config["SEG"]['init']['GLOBAL_TOKEN_W']
        self.prompt_num=self.GLOBAL_TOKEN_H*self.GLOBAL_TOKEN_W
        self.image_path= self.config["SEG"]['init']['image_path']

        self.feature_dim= self.config["SEG"]['init']['feature_dim']
        self.prompt_dimension=256*self.GLOBAL_TOKEN_H*self.GLOBAL_TOKEN_W

        self.palette = color_palette()

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,output_var

    def predict(self, img, query):

        print ('query:',query)
        print ('segmentation categpry_name_pool', self.categpry_name_pool)

        inputs = self.feature_extractor(images=img, return_tensors="pt")
        inputs = {k:v.to(self.device) for k,v in inputs.items()}

        if query in self.categpry_name_pool.keys():
            print ('this segmentation class has been learned')
            image_features, pixel_embeddings=self.model.feature_extractor(inputs["pixel_values"])
            image_features=image_features.detach()
            pixel_embeddings=pixel_embeddings.detach()

            image_features=image_features.squeeze()
            image_features=image_features.view(image_features.shape[0],image_features.shape[1]*image_features.shape[2])
            image_features=torch.mean(image_features,dim=1, keepdim=True)
            image_features=image_features.permute(1,0) 

            this_prompt=self.prompt_generation(query,self.categpry_name_pool,image_features)
            this_prompt=this_prompt.detach()
            this_prompt=this_prompt.view(1,256,self.prompt_num)

            with torch.no_grad():
                outputs = self.model(inputs["pixel_values"],use_prompt=True, outer_prompt=this_prompt)
            post_outputs = self.feature_extractor.post_process_panoptic_segmentation(outputs)[0]
            instance_map = post_outputs['segmentation'].cpu().numpy()

        else:
            print ('this segmentation class has not been learned')
            with torch.no_grad():
                outputs = self.model(**inputs)
            post_outputs = self.feature_extractor.post_process_panoptic_segmentation(outputs)[0]
            instance_map = post_outputs['segmentation'].cpu().numpy()


        ##########added to show segmentation results##########
        saved_outputs = self.feature_extractor.post_process_panoptic_segmentation(outputs,target_sizes=[img.size[::-1]])[0]
        saved_instance_map = saved_outputs['segmentation'].cpu()

        color_segmentation_map = np.zeros((saved_instance_map.shape[0], saved_instance_map.shape[1], 3), dtype=np.uint8) # height, width, 3

        count=0
        for label, color in enumerate(self.palette):
            if sum(sum(saved_instance_map == label))!=0:
                color_segmentation_map[saved_instance_map == label, :] = self.palette[count]
                count=count+1

        ground_truth_color_seg = color_segmentation_map[..., ::-1]
        print ('image size',img.size)
        print ('ground_truth_color_seg size',ground_truth_color_seg.shape)
        img_result = np.array(img) * 0.5 + ground_truth_color_seg * 0.5
        img_result = img_result.astype(np.uint8)
        img_result=Image.fromarray(img_result)
        # img.save(f'./{filter_name}/after_25_{image_path}')
        ##########added to show segmentation results##########


        objs = []
        print(post_outputs.keys())
        for seg in post_outputs['segments_info']:
            inst_id = seg['id']
            label_id = seg['label_id']
            category = self.model.config.id2label[label_id]
            mask = (instance_map==inst_id).astype(float)
            resized_mask = np.array(
                Image.fromarray(mask).resize(
                    img.size,resample=Image.BILINEAR))
            Y,X = np.where(resized_mask>0.5)
            x1,x2 = np.min(X), np.max(X)
            y1,y2 = np.min(Y), np.max(Y)
            num_pixels = np.sum(mask)
            objs.append(dict(
                mask=resized_mask,
                category=category,
                box=[x1,y1,x2,y2],
                inst_id=inst_id
            ))

        return objs, img_result


    def execute(self,prog_step,query_name=''):
        img_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs, seg_img = self.predict(img,query=query_name)
        # prog_step.state['Seg_Results'] = seg_img
        prog_step.state[output_var] = objs

        return objs


    def prompt_generation(self,query_name,categpry_name_pool,image_feature):
        if query_name in categpry_name_pool.keys():
            print ('generate prompt')
            category_prompt_pool=categpry_name_pool[query_name][0]
            category_feature_pool=categpry_name_pool[query_name][1]

            prompt_generate_model=MultiHeadAttention(n_head=self.config["SEG"]['prompt_generation']['n_head'], d_model=self.feature_dim, d_k=self.feature_dim, d_v=self.prompt_dimension)
            prompt_generate_model=prompt_generate_model.to(self.device)
            prompt_generate_model.eval()
            image_feature=image_feature.unsqueeze(0)
            category_feature_pool=category_feature_pool.unsqueeze(0)
            category_prompt_pool=category_prompt_pool.unsqueeze(0)
            category_feature_pool=category_feature_pool.to(self.device)
            category_prompt_pool=category_prompt_pool.to(self.device)

            print ('image_feature',image_feature.shape)
            print ('category_feature_pool',category_feature_pool.shape)
            print ('category_prompt_pool',category_prompt_pool.shape)
            this_prompt=prompt_generate_model(image_feature, category_feature_pool, category_prompt_pool)
            return this_prompt

        else:
            return None

    def update(self, query):

        if query in self.categpry_name_pool.keys():
            print ('this concept has been learned in the SEG model')
        else:
            category_dict,image_category_list=get_LVIS(path = self.config["SEG"]['update']['LVIS_path'])

            if query in category_dict.keys():
                given_category_id=category_dict[query]
                give_category_data=image_category_list[given_category_id-1]

                prompt_pool=[]
                feature_pool=[]

                for i in range (self.data_num):
                    sampled_data = random.sample(give_category_data,1)

                    image_features, learnedd_prompt=self.prompt_train(sampled_data)

                    prompt_pool.append(learnedd_prompt)# size(1, 25600)
                    self.model.init_prompt()
                    prompt_pool_tensor=torch.cat(prompt_pool,dim=0)

                    feature_pool.append(image_features) # size(1,1024)
                    feature_pool_tensor=torch.cat(feature_pool,dim=0)
                self.categpry_name_pool[query]=[prompt_pool_tensor,feature_pool_tensor] 
                print ('self.categpry_name_pool',self.categpry_name_pool)
            else:
                print ('we cannot learn such concept for the SEG model')



    def prompt_train(self,sampled_data):

        to_optim =  [
                    {'params':self.model.model.transformer_module.prompt_embeddings,'lr':self.config["SEG"]['prompt_train']['lr']}, ##  ori
                    ]
        optimizer = torch.optim.AdamW(to_optim)

        image=Image.open(self.image_path+sampled_data[0]['path']).convert('RGB')

        categories=sampled_data[0]['categories']

        category_ids=sampled_data[0]['category_ids']
        category_ids=torch.tensor(np.array(category_ids))

        segmentations=sampled_data[0]['segmentations']
        segmentations=ann_to_mask(sampled_data[0])

        segmentation_maps=torch.sum(segmentations*category_ids.view(category_ids.shape[0],1,1).expand(category_ids.shape[0],segmentations.shape[1],segmentations.shape[2]),dim=0)

        for j in tqdm(range (self.iterative_num)):
            # print ('----------------')

            inputs = self.feature_extractor(images=image, segmentation_maps=segmentation_maps, return_tensors="pt")      
            inputs['pixel_values']=inputs['pixel_values'].to(self.device)
            inputs['pixel_mask']=inputs['pixel_mask'].to(self.device)
            class_num=inputs["class_labels"][0].shape[0]
            inputs["class_labels"][0]=torch.tensor(np.arange(0,class_num,1)).to(self.device)
            inputs["mask_labels"][0]=inputs["mask_labels"][0].to(self.device)

            outputs = self.model(inputs["pixel_values"],
                    class_labels=inputs["class_labels"],
                    mask_labels=inputs["mask_labels"], use_prompt=True, outer_prompt=None)

            print ('loss',outputs.loss)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        learnedd_prompt=torch.tensor(self.model.model.transformer_module.prompt_embeddings.data)
        learnedd_prompt=learnedd_prompt.cpu().detach()  #learnedd_prompt torch.Size([1, 256, 100])
        learnedd_prompt=learnedd_prompt.view(1,learnedd_prompt.shape[1]*learnedd_prompt.shape[2])

        inputs = self.feature_extractor(images=image, return_tensors="pt")   
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        image_features, pixel_embeddings=self.model.feature_extractor(inputs["pixel_values"])
        image_features=image_features.cpu().detach()
        pixel_embeddings=pixel_embeddings.detach()
        image_features=image_features.squeeze()
        image_features=image_features.view(image_features.shape[0],image_features.shape[1]*image_features.shape[2])
        image_features=torch.mean(image_features,dim=1, keepdim=True)
        image_features=image_features.permute(1,0) 

        return image_features, learnedd_prompt

    