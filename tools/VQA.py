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
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from transformers import (ViltProcessor, ViltForQuestionAnswering, 
    OwlViTProcessor, OwlViTForObjectDetection,
    MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation,
    CLIPProcessor, CLIPModel, AutoProcessor, BlipForQuestionAnswering)
from diffusers import StableDiffusionInpaintPipeline
import ruamel.yaml as yaml


from engine.nms import nms

####  blip_vqa 

from tools.blip_vqa.blip_vqa import blip_vqa
from tools.blip_vqa.prompt_generation import Prompt_Generation_Model

all_updated_model_config_path='configs/all_updated_model_config.yaml'
all_model_config= yaml.load(open(all_updated_model_config_path, 'r'), Loader=yaml.Loader)


from engine.data_utils import pre_question
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode



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




class VQAInterpreter():
    step_name = 'VQA'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.config = all_model_config

        self.model = blip_vqa(allconfig=self.config, pretrained=self.config['VQA']['init']['pretrained'], image_size=self.config['VQA']['init']['image_size'], 
                        vit=self.config['VQA']['init']['vit'], vit_grad_ckpt=self.config['VQA']['init']['vit_grad_ckpt'], 
                        vit_ckpt_layer=self.config['VQA']['init']['vit_ckpt_layer'], init_tokenizer_path = self.config['VQA']['init']['init_tokenizer_path'])
        self.model.eval()
        self.model = self.model.to(self.device) 
        self.prompt_generate_model=Prompt_Generation_Model(n_head=self.config['VQA']['init']['n_head'], d_model=2*self.config['VQA']['feature_dim'], d_k=2*self.config['VQA']['feature_dim'], d_v=self.config['VQA']['prompt_num']*self.config['VQA']['feature_dim'])
        self.prompt_generate_model.eval()
        self.prompt_generate_model = self.prompt_generate_model.to(self.device)  

        self.exist_feature=[]
        self.exist_prompt=[]
        self.store_num=self.config['VQA']['init']['store_num']
        self.transform = transforms.Compose([
            transforms.Resize((self.config['VQA']['init']['resize'], self.config['VQA']['init']['resize']),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(tuple(self.config['VQA']['init']['normalization_parameters']['mean']), tuple(self.config['VQA']['init']['normalization_parameters']['std_dev'])),
            ]) 

        self.temporal_q=[]
        self.temporal_v=[]
        self.temporal_a=[]


    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        args = parse_result['args']
        img_var = args['image']
        question = eval(args['question'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,question,output_var

    def prompt_generate(self,img,question):

        exist_feature_tensor=torch.concat(self.exist_feature,dim=0)
        exist_prompt_tensor=torch.concat(self.exist_prompt,dim=0) 
        exist_feature_tensor=exist_feature_tensor.to(self.device)
        exist_prompt_tensor=exist_prompt_tensor.to(self.device)
        exist_feature_tensor=exist_feature_tensor.unsqueeze(0)
        exist_prompt_tensor=exist_prompt_tensor.unsqueeze(0)

        self.prompt_generate_model.eval()
        self.model.eval()

        result_before, visual_feature, question_states = self.model(img, question, train=False, inference='generate', prompt_use=False) 

        visual_feature,_=torch.max(visual_feature,dim=1)
        question_states,_=torch.max(question_states,dim=1)
        visual_feature=visual_feature/torch.norm(visual_feature)
        question_states=question_states/torch.norm(question_states)

        this_feature=torch.cat((visual_feature, question_states), dim=1)
        this_feature=this_feature.unsqueeze(0)

        this_prompt=self.prompt_generate_model(this_feature, exist_feature_tensor, exist_prompt_tensor)

        this_prompt=this_prompt.view(self.config['VQA']['prompt_num'], self.config['VQA']['feature_dim'])

        return this_prompt


    def predict(self,img,question):

        self.model.eval()

        img = self.transform(img)
        img = img.to(self.device,non_blocking=True)
        img = img.unsqueeze(0)

        question = pre_question(question)
        question_list=[]
        question_list.append(question)

        if len (self.exist_feature) < self.store_num:
            outputs, _, _ = self.model(img, question_list, train=False, inference='generate', prompt_use=False) 
        else:
            this_prompt=self.prompt_generate(img, question_list)
            outputs, _, _ = self.model(img, question_list, train=False, inference='generate', prompt_use=True, outer_prompt=this_prompt)   
        
        return outputs[0]


    def question_arg(self,question,prog_step):
        temp_question=question
        while (temp_question.find('{')>0 and temp_question.find('}')>0):
            left_index=temp_question.find('{')
            right_index=temp_question.find('}')

            left=temp_question[:left_index]
            right=temp_question[right_index+1:]
            arg=temp_question[left_index+1:right_index]
            middle=prog_step.state[arg]

            temp_question=left+middle+right

        return temp_question


    def execute(self,prog_step):
        img_var,question,output_var = self.parse(prog_step)
        if question.find('{')>0 and question.find('}')>0:
            question=self.question_arg(question,prog_step)
        img = prog_step.state[img_var]
        answer = self.predict(img,question)
        prog_step.state[output_var] = answer

        return answer


    def prompt_train(self, image, question, answer):
        i=0
        max_step= self.config['VQA']['prompt_train']['max_step']
        correct_flag=1

        to_optim =  [
                    {'params':self.model.text_decoder.bert.prompt,'lr':self.config['VQA']['prompt_train']['init_lr'],'weight_decay':self.config['VQA']['weight_decay']},
                    ]
        optimizer = torch.optim.AdamW(to_optim)


        while (correct_flag==1 and i <max_step):

            self.model.eval()  
            loss = self.model(image, question, answer, train=True, n=None, prompt_use=True)   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

            result=self.prompt_train_evaluation(image, question)
            
            if result == answer:
                correct_flag=0

            i=i+1
        
        prompt=torch.tensor(self.model.text_decoder.bert.prompt.data)
        self.model.init_prompt()
        return prompt, correct_flag

    def prompt_train_evaluation(self, image, question):

            self.model.eval()  
            result, _, _ = self.model(image, question, train=False, inference='generate', prompt_use=True) 

            return result

    def update(self, correct, data):
        ##### save prompt and features into experience pool
        print ('-----------------now, correct is', correct)
        question=data['question']
        image=data['image'] 
        answer=data['answer'] 

        image = self.transform(image)
        image = image.to(self.device,non_blocking=True)
        image = image.unsqueeze(0)

        question = pre_question(question)
        question_list=[]
        question_list.append(question)

        _, visual_feature, question_states = self.model(image, question_list, train=False, inference='generate', prompt_use=False) 
        visual_feature,_=torch.max(visual_feature,dim=1)
        question_states,_=torch.max(question_states,dim=1)
        visual_feature=visual_feature/torch.norm(visual_feature)
        question_states=question_states/torch.norm(question_states)
        this_feature=torch.cat((visual_feature, question_states), dim=1)  


        if correct:
            if len (self.exist_feature) < self.store_num:
                self.exist_feature.append(this_feature.detach())
                self.exist_prompt.append(torch.zeros(1, self.config['VQA']['prompt_num']*self.config['VQA']['feature_dim']).to(self.device))
            else:
                exist_feature_tensor=torch.concat(self.exist_feature,dim=0)
                exist_prompt_tensor=torch.concat(self.exist_prompt,dim=0) 
                exist_feature_tensor=exist_feature_tensor.to(self.device)
                exist_prompt_tensor=exist_prompt_tensor.to(self.device)
                exist_feature_tensor=exist_feature_tensor.unsqueeze(0).detach()
                exist_prompt_tensor=exist_prompt_tensor.unsqueeze(0).detach()

                this_feature=this_feature.unsqueeze(0)
                self.prompt_generate_model.eval()
                this_prompt=self.prompt_generate_model(this_feature, exist_feature_tensor, exist_prompt_tensor)
                this_prompt=this_prompt.view(self.config['VQA']['prompt_num'], self.config['VQA']['feature_dim'])  
                self.exist_feature.append(this_feature.squeeze(0).detach())
                self.exist_prompt.append(this_prompt.detach().view(1,self.config['VQA']['prompt_num']*self.config['VQA']['feature_dim']).to(self.device)) 
        else:
            prompt, correct_flag=self.prompt_train(image, question_list, answer)
            if correct_flag==0:
                if len (self.exist_feature) < self.store_num:
                    self.exist_feature.append(this_feature.detach())
                    self.exist_prompt.append(prompt.detach().view(1,self.config['VQA']['prompt_num']*self.config['VQA']['feature_dim']))
                else:
                    self.exist_feature.append(this_feature.detach())
                    self.exist_prompt.append(prompt.detach().view(1,self.config['VQA']['prompt_num']*self.config['VQA']['feature_dim']))

        print ('size of self.exist_feature:', len(self.exist_feature))
        print ('size of self.exist_prompt:', len(self.exist_prompt))


