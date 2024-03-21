import torch
import numpy as np
import io, tokenize
from augly.utils.base_paths import EMOJI_DIR
from PIL import Image,ImageDraw,ImageFont,ImageFilter


from engine.Gpt import get_eval
from engine.Crawl import  crawl_baidu 
from tools import clip
import yaml

from tools.FaceDet import FaceDetInterpreter
from tools.LOC import Loc2Interpreter

all_updated_model_config_path='configs/all_updated_model_config.yaml'
all_model_config=  yaml.load(open(all_updated_model_config_path, 'r'), Loader=yaml.Loader)


def mask_image(img,mask):
    mask = np.tile(mask[:,:,np.newaxis],(1,1,3))
    img = np.array(img).astype(float)
    img = np.array(mask*img).astype(np.uint8)
    return Image.fromarray(img)

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


    

class ClassifyInterpreter():
    step_name = 'CLASSIFY'

    def __init__(self,llama_generator):
        print(f'Registering {self.step_name} step')
        self.config = all_model_config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model, self.processor = clip.load(self.config["CLASSIFY"]['init']['clip_pretrained'], device=self.device, prompt_num=self.config["CLASSIFY"]['init']['prompt_num'], jit=False)
        self.model.eval()

        self.feature_pool = []
        self.prompt_pool = []
        self.name_pool = []

        self.is_face=False

        self.llama_generator=llama_generator
        self.max_gen_len= self.config["CLASSIFY"]['init']['max_gen_len']
        self.temperature= self.config["CLASSIFY"]['init']['temperature']
        self.top_p= self.config["CLASSIFY"]['init']['top_p']


        self.prompt_template_person = """Query: List three popular person names different from the person Barack Obama, separated by commas. Directly output their names without including any numbers or other content.
List:
Michelle Obama, Joseph Biden, Vladimir Putin

Query: List three popular person names different from the person Elon Musk, separated by commas. Directly output their names without including any numbers or other content.
List:
Gwyneth Paltrow, Gwynne Shotwell, JB Straubel

Query: List three popular person names different from the person {keyword}, separated by commas. Directly output their names without including any numbers or other content.
List:"""


        self.prompt_template_entity = """Query: List three entities that are in the same category as the entity Real Madrid CF but different from Real Madrid CF. Only their names, separated by a comma, in the format 'object1, object2, object2' without including any numbers or other content. If unable to find three, it's acceptable to have fewer than three, and please avoid any extra words.
List:
FC Barcelona, FC Bayern Munich, Manchester United FC

Query: List three entities that are in the same category as the entity apple but different from apple. Only their names, separated by a comma, in the format 'object1, object2, object2' without including any numbers or other content. If unable to find three, it's acceptable to have fewer than three, and please avoid any extra words.
List:
banana, watermelon, pear

Query: List three entities that are in the same category as the entity {keyword} but different from {keyword}. Only their names, separated by a comma, in the format 'object1, object2, object2' without including any numbers or other content. If unable to find three, it's acceptable to have fewer than three, and please avoid any extra words.
List:"""


    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        image_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        category_var = parse_result['args']['categories']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return image_var,obj_var,category_var,output_var

    def calculate_sim(self,inputs):
        img_feats = self.model.get_image_features(inputs['pixel_values'])
        text_feats = self.model.get_text_features(inputs['input_ids'])
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(img_feats,text_feats.t())

    def inference(self, label, inference_image, neg_samples, use_prompt, outer_prompt=None):
        self.model.eval()
        classes = [label]
        texts = [f"a photo of {label}"]
        
        images_ori = inference_image
        if len(neg_samples) > 0:
            neg_images = neg_samples
            neg_texts = [f"a photo of not {label}"] * len(neg_images)
            images_ori = images_ori + neg_images
    
        images = torch.stack([self.processor(image).to(self.device) for image in images_ori], dim=0)
        texts = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(images, use_prompt=use_prompt, outer_prompt=outer_prompt)
            text_features = self.model.encode_text(texts, use_prompt=False)
                
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
      
        similarity = torch.matmul(text_features,image_features.t())
        return images_ori, similarity, classes


    def predict(self, queries, images):
        all_similarity = []
        print ('queries',queries)
        print ('CLASSIFY name_pool',self.name_pool)
        print ('num of images', len(images))
        for q in queries: 
            if q in self.name_pool:
                indices = [i for i, name in enumerate(self.name_pool) if name == q]
                image_prompt_pool = [self.prompt_pool[i] for i in indices]
                image_feature_pool = [self.feature_pool[i] for i in indices]
                image_prompt_pool  = torch.stack(image_prompt_pool, dim=0).squeeze()
                image_feature_pool = torch.stack(image_feature_pool, dim=0).squeeze()
                image_feature_pool=image_feature_pool.view(-1,image_feature_pool.shape[-1])
                new_prompt_list = []
                sim_list = []
                for image in images:
                    image = torch.stack([self.processor(image).to(self.device)], dim=0)
                    with torch.no_grad():
                        image_feature = self.model.encode_image(image, use_prompt=False)
                    img_sim = self.cal_image_sim2(image_feature, image_feature_pool)
                    print ('img_sim',img_sim.shape)
                    print ('img_sim',img_sim)
                    sim_list.append(img_sim)

                concatenated_tensor = torch.cat(sim_list, dim=0)

                print ('concatenated_tensor',concatenated_tensor.shape)
                sum_concatenated_tensor = torch.sum(concatenated_tensor, dim=1)
                _,index=torch.max(sum_concatenated_tensor,0)
                result_tensor=concatenated_tensor[index,:]
                print ('similarity index',index)

                norm_sim = result_tensor / result_tensor.sum() ### 改成softmax?
                new_prompt = torch.sum(image_prompt_pool * norm_sim.view(-1, 1, 1, 1), dim=0)
                
                ori_images, similarity, classes = self.inference(label=q, inference_image=images, neg_samples=[], use_prompt=True, outer_prompt=new_prompt)
                print ('after similarity', similarity)
                all_similarity.append(similarity)
            else:
                ori_images, similarity, classes = self.inference(label=q, inference_image=images, neg_samples=[], use_prompt=False)
                print ('before similarity', similarity)
                all_similarity.append(similarity)
        all_similarity=torch.concat(all_similarity,dim=0)
        all_similarity=all_similarity.t()
        return all_similarity

    def cal_image_sim2(self, image_feature, img_feature_pool):

        image_feature_norm = image_feature/ image_feature.norm(p=2, dim=-1, keepdim=True)
        img_feature_pool_norm = img_feature_pool / img_feature_pool.norm(p=2, dim=-1, keepdim=True)    
        similarities = torch.mm(image_feature_norm.view(1,-1),img_feature_pool_norm.t())

        return similarities


    def query_obj(self,query,objs,img):
        if len(objs)==0:
            images = [img]
            return []
        else:
            images = [img.crop(obj['box']) for obj in objs]

        for obj in objs:
            print ("obj['box']",obj['box'])

        with torch.no_grad():
            sim=self.predict(query, images)

            print ('sim',sim.shape)
            print ('sim',sim)
            
        
        if len(query)==1:
            scores = sim.cpu().numpy()
            obj_ids = scores.argmax(0)
            obj = objs[obj_ids[0]]
            obj['class']=query[0]
            obj['class_score'] = 100.0*scores[obj_ids[0],0]
            return [obj]

        scores = sim.cpu().numpy()
        cat_ids = scores.argmax(1)
        for i,(obj,cat_id) in enumerate(zip(objs,cat_ids)):
            class_name = query[cat_id]
            class_score = scores[i,cat_id]
            obj['class'] = class_name #+ f'({score_str})'
            obj['class_score'] = round(class_score*100,1)

        objs = sorted(objs,key=lambda x: x['class_score'],reverse=True)
        objs = [obj for obj in objs if 'class' in obj]
        classes = set([obj['class'] for obj in objs])
        new_objs = []
        for class_name in classes:
            cls_objs = [obj for obj in objs if obj['class']==class_name]

            max_score = 0
            max_obj = None
            for obj in cls_objs:
                if obj['class_score'] > max_score:
                    max_obj = obj
                    max_score = obj['class_score']

            new_objs.append(max_obj)

        return new_objs




    def execute(self,prog_step,is_face=False):
        self.is_face = is_face
        image_var,obj_var,category_var,output_var = self.parse(prog_step)
        img = prog_step.state[image_var]
        objs = prog_step.state[obj_var]
        if category_var in prog_step.state.keys():
            cats = prog_step.state[category_var]
        else:
            cats=[category_var]        
        objs = self.query_obj(cats, objs, img)
        prog_step.state[output_var] = objs

        return objs




    def data_process(self, images, is_mask=False):
        image_processed = []
        for image in images:
            image_ori = Image.open(image['image_path']).convert("RGB")
            if self.is_face:
                for i, face in enumerate(image['faces']):
                    image_proc = mask_image(image_ori, face['mask']).crop(face['box'])
                    image_proc.save(f"{image['image_path'][:-4]}_mask{i}.jpg")
                    image_processed.append(image_proc)
            else:
                for i, obj in enumerate(image['objs']):
                    image_proc = image_ori.crop(obj)
                    image_proc.save(f"{image['image_path'][:-4]}_obj{i}.jpg")
                    image_processed.append(image_proc)
                
        return image_processed

    def get_data(self, keyword, category_name):
        ori_keyword=keyword
        search_category_name=category_name.lstrip("'").rstrip("'")
        pic_num = self.config["CLASSIFY"]['get_data']['pic_num']
        val_num = self.config["CLASSIFY"]['get_data']['val_num']
        images_path = self.config["CLASSIFY"]['get_data']['images_path']
        if category_name!='':
            keyword=keyword+'_'+search_category_name
        url = self.config["CLASSIFY"]['get_data']['url_start'] + keyword + self.config["CLASSIFY"]['get_data']['url_end']
        train_images_list = crawl_baidu(url, images_path, keyword, pic_num+val_num, self.is_face, FaceDetInterpreter(), Loc2Interpreter(),category_name=search_category_name)
    
        if self.is_face == True:
            print ('ask person name')
            prompt = self.prompt_template_person.format(keyword=ori_keyword)
        else:
            print ('ask object name')
            prompt = self.prompt_template_entity.format(keyword=ori_keyword)
            
        answer = get_eval(prompt,dict(model=self.llama_generator,max_gen_len=self.max_gen_len,temperature=self.temperature,top_p=self.top_p))
        names_list = answer.split(", ")
        names_list=list(set(names_list))
        print('----names_list----',names_list)
        
        print("-----------------------------------downloading negtive samples")
        neg_num=self.config["CLASSIFY"]['get_data']['neg_num']
        images_path = self.config["CLASSIFY"]['get_data']['images_path_2']
        neg_samples_list = []
        for name in names_list:
            if category_name!='':
                name=name+'_'+search_category_name            
            url = self.config["CLASSIFY"]['get_data']['url_start']+name+self.config["CLASSIFY"]['get_data']['url_end']
            neg_samples = crawl_baidu(url, images_path, name, neg_num, self.is_face, FaceDetInterpreter(), Loc2Interpreter(),category_name=search_category_name)
            neg_samples_list.extend(neg_samples)
        
        train_images_list = self.data_process(train_images_list, is_mask=False)
        neg_samples_list = self.data_process(neg_samples_list, is_mask=False)

        return train_images_list, neg_samples_list 
    


    # update prompt
    def update(self, query_list, category_name):
        steps = self.config["CLASSIFY"]['update']['steps']
        given_lr = float(self.config["CLASSIFY"]['update']['given_lr'])
        stop_value = self.config["CLASSIFY"]['update']['stop_value']
        print ('query_list in update', query_list)
        print ('self.name_pool in update', self.name_pool)

        for query in query_list:

            if query in self.name_pool:
                print ('the concept' + query + 'has been learned')
            else:
                if len (query) > 100:
                    continue
                train_images_list, neg_samples_list = self.get_data(query, category_name)
                print("-----------------------------------learning prompt")
                image_feature_pool = []
                image_prompt_pool = []
                for train_image in train_images_list[1:]:
                    prompt, flag, image_feature, loss = self.train_prompt(steps=steps, label=query, train_images=[train_image], val_images=[train_images_list[1]], neg_samples=neg_samples_list, given_lr=given_lr, stop_value=stop_value)
                    if flag == 1 :
                        self.feature_pool.append(image_feature)
                        self.prompt_pool.append(prompt)
                        self.name_pool.append(query)
                        
                self.model.visual.init_prompt()


    def train_prompt(self, steps, label, train_images, val_images, neg_samples, given_lr, stop_value):
        pos_images = train_images
        pos_texts = [f"a photo of {label}"] * len(train_images)
        
        neg_images = neg_samples
        neg_texts = [f"a photo of not {label}"] * len(neg_images)
            
        images = pos_images + neg_images 
        texts =  pos_texts + neg_texts 
            
        images = torch.stack([self.processor(image).to(self.device) for image in images], dim=0)
        texts = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            image_feature = self.model.encode_image(torch.stack([self.processor(image).to(self.device) for image in pos_images], dim=0), use_prompt=False) # 1
        
        
        optimizer = torch.optim.Adam([self.model.visual.vision_prompt], lr=given_lr, betas=tuple(self.config["CLASSIFY"]['train_prompt']['optimizer']['betas']), eps=float(self.config["CLASSIFY"]['train_prompt']['optimizer']['eps']), weight_decay=self.config["CLASSIFY"]['train_prompt']['optimizer']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config["CLASSIFY"]['train_prompt']['scheduler']['step_size'], gamma=self.config["CLASSIFY"]['train_prompt']['scheduler']['gamma'])
        
        loss_img = torch.nn.CrossEntropyLoss()
        loss_txt = torch.nn.CrossEntropyLoss()
        
        flag = 0
        value0 = 0
        for i in range(steps):
        
            self.model.eval() 
            logits_per_image, logits_per_text = self.model(images, texts, use_prompt=True)
            ground_truth = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=self.device)
            
            loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
                
            print ('i', i, 'prompt loss', loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()      
            scheduler.step()

            ori_images, similarity, classes = self.inference(label=label, inference_image=[val_images[0]], neg_samples=neg_samples, use_prompt=True)
            predict, value, classes, similarity = self.select_class(ori_images, similarity, classes)

            if value >= value0:
                value0 = value
            if predict == 0 and value > stop_value:
                print("Stop at step: ", i)
                flag = 1
                break
        prompt = torch.tensor(self.model.visual.vision_prompt.data)
        self.model.visual.init_prompt()
        return prompt, flag, image_feature, loss


    def select_class(self, ori_images, similarity, classes):
        values_cls, indices_cls = similarity[0].topk(1)
        values_label = similarity[0][0]
        return indices_cls.item(), values_label.item(), classes, similarity




