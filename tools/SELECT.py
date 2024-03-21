import torch
import io, tokenize
from augly.utils.base_paths import EMOJI_DIR
from PIL import Image,ImageDraw,ImageFont,ImageFilter

# import ruamel.yaml as yaml
from tools import clip
import yaml
import numpy as np
from engine.Gpt import get_eval
from engine.Crawl import crawl_google, crawl_baidu

import torch.utils.checkpoint

all_updated_model_config_path='configs/all_updated_model_config.yaml'
all_model_config=  yaml.load(open(all_updated_model_config_path, 'r'), Loader=yaml.Loader)

###
from tools.SEG import SegmentInterpreter
from tools.FaceDet import FaceDetInterpreter


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




class SelectInterpreter():
    step_name = 'SELECT'

    def __init__(self,llama_generator):
        self.is_face = None ####
        print(f'Registering {self.step_name} step')
        self.config = all_model_config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.model = CLIPModel.from_pretrained(
        #     "openai/clip-vit-large-patch14").to(self.device)
        # self.model.eval()
        # self.processor = CLIPProcessor.from_pretrained(
        #     "openai/clip-vit-large-patch14")
        
        self.model, self.processor = clip.load(self.config["SELECT"]['init']['clip_pretrained'], device=self.device, prompt_num=self.config["SELECT"]['init']['prompt_num'], jit=False)
        self.model.eval()
        self.face_dec = FaceDetInterpreter()
        self.seg= SegmentInterpreter()
        self.feature_pool = []
        self.prompt_pool = []
        self.name_pool = []

        self.llama_generator=llama_generator
        self.max_gen_len = self.config["SELECT"]['init']['max_gen_len']
        self.temperature = self.config["SELECT"]['init']['temperature']
        self.top_p= self.config["SELECT"]['init']['top_p']

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
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        query = eval(parse_result['args']['query']).split(',')
        category = eval(parse_result['args']['category'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,query,category,output_var

    # def calculate_sim(self,inputs):
    #     img_feats = self.model.get_image_features(inputs['pixel_values'])
    #     text_feats = self.model.get_text_features(inputs['input_ids'])
    #     img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
    #     text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
    #     return torch.matmul(img_feats,text_feats.t())

    def query_obj(self,query,objs,img):
        # images = [img.crop(obj['box']) for obj in objs]
        # images = vis_masks(image_ori, face['mask']).crop(face['box'])
        images = [mask_image(img, obj['mask']).crop(obj['box']) for obj in objs]
        # text = [f'a photo of {q}' for q in query]
        # inputs = self.processor(
        #     text=text, images=images, return_tensors="pt", padding=True)
        # inputs = {k:v.to(self.device) for k,v in inputs.items()}
        # with torch.no_grad():
        #     scores = self.calculate_sim(inputs).cpu().numpy()
        obj_ids = self.predict(query, images)
        # obj_ids = scores.argmax(0)
        return [objs[i] for i in obj_ids]


    def query_string_match(self,objs,q):
        obj_cats = [obj['category'] for obj in objs]
        q = q.lower()
        for cat in [q,f'{q}-merged',f'{q}-other-merged']:
            if cat in obj_cats:
                return [obj for obj in objs if obj['category']==cat]
        
        return None

    def execute(self,prog_step,is_face=True):
        self.is_face = is_face
        img_var,obj_var,query,category,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        select_objs = []

        if category is not None:
            cat_objs = [obj for obj in objs if obj['category'] in category]
            if len(cat_objs) > 0:
                objs = cat_objs


        if category is None:
            for q in query:
                matches = self.query_string_match(objs, q)
                if matches is None:
                    continue
                
                select_objs += matches

        # if query is not None and len(select_objs)==0:
        if len(query) > len(select_objs):
            select_objs = self.query_obj(query, objs, img)
        prog_step.state[output_var] = select_objs

        return select_objs
    
    # def execute(self,prog_step,inspect=False):
    #     img_var,obj_var,query,category,output_var = self.parse(prog_step)
    #     img = prog_step.state[img_var]
    #     objs = prog_step.state[obj_var]
    #     select_objs = []

    #     if category is not None:
    #         cat_objs = [obj for obj in objs if obj['category'] in category]
    #         if len(cat_objs) > 0:
    #             objs = cat_objs


    #     if category is None:
    #         for q in query:
    #             matches = self.query_string_match(objs, q)
    #             if matches is None:
    #                 continue
                
    #             select_objs += matches

    #     if query is not None and len(select_objs)==0:
    #         select_objs = self.query_obj(query, objs, img)

    #     prog_step.state[output_var] = select_objs
    #     if inspect:
    #         select_obj_img = vis_masks(img, select_objs)
    #         html_str = self.html(img_var, obj_var, query, category, output_var, select_obj_img)
    #         return select_objs, html_str

    #     return select_objs
    
    
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
                
            
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
      
        similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
        return images_ori, similarity, classes
    
    def cal_image_sim2(self, image_feature, img_feature_pool):


        image_feature_norm = image_feature/ image_feature.norm(p=2, dim=-1, keepdim=True)
        img_feature_pool_norm = img_feature_pool / img_feature_pool.norm(p=2, dim=-1, keepdim=True)    
        similarities = torch.mm(image_feature_norm.view(1,-1),img_feature_pool_norm.t())

        return similarities


    def generate_prompt2(self, sim_list):
        max_value = None
        max_position = None

        # 遍历主列表和子列表
        for i, sublist in enumerate(sim_list):
            for j, value in enumerate(sublist):
                # 如果找到新的最大值
                if max_value is None or value > max_value:
                    max_value = value
                    max_position = (i, j)
        return max_value, max_position
    
    def predict(self, queries, images):
        results = []
        print ('queries',queries)
        print ('self.name_pool',self.name_pool)
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
                predict, value, classes, similarity = self.select_class(ori_images, similarity, classes)
                print ('after similarity', similarity)
                results.append(predict)
            else:
                ori_images, similarity, classes = self.inference(label=q, inference_image=images, neg_samples=[], use_prompt=True, outer_prompt=None)
                print ('before similarity', similarity)
                predict, value, classes, similarity = self.select_class(ori_images, similarity, classes)
                results.append(predict)
        return results

            
            
    def select_class(self, ori_images, similarity, classes):
        values_cls, indices_cls = similarity[0].topk(1)
        values_label = similarity[0][0]
        return indices_cls.item(), values_label.item(), classes, similarity
    
    def cal_image_sim(self, image_feature, img_feature_pool):
        distances = torch.norm(img_feature_pool - image_feature, dim=1)
        similarities = 1.0 / (distances)
        normalized_similarities = similarities / similarities.sum()
        
        return normalized_similarities
    
    def generate_prompt(self, image, img_feature_pool, img_prompt_pool):
        image = torch.stack([self.processor(image).to(self.device)], dim=0)
        with torch.no_grad():
            image_feature = self.model.encode_image(image, use_prompt=False)

        normalized_similarities = self.cal_image_sim(image_feature, img_feature_pool)
        new_prompt = torch.sum(img_prompt_pool * normalized_similarities.view(-1, 1, 1, 1), dim=0)

        return new_prompt

    def get_faces(self, image_path):
        faces = self.face_dec.det_face(Image.open(image_path).convert('RGB'))
        images = {}
        images['image_path'] = image_path
        images['faces'] = faces
        return images
    def get_objs(self, image_path):
        objs = self.seg.predict(Image.open(image_path).convert('RGB'))
        images = {}
        images['image_path'] = image_path
        images['objs'] = objs
        return images
    
    def data_process(self, images, is_mask=False):
        image_processed = []
        for image in images:
            image_ori = Image.open(image['image_path']).convert("RGB")
            if self.is_face:
                # one_face
                for i, face in enumerate(image['faces']):
                    image_proc = mask_image(image_ori, face['mask']).crop(face['box'])
                    # image_pil = image_ori.crop(face["box"])
                    image_proc.save(f"{image['image_path'][:-4]}_mask{i}.jpg")
                    image_processed.append(image_proc)
            
            else:
                # 返回crop mask后的物体
                if is_mask:
                    # ## mask & crop
                    for i, obj in enumerate(image['objs']):
                        image_proc = mask_image(image_ori, obj['mask']).crop(obj['box'])
                        image_proc.save(f"{image['image_path'][:-4]}_mask{i}.jpg")
                        image_processed.append(image_proc)
                else:
                    # no mask only crop
                    image_proc = Image.open(image['image_path']).convert("RGB")
                    image_processed.append(image_proc)
                
        return image_processed

    def get_data(self, keyword):
        # train val
        pic_num = self.config["SELECT"]['get_data']['pic_num']
        val_num = self.config["SELECT"]['get_data']['val_num']
        images_path = self.config["SELECT"]['get_data']['images_path']
        url = self.config["SELECT"]['get_data']['url_start'] + keyword +  self.config["SELECT"]['get_data']['url_end']
        train_images_list = crawl_baidu(url, images_path, keyword, pic_num+val_num, self.is_face, FaceDetInterpreter(), SegmentInterpreter())


        if self.is_face == True:
            print ('ask person name')
            prompt = self.prompt_template_person.format(keyword=keyword)
        else:
            print ('ask object name')
            prompt = self.prompt_template_entity.format(keyword=keyword)

        answer = get_eval(prompt,dict(model=self.llama_generator,max_gen_len=self.max_gen_len,temperature=self.temperature,top_p=self.top_p))
        names_list = answer.split(", ")
        print(names_list)
        
        print("-----------------------------------downloading negtive samples")
        # neg
        neg_num=  self.config["SELECT"]['get_data']['neg_num']
        images_path = self.config["SELECT"]['get_data']['images_path_2'] 
        neg_samples_list = []
        for name in names_list:
            url = self.config["SELECT"]['get_data']['url_start'] + name +  self.config["SELECT"]['get_data']['url_end']
            neg_samples = crawl_baidu(url, images_path, name, neg_num, self.is_face, FaceDetInterpreter(), SegmentInterpreter())
            neg_samples_list.extend(neg_samples)
        
        train_images_list = self.data_process(train_images_list, is_mask=False)
        neg_samples_list = self.data_process(neg_samples_list, is_mask=False)

        return train_images_list, neg_samples_list 
    
    # update prompt
    def update(self, query):
        steps = self.config["SELECT"]['update']['steps']

        if len (query) < 100:        
            train_images_list, neg_samples_list = self.get_data(query)
            print("-----------------------------------learning prompt")
            image_feature_pool = []
            image_prompt_pool = []
            for train_image in train_images_list[1:]:
                prompt, flag, image_feature, loss = self.train_prompt(steps=steps, label=query, train_images=[train_image], val_images=[train_images_list[1]], neg_samples=neg_samples_list)
                if flag == 1 :
                    self.feature_pool.append(image_feature)
                    self.prompt_pool.append(prompt)
                    self.name_pool.append(query)
                    
            self.model.visual.init_prompt()
        

    def train_prompt(self, steps, label, train_images, val_images, neg_samples):
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
        

        optimizer = torch.optim.Adam([self.model.visual.vision_prompt], lr=float(self.config["SELECT"]['train_prompt']['optimizer']['lr']), 
                                     betas=tuple(self.config["SELECT"]['train_prompt']['optimizer']['betas']), eps=float(self.config["SELECT"]['train_prompt']['optimizer']['eps']), 
                                     weight_decay=self.config["SELECT"]['train_prompt']['optimizer']['weight_decay'])
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= self.config["SELECT"]['train_prompt']['scheduler']['step_size'],
                                                     gamma=self.config["SELECT"]['train_prompt']['scheduler']['gamma'])
        
        loss_img = torch.nn.CrossEntropyLoss()
        loss_txt = torch.nn.CrossEntropyLoss()

        
        flag = 0
        value0 = 0
        for i in range(steps):
        
            self.model.eval() 
            logits_per_image, logits_per_text = self.model(images, texts, use_prompt=True)
            ground_truth = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=self.device)
            
            loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            print ('loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()      
            scheduler.step()

            ori_images, similarity, classes = self.inference(label=label, inference_image=[val_images[0]], neg_samples=neg_samples, use_prompt=True)
            predict, value, classes, similarity = self.select_class(ori_images, similarity, classes)

            if value >= value0:
                value0 = value
            print ('value', value)
            if predict == 0 and value > self.config["SELECT"]['train_prompt']['optimizer']['threshold']:
                print("Stop at step: ", i)
                print(similarity)
                flag = 1
                break
        prompt = torch.tensor(self.model.visual.vision_prompt.data)
        self.model.visual.init_prompt()
        return prompt, flag, image_feature, loss

      