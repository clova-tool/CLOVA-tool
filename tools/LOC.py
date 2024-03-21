

import os
import torch
import io, tokenize
from augly.utils.base_paths import EMOJI_DIR
import augly.image as imaugs
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import ruamel.yaml as yaml


from engine.nms import nms


#############################################file of LOC
import torch
from transformers_self.src.transformers.models.owlvit.modeling_owlvit import OwlViTForObjectDetection
from transformers.models.owlvit.processing_owlvit import OwlViTProcessor
from transformers_self.util_LOC.get_LVIS import get_LVIS
import torch.nn.functional as F
import random
import transformers_self.util_LOC.transforms as T
from tqdm import tqdm
from transformers_self.src_LOC.matcher import build_matcher
from transformers_self.util_LOC.bulid import SetCriterion_early_exit
from transformers_self.util_LOC.bulid import coco_plus_resize
import math
import sys
import transformers_self.util_LOC.misc as utils
#############################################

all_updated_model_config_path='configs/all_updated_model_config.yaml'
all_model_config=  yaml.load(open(all_updated_model_config_path, 'r'), Loader=yaml.Loader)


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


class LocInterpreter():
    step_name = 'LOC'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.config = all_model_config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print ('self.device',self.device)
        ## Q3 model version
        self.processor = OwlViTProcessor.from_pretrained(
            self.config["LOC"]['init']['pretrained_processor'],)
        self.model = OwlViTForObjectDetection.from_pretrained(
            self.config["LOC"]['init']['pretrained_model']).to(self.device)
        self.model.eval()
        self.thresh = self.config["LOC"]['init']['thresh']
        self.nms_thresh = self.config["LOC"]['init']['nms_thresh']

        self.saved_prompt_dict = {}
        self.saved_image_dict = {}

        # seed = args.seed  Q1
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # random.seed(seed)


    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_name = eval(parse_result['args']['object'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_name,output_var
    
    def normalize_coord(self,bbox,img_size):
        w,h = img_size
        x1,y1,x2,y2 = [int(v) for v in bbox]
        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(x2,w-1)
        y2 = min(y2,h-1)
        return [x1,y1,x2,y2]

    def predict(self,img,obj_name):
        self.model.eval()
        encoding = self.processor(text=[[f'a photo of {obj_name}']], images=img, return_tensors="pt").to(self.device) #### [pixel_value] [1, 3, 768, 768] [input_ids] [1204, 16]
        encoding["prompt_use"] = False
        encoding["outer_prompt"] = "ori"
        with torch.no_grad():
            outputs = self.model(**encoding)
            for k,v in outputs.items():
                if v is not None:
                    outputs[k] = v.to('cpu') if isinstance(v, torch.Tensor) else v
            cur_image_feature = outputs.vision_model_output.last_hidden_state
            count_pseudo_ori = self.count_pseudo_right(outputs)
        target_sizes = torch.Tensor([img.size[::-1]])
        target_sizes=target_sizes.cpu()
        results = self.processor.post_process_object_detection(outputs=outputs,threshold=self.thresh,target_sizes=target_sizes)
        boxes, scores = results[0]["boxes"], results[0]["scores"]
        boxes = boxes.cpu().detach().numpy().tolist()
        scores = scores.cpu().detach().numpy().tolist()
        if len(boxes)> 0:
            boxes, scores = zip(*sorted(zip(boxes,scores),key=lambda x: x[1],reverse=True))
            selected_boxes = []
            selected_scores = []
            for i in range(len(scores)):
                if scores[i] > self.thresh:
                    coord = self.normalize_coord(boxes[i],img.size)
                    selected_boxes.append(coord)
                    selected_scores.append(scores[i])

            selected_boxes, selected_scores = nms(
                selected_boxes,selected_scores, self.nms_thresh)
            
            return selected_boxes
        else:
            if obj_name in self.saved_prompt_dict:
                save_prompt = self.saved_prompt_dict[obj_name]
                save_image = self.saved_image_dict[obj_name]
                pseudo_sel =[]
                pseudo_weight =[]
                inputs = self.processor(text=[[f'a photo of {obj_name}']], images=img, return_tensors="pt").to(self.device) #### [pixel_value] [1, 3, 768, 768] [input_ids] [1204, 16]
                save_prompt_tensor = torch.cat(save_prompt) ### [5, 768]
                inputs["prompt_use"] = True
                for i in range(len(save_prompt_tensor)):
                    inputs["outer_prompt"] = save_prompt_tensor[i].unsqueeze(0)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    count_pseudo_cur = self.count_pseudo_right(outputs)
                    if count_pseudo_cur >= 1 and count_pseudo_cur >= count_pseudo_ori:
                        pseudo_sel.append(True)
                        pseudo_weight.append(count_pseudo_cur)
                    else:
                        pseudo_sel.append(False)

                inputs = self.processor(text=[[f'a photo of {obj_name}']], images=img, return_tensors="pt").to(self.device) #### [pixel_value] [1, 3, 768, 768] [input_ids] [1204, 16]
                inputs["prompt_use"] = True
                pseudo_weight =torch.tensor(pseudo_weight)
                inputs["outer_prompt"], sim =  self.get_prompt_new(torch.cat(save_image)[pseudo_sel], save_prompt_tensor[pseudo_sel], cur_image_feature, pseudo_weight) #torch.mean(save_prompt_tensor).unsqueeze(0)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    for k,v in outputs.items():
                        if v is not None:
                            outputs[k] = v.to('cpu') if isinstance(v, torch.Tensor) else v
                    
                target_sizes = torch.Tensor([img.size[::-1]])
                target_sizes=target_sizes.cpu()
                results = self.processor.post_process_object_detection(outputs=outputs,threshold=self.thresh,target_sizes=target_sizes)
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
                        coord = self.normalize_coord(boxes[i],img.size)
                        selected_boxes.append(coord)
                        selected_scores.append(scores[i])

                selected_boxes, selected_scores = nms(
                    selected_boxes,selected_scores, 0.5)
                selected_boxes_ori = selected_boxes
                # #### <--用于可视化，可以注释掉-->
                # if True:
                #     from src.util import BoxUtil   
                #     import matplotlib.pyplot as plt    
                #     def normalize_coord_new(bbox,img_size):
                #         # w,h = img_size
                #         w, h = img_size
                #         boxes = torch.tensor(bbox) / torch.tensor([w, h, w, h], dtype=torch.float32)
                #         boxes = BoxUtil.box_convert(boxes, "xyxy", "cxcywh")
                #         return boxes
                    
                #     def show_nms(image, boxes, scores, class_name):
                #         input_image = np.asarray(image).astype(np.float32) / 255.0
                #         plot_nms(input_image, boxes, scores, class_name)

                #     def plot_nms(input_image, boxes, scores, class_name):
                #         fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                #         ax.imshow(input_image, extent=(0, 1, 1, 0))
                #         ax.set_axis_off()
                #         for box, label_data in zip(boxes, scores):
                #             box, score = box, label_data
                #             cx, cy, w, h = box.cpu().numpy()
                #             ax.plot([cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
                #                     [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], "r")
                #             ax.text(
                #                 cx - w / 2,
                #                 cy + h / 2 + 0.015,
                #                 f"{class_name}: {score:1.2f}",
                #                 ha="left",
                #                 va="top",
                #                 color="red",
                #                 bbox={
                #                     "facecolor": "white",
                #                     "edgecolor": "red",
                #                     "boxstyle": "square,pad=.3"
                #                 })
                #         fig.savefig(class_name + "_nms.jpg")
                #         plt.close()

                #     for i in range(len(selected_boxes)):
                #         selected_boxes[i] = normalize_coord_new(selected_boxes[i], img.size)
                #     show_nms(img, selected_boxes, selected_scores, class_name + "_softmax_nms_")
                # #### <--用于可视化，可以注释掉-->

                return selected_boxes_ori
             
            else:
                return []
            
    def get_prompt_new(self,save_image_tensor, save_prompt_tensor, cur_image_feature, pseudo_weight):
        save_image_tensor = save_image_tensor[:,0:1,:]
        cur_image_feature =cur_image_feature[:,0:1,:]
        if len(pseudo_weight) == 0:
            return torch.zeros(1, 1, 768).cuda(), None
        cur_image_feature = cur_image_feature.expand(len(save_image_tensor), -1, -1)
        euclidean_distances = torch.norm(cur_image_feature - save_image_tensor, dim=(1, 2))
        sim = F.softmax(-euclidean_distances / 5, dim=0) 
        sim = (sim + pseudo_weight.cuda() / torch.sum(pseudo_weight).cuda() ) / 2
        res  = torch.sum(sim.view(sim.shape[0], 1, 1) * save_prompt_tensor, dim =0).unsqueeze(0)
        return res, sim    


    def update(self, obj_name):
        select_number = self.config["LOC"]['update']['select_number']
        max_class_num = self.config["LOC"]['update']['max_class_num']
        epochs = self.config["LOC"]['update']['epochs']
        clip_max_norm = self.config["LOC"]['update']['clip_max_norm']
        category_dict, image_category_list, image_path, train_json = get_LVIS(path = self.config["LOC"]['update']['LVIS_path'])
        if obj_name in category_dict.keys(): ### Q2:同一个类别是否需要多次更新
            if obj_name in self.saved_prompt_dict.keys():
                print(obj_name + " already exits")
                return 
            class_name_list = []
            for cat in category_dict:
                cat = cat.replace("(", "")
                cat = cat.replace(")", "")
                if len(cat.split('_')) == 1:
                    class_name_list.append("a photo of " + str(cat))
                else:
                    cur = ""
                    for data in cat.split('_'):
                        cur = cur + " " + data
                    class_name_list.append("a photo of" + cur)
            print("LVIS dataset loaded")
        
            ## optimizer
            for name, parameter in self.model.named_parameters():
                if "prompt" in name:
                    continue
                parameter.requires_grad = False
            print("the following paramters are updated")
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    print(f"  {name}")
            optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config["LOC"]['update']['lr']), weight_decay=self.config["LOC"]['update']['weight_decay'])
            print("optimizer loaded")

            ## dataset
            given_category_id=category_dict[obj_name]
            give_category_data=image_category_list[given_category_id-1]  
            random.shuffle(give_category_data)
            sampled_data = give_category_data[:select_number]
            transforms_lvis =  self.make_lvis_transforms()

            ## loss
            matcher = build_matcher()
            losses = ['labels_lvis', 'boxes', 'cardinality']
            weight_dict = {'loss_ce': 1, 'loss_bbox': 1}
            weight_dict['loss_giou'] = 1
            criterion = SetCriterion_early_exit(num_classes = max_class_num, matcher = matcher, weight_dict = weight_dict, losses = losses, focal_alpha=0.3)
            ### 自己限定的，可修改，此为pos+neg class number, 不包括no object
            criterion.to(self.device)
            print("loss functions loaded")

            save_image = []
            save_prompt = []
            save_image_list = []
            weight_test = []
            num_0_exit = 0
            use_list = []
            save_pos = []
            print("start train")
            print(len(sampled_data))

            for num_data, data_epoch in tqdm(enumerate(sampled_data)):
                if data_epoch['path'] not in use_list:
                    use_list.append(data_epoch['path'])
                else:
                    continue
                losses = []
                self.model.owlvit.vision_model.init_prompt_embedding()
                ### train
                self.model.train()
                criterion.train()
                criterion.check_cls = False
                criterion.check_box = False

                this_image_path= [os.path.join(image_path, data_epoch['path'])]
                this_epoch_bbox = [data_epoch['bboxes']]

                images = [Image.open(img).convert("RGB") for img in this_image_path] ###[3, 334, 500]
                boxes = torch.tensor(this_epoch_bbox).to(self.device)  ## [1, 1, 4]
                images, boxes = coco_plus_resize(images, boxes, transforms_lvis)
                boxes = boxes.to(self.device).squeeze(0) ### relative xyxy
                if len(boxes.shape) == 1:
                    boxes = boxes.unsqueeze(0)
                if  len(set(data_epoch['category_ids'])) == 1:
                    indd = torch.tensor(data_epoch['category_ids']) == given_category_id
                    pos_label = torch.tensor(data_epoch['category_ids'])[indd].tolist()
                    boxes = boxes[indd]
                    if len(set(pos_label)) + len(set(data_epoch['neg_category_ids'])) >= 15:
                        all_label = set(pos_label + random.sample(data_epoch['neg_category_ids'], 14))
                        # raise ValueError("sum of positve and negative are above 60")
                    else:
                        all_label = set(pos_label + data_epoch['neg_category_ids'])
                        ### pseudo neg
                        rest_label = [cat for cat in range(len(category_dict)) if cat not in all_label]
                        random.shuffle(rest_label)
                        all_label = set(list(all_label) + rest_label[:15 - len(all_label)])
                    assert len(all_label) == 15
                else:

                    indd = torch.tensor(data_epoch['category_ids']) == given_category_id
                    rest_pos = set(data_epoch['category_ids'])
                    sel_other_pos = given_category_id
                    for i in range(min(len(rest_pos)-1, 1)):
                        rest_pos.remove(sel_other_pos)
                        sel_other_pos = random.choice(list(rest_pos))
                        indd = [i or j for i, j in zip(indd, torch.tensor(data_epoch['category_ids']) == sel_other_pos)]
                        indd = torch.tensor(indd)


                    pos_label = torch.tensor(data_epoch['category_ids'])[indd].tolist()
                    boxes = boxes[indd]
                    if len(set(pos_label)) + len(set(data_epoch['neg_category_ids'])) >= 15:
                        all_label = set(pos_label + random.sample(data_epoch['neg_category_ids'], 15 - len(set(pos_label))))
                        # raise ValueError("sum of positve and negative are above 60")
                    else:
                        all_label = set(pos_label + data_epoch['neg_category_ids'])
                        ### pseudo neg
                        rest_label = [cat for cat in range(len(category_dict)) if cat not in all_label]
                        random.shuffle(rest_label)
                        all_label = set(list(all_label) + rest_label[:15 - len(all_label)])
                    assert len(all_label) == 15

                ###
                cat_to_label = {}
                for num, cat in enumerate(all_label):
                    cat_to_label[cat] = num
                this_epoch_category = []
                for cat in pos_label:
                    this_epoch_category.append(cat_to_label[cat])
                use_for_save_label = [class_name_list[cat -1] for cat in pos_label]
                batch_text_list = []
                for idx in all_label:
                    batch_text_list.append(class_name_list[idx -1])
                batch_text_list.append(" ")
                labels = torch.tensor(this_epoch_category).to(self.device).squeeze(0)  ## label [1,6]
                if len(labels.shape) == 0:
                    labels = labels.unsqueeze(0)
                if torch.min(labels).item() < 0:
                    raise ValueError("label out of index")
                targets = [{'boxes': boxes, 'labels': labels}] ## [55,4] [55]

                for epoch in range(epochs):
                    optimizer.zero_grad()
                    inputs = self.processor(text=[batch_text_list], images=images, return_tensors="pt").to(self.device) #### [pixel_value] [1, 3, 768, 768] [input_ids] [1204, 16]
                    inputs["prompt_use"] = True
                    outer_prompt = None
                    outputs = self.model(**inputs)  ### pred_sims [1, 576, 1203], all_pred_boxes ### [1, 576, 4]
                    loss_dict, early_exit, pred_box, pred_labels = criterion(outputs, targets)
                    if early_exit:
                        print("early exit", epoch)                    
                        if epoch == 0:
                            num_0_exit = num_0_exit + 1
                        break                
                    weight_dict = criterion.weight_dict
                    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                    loss_dict_reduced = utils.reduce_dict(loss_dict)
                    loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                                for k, v in loss_dict_reduced.items()}
                    loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                                for k, v in loss_dict_reduced.items() if k in weight_dict}
                    losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

                    loss_value = losses_reduced_scaled.item()
                    if not math.isfinite(loss_value):
                        print("Loss is {}, stopping training".format(loss_value))
                        print(loss_dict_reduced)
                        sys.exit(1)

                    losses.backward()
                    max_norm = clip_max_norm
                    if max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                    optimizer.step()

                if criterion.check_cls and criterion.check_box and epoch > 0:
                    save_image_list.append(images)
                    save_prompt.append(self.model.owlvit.vision_model.prompt_embeddings.data.clone())
                    

            print("number of all_samples: ", len(sampled_data))
            print("number of save: ", len(save_prompt))
            print("number of 0_exit: ", num_0_exit)
            self.model.eval()
            criterion.eval()
            for data in save_image_list:
                inputs = self.processor(text=batch_text_list, images=data, return_tensors="pt").to(self.device) #### [pixel_value] [1, 3, 768, 768] [input_ids] [1204, 16]
                inputs["prompt_use"] = False
                inputs["outer_prompt"] = "ori"
                with torch.no_grad():
                    outputs = self.model(**inputs)
                save_image.append(outputs.vision_model_output.last_hidden_state)
            if len(save_prompt) > 0:
                self.saved_prompt_dict[obj_name] = save_prompt
                self.saved_image_dict[obj_name] = save_image
        else:
            print("LVIS dataset does not contain " + str(obj_name))


    def make_lvis_transforms(self):

        return T.Compose_lvis([
                T.RandomResize([(768, 768)], max_size=1333),
        ])


    def count_pseudo_right(self, outputs):

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
    

    def top_box(self,img):
        w,h = img.size        
        return [0,0,w-1,int(h/2)]

    def bottom_box(self,img):
        w,h = img.size
        return [0,int(h/2),w-1,h-1]

    def left_box(self,img):
        w,h = img.size
        return [0,0,int(w/2),h-1]

    def right_box(self,img):
        w,h = img.size
        return [int(w/2),0,w-1,h-1]

    def box_image(self,img,boxes,highlight_best=True):
        img1 = img.copy()
        draw = ImageDraw.Draw(img1)
        for i,box in enumerate(boxes):
            if i==0 and highlight_best:
                color = 'red'
            else:
                color = 'blue'

            draw.rectangle(box,outline=color,width=5)

        return img1



    def execute(self,prog_step):
        img_var,obj_name,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        if obj_name=='TOP':
            bboxes = [self.top_box(img)]
        elif obj_name=='BOTTOM':
            bboxes = [self.bottom_box(img)]
        elif obj_name=='LEFT':
            bboxes = [self.left_box(img)]
        elif obj_name=='RIGHT':
            bboxes = [self.right_box(img)]
        else:
            bboxes = self.predict(img,obj_name)

        box_img = self.box_image(img, bboxes)
        prog_step.state[output_var] = bboxes
        prog_step.state[output_var+'_IMAGE'] = box_img

        return bboxes


class Loc2Interpreter(LocInterpreter):

    def execute(self,prog_step):
        img_var,obj_name,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        bboxes = self.predict(img,obj_name)

        objs = []
        for box in bboxes:
            objs.append(dict(
                box=box,
                category=obj_name
            ))
        prog_step.state[output_var] = objs


        return objs
    