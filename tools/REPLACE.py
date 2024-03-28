import io, tokenize
import torch
import numpy as np
from augly.utils.base_paths import EMOJI_DIR
import augly.image as imaugs

from PIL import Image,ImageDraw,ImageFont,ImageFilter
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline
from engine.Crawl import crawl_google, crawl_baidu

import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler

from tqdm import tqdm
import yaml
import os
from torch.utils.data import Dataset
from engine.template import imagenet_templates_small, imagenet_style_templates_small
from torchvision import transforms
import random
import PIL
from packaging import version

all_updated_model_config_path='configs/all_updated_model_config.yaml'
all_model_config=  yaml.load(open(all_updated_model_config_path, 'r'), Loader=yaml.Loader)
from tools.SEG import SegmentInterpreter
from tools.FaceDet import FaceDetInterpreter

def dummy(images, **kwargs):
    return images, False

    
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





class ReplaceInterpreter():
    step_name = 'REPLACE'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.config = all_model_config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # sd_model = "runwayml/stable-diffusion-v1-5"
        # model_name = "runwayml/stable-diffusion-inpainting"
        # print ('sd_pipe',self.config["REPLACE"]['init']['sd_pipe']['pretrained'])
        # print ('pipe',self.config["REPLACE"]['init']['pipe']['pretrained'])
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(self.config["REPLACE"]['init']['sd_pipe']['pretrained'])
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.config["REPLACE"]['init']['pipe']['pretrained'],
            # revision="fp16",
            # torch_dtype=torch.float16,
            # inpainting_fill = 3, # 蒙版遮住的内容， 0填充， 1原图 2潜空间噪声 3潜空间数值零
            # inpaint_full_res = True, # inpaint area, False: whole picture True：only masked
            # inpaint_full_res_padding = 16, #Only masked padding, pixels 32
            # inpainting_mask_invert =  0,  # 蒙版模式 0重绘蒙版内容 1 重绘非蒙版内容
            # styles = [ "whole", "complete" ],
            # negative_prompt = ['parts of neighboring objects', "filling the surrounding objects", "incomplete"],
            # mask_blur = 8,
            )
    


        self.pipe = self.pipe.to(self.device)
        self.sd_pipe = self.sd_pipe.to(self.device)
        # self.pipe.safety_checker = dummy
        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False
        self.sd_pipe.safety_checker = None
        self.sd_pipe.requires_safety_checker = False
        
        self.download_image_num = self.config["REPLACE"]['init']['download_image_num']
        self.save_crawl_pics_path = self.config["REPLACE"]['init']['save_crawl_pics_path']
        self.token_pool = []
        self.emb_pool = []
        
    def crawl_pics(self, token, crawl_path):
        url = self.config["REPLACE"]['crawl_pics']['url_start'] + token + self.config["REPLACE"]['crawl_pics']['url_end']
        # train_images_list = crawl_google(url, crawl_path, token, self.download_image_num, is_face=False, FaceDetInterpreter=FaceDetInterpreter(), SegmentInterpreter=SegmentInterpreter(), only_image=True)
        train_images_list = crawl_baidu(url, crawl_path, token, self.download_image_num, is_face=False, FaceDetInterpreter=FaceDetInterpreter(), SegmentInterpreter=SegmentInterpreter(), only_image=True)
        

    
    
    def update(self, prompt, learnable_property="object"):
        new_token = prompt.replace(" ", "_")
        crawl_path = f"{self.save_crawl_pics_path}"
        self.crawl_pics(prompt, crawl_path)
        # update_prompt(prompt, new_token, crawl_path, learnable_property=learnable_property)


        # cfg = {

        #     "warmup_epochs": 500,
        #     "lr_scheduler": "constant", 
        #     "learning_rate": 1e-3,
        #     # "gradient_accumulation_steps": 1,
        #     "mixed_precision": "no",  
        #     "num_train_epochs": 4000,
        #     "train_batch_size": 4,
        #     "seed": 23,

        #     "learnable_property": learnable_property, 
        #     "initializer_token": new_token,
        #     "placeholder_token":  new_token,
        #     "train_data_dir": f"{crawl_path}/{prompt}",
        #     # "output_dir": "./embeddings/test_breadfruit_tree_sd_3000_5e-3_1.bin"
        # }


        cfg = {
            "warmup_epochs": self.config["REPLACE"]['update']['warmup_epochs'],
            "lr_scheduler": "constant", 
            "learning_rate": float(self.config["REPLACE"]['update']['learning_rate']),
            "gradient_accumulation_steps": self.config["REPLACE"]['update']['gradient_accumulation_steps'],
            "mixed_precision": "no",  
            "num_train_epochs": self.config["REPLACE"]['update']['num_train_epochs'],
            "train_batch_size": self.config["REPLACE"]['update']['train_batch_size'],
            "seed":  self.config["REPLACE"]['update']['seed'],
            
            "learnable_property": learnable_property, 
            "initializer_token": new_token,
            "placeholder_token":  new_token,
            "train_data_dir": f"{crawl_path}/{prompt}",
            # "output_dir": "./embeddings/test_breadfruit_tree_sd_3000_5e-3_1.bin"
        }
        accelerator_project_config = ProjectConfiguration(total_limit=None)
        accelerator = Accelerator(
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        # mixed_precision=mixed_precision,
        # log_with="wandb",
        project_config=accelerator_project_config,
        )
        if cfg["seed"] is not None:
            set_seed(cfg["seed"])
            
        noise_scheduler = self.sd_pipe.scheduler
        unet = self.sd_pipe.unet
        text_encoder = self.sd_pipe.text_encoder
        tokenizer = self.sd_pipe.tokenizer
        vae = self.sd_pipe.vae
        
        num_added_tokens = tokenizer.add_tokens(cfg["placeholder_token"])
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {cfg['placeholder_token']}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )
            
        # token_ids = tokenizer.encode(cfg["initializer_token"], add_special_tokens=False)
        token_ids = [tokenizer.convert_tokens_to_ids(cfg["initializer_token"])]
        
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        initializer_token_id = token_ids[0]
        placeholder_token_id = tokenizer.convert_tokens_to_ids(cfg["placeholder_token"])
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id] 
        
        # Freeze vae and unet
        vae.requires_grad_(False)
        unet.requires_grad_(False)
        # Freeze all parameters except for the token embeddings in text encoder
        text_encoder.text_model.encoder.requires_grad_(False)
        text_encoder.text_model.final_layer_norm.requires_grad_(False)
        text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
        
    
        optimizer = torch.optim.AdamW(
            text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
            lr= cfg["learning_rate"], 
            betas=tuple(self.config["REPLACE"]['update']['optimizer']['betas']),
            weight_decay= float(self.config["REPLACE"]['update']['optimizer']['weight_decay']),
            eps= float(self.config["REPLACE"]['update']['optimizer']['eps']),
        )
        
        # Dataset and DataLoaders creation:
        train_dataset = TextualInversionDataset(
            data_root=cfg["train_data_dir"],
            tokenizer=tokenizer,
            size=self.config["REPLACE"]['update']['train_dataset']['size'],
            placeholder_token=cfg["placeholder_token"],
            # repeats=100,
            learnable_property=cfg["learnable_property"], # object style
            center_crop=True,
            set="train",
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg["train_batch_size"], shuffle=True, num_workers=0
        )
        

        # lr_scheduler = get_scheduler(
        #     cfg["lr_scheduler"],
        #     optimizer=optimizer,
        # )

        steps_per_epoch=len(train_dataloader)
        print("steps_per_epoch", steps_per_epoch)
        lr_scheduler = get_scheduler(
            cfg["lr_scheduler"],
            optimizer=optimizer,
            num_warmup_steps=cfg["warmup_epochs"]*steps_per_epoch,
            num_training_steps=cfg["num_train_epochs"] * steps_per_epoch,
        )


        # Prepare everything with our `accelerator`.
        text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            text_encoder, optimizer, train_dataloader, lr_scheduler
        )
        
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        # Move vae and unet to device and cast to weight_dtype
        unet.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)
    
        total_batch_size = cfg["train_batch_size"] * accelerator.num_processes * cfg["gradient_accumulation_steps"]

        orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
        
        for epoch in tqdm(range(0, cfg["num_train_epochs"])):
            # print("epoch", epoch)
            text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                # print("step",step)
                with accelerator.accumulate(text_encoder):
                    
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                    latents = latents * vae.config.scaling_factor
                     
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
            
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)

                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    print ('REPLACE,loss',loss)
                    
                    accelerator.backward(loss)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    index_no_updates = torch.arange(len(tokenizer)) != placeholder_token_id
                    with torch.no_grad():
                        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                            index_no_updates
                        ] = orig_embeds_params[index_no_updates]
 
                # logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        
        learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
        self.token_pool.append(new_token)
        self.emb_pool.append(learned_embeds.detach().cpu())
        
    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        prompt = eval(parse_result['args']['prompt'])
        output_var = parse_result['output_var']
        # instruction = parse_result['args']['instruction']
        assert(step_name==self.step_name)
        # return img_var,obj_var,prompt,output_var,instruction
        return img_var,obj_var,prompt,output_var
    

    def create_mask_img(self,objs):
        mask = objs[0]['mask']
        mask[mask>0.5] = 255
        mask[mask<=0.5] = 0
        mask = mask.astype(np.uint8)
        return Image.fromarray(mask)

    def merge_images(self,old_img,new_img,mask):
        print(mask.size,old_img.size,new_img.size)

        mask = np.array(mask).astype(np.float)/255
        mask = np.tile(mask[:,:,np.newaxis],(1,1,3))
        img = mask*np.array(new_img) + (1-mask)*np.array(old_img)
        return Image.fromarray(img.astype(np.uint8))

    def resize_and_pad(self,img,size=(512,512)):
        new_img = Image.new(img.mode,size)
        thumbnail = img.copy()
        thumbnail.thumbnail(size)
        new_img.paste(thumbnail,(0,0))
        W,H = thumbnail.size
        return new_img, W, H
    
        # def predict(self, token, learnable_property="object"):
    def predict(self, img, mask, prompt, learnable_property="object"):

        print ('REPLACE token_pool',len(self.token_pool))
        print ('REPLACE emb_pool',len(self.emb_pool))

        modified_token = prompt.replace(" ", "_")
        """"
        # if modified_token not in self.token_pool:
        #     crawl_path = f"{self.save_crawl_pics_path}"
        #     self.crawl_pics(prompt, crawl_path)
            # self.update_prompt(prompt, modified_token, crawl_path, learnable_property=learnable_property)
        """
        if modified_token in self.token_pool:
            print ('self.token_pool', self.token_pool)
            index = self.token_pool.index(modified_token) 
            learned_embeds = {modified_token: self.emb_pool[index]}
            # learned_token = torch.load(learned_embeds)
            self.pipe.load_textual_inversion(learned_embeds, token=modified_token)
        
        mask,_,_ = self.resize_and_pad(mask)
        init_img,W,H = self.resize_and_pad(img)
        new_img = self.pipe(
            prompt=prompt,
            image=init_img,
            mask_image=mask,
            # strength=0.98,
            # guidance_scale=self.config["REPLACE"]['predict']['new_img']['guidance_scale'],
            num_inference_steps=self.config["REPLACE"]['predict']['new_img']['num_inference_steps'], #200
        ).images[0]
        return new_img.crop((0,0,W-1,H-1)).resize(img.size)
    

    def create_maskimg(self,objs,img):
        box = objs[0]['box']
        new_img = img.crop(box)
        # mask[mask>0.5] = 255
        # mask[mask<=0.5] = 0
        # mask = mask.astype(np.uint8)
        # return Image.fromarray(mask)
        return new_img
    
    # def get_caption(self, img, instruction):
    #     # image = Image.open('images/inputs/birdtest.png')
    #     text=I2T_function(img)

    #     print(text)
    #     prompt = f"Here is a description of an image: '{text}.' \nThe command to edit the image: '{instruction}.' \nWhat is the precise description of the modified image? Just give the description without any extra words."
    #     # print(prompt)
    #     answer = get_eval(content=prompt, max_tokens = 20)["content"]
    #     return answer
    
    def execute(self,prog_step):
        # img_var,obj_var,prompt,output_var,instruction = self.parse(prog_step)
        img_var,obj_var,prompt,output_var = self.parse(prog_step)
        
        img = prog_step.state[img_var] # whole image
        objs = prog_step.state[obj_var]
        mask = self.create_mask_img(objs) # 得到mask
        new_img = self.predict(img, mask, prompt)
        # new_img = self.predict(new_img, mask, prompt)

        prog_step.state[output_var] = new_img

        return new_img




if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
    

class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        # repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images 

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example