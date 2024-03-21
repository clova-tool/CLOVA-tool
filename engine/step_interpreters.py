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


from .nms import nms

from tools.blip_vqa.blip_vqa import blip_vqa


from engine.data_utils import pre_question
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode



from tools.CLASSIFY import ClassifyInterpreter
from tools.FaceDet import FaceDetInterpreter
from tools.LOC import Loc2Interpreter, LocInterpreter
from tools.REPLACE import ReplaceInterpreter
from tools.SELECT import SelectInterpreter
from tools.SEG import SegmentInterpreter
from tools.VQA import VQAInterpreter
from tools.unupdated_functions import TagInterpreter,  ListInterpreter, EmojiInterpreter,  BgBlurInterpreter, ColorpopInterpreter, \
    CropAheadInterpreter, CropBehindInterpreter, CropInFrontInterpreter, CropInFrontOfInterpreter, CropFrontOfInterpreter, CropBelowInterpreter, \
    CropAboveInterpreter, CropLeftOfInterpreter, CropRightOfInterpreter, CropInterpreter, CountInterpreter, ResultInterpreter, EvalInterpreter





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



def dummy(images, **kwargs):
    return images, False




def register_step_interpreters(task,llama_generator):
    if task=='nlvr':
        return dict(
            VQA=VQAInterpreter(),
            EVAL=EvalInterpreter(),
            RESULT=ResultInterpreter()
        )
    elif task=='gqa':
        return dict(
            LOC=LocInterpreter(),
            COUNT=CountInterpreter(),
            CROP=CropInterpreter(),
            CROP_RIGHTOF=CropRightOfInterpreter(),
            CROP_LEFTOF=CropLeftOfInterpreter(),
            CROP_FRONTOF=CropFrontOfInterpreter(),
            CROP_INFRONTOF=CropInFrontOfInterpreter(),
            CROP_INFRONT=CropInFrontInterpreter(),
            CROP_BEHIND=CropBehindInterpreter(),
            CROP_AHEAD=CropAheadInterpreter(),
            CROP_BELOW=CropBelowInterpreter(),
            CROP_ABOVE=CropAboveInterpreter(),
            VQA=VQAInterpreter(),
            EVAL=EvalInterpreter(),
            RESULT=ResultInterpreter()
        )
    elif task=='imgedit':
        return dict(
            FACEDET=FaceDetInterpreter(),
            SEG=SegmentInterpreter(),
            SELECT=SelectInterpreter(llama_generator),
            COLORPOP=ColorpopInterpreter(),
            BGBLUR=BgBlurInterpreter(),
            REPLACE=ReplaceInterpreter(),
            EMOJI=EmojiInterpreter(),
            RESULT=ResultInterpreter()
        )
    elif task=='knowtag':
        return dict(
            FACEDET=FaceDetInterpreter(),
            LIST=ListInterpreter(llama_generator),
            CLASSIFY=ClassifyInterpreter(llama_generator),
            RESULT=ResultInterpreter(),
            TAG=TagInterpreter(),
            LOC=Loc2Interpreter()
        )