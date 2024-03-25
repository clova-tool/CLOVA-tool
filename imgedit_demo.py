import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from PIL import Image
from IPython.core.display import HTML
import torch
import ruamel.yaml as yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import numpy as np
from framework.clova import CLOVA

LLM_config_path='configs/LLM_config.yaml'
LLM_config=  yaml.load(open(LLM_config_path, 'r'), Loader=yaml.Loader)

#####create the model
CLOVA_model=CLOVA(LLM_config)
#####create the model


#################dataset construction#################
dataset_train_path=LLM_config['IMGEDIT']['dataset_train_path']
dataset_test_path=LLM_config['IMGEDIT']['dataset_test_path']
image_folder=LLM_config['IMGEDIT']['image_path']
result_save_path=LLM_config['IMGEDIT']['result_save_path']

if os.path.exists(result_save_path):
    print('file exists')
else:
    print('results file not exists')
    os.mkdir(result_save_path)

#################dataset construction#################





#################start test before train#################
with open(dataset_test_path, 'r') as f:   
    lines = f.readlines()
    n_data= len(lines)      
    pbar = tqdm(lines)
    i=0
    correct_count=0
    total_count=0
    failed_prog=0

    for line in pbar:
        print ('\n=====================test=====================test=====================test=====================test=====================test=============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================')

        total_count=total_count+1
        image_path,question=line.split(';')
        image = Image.open(image_folder+image_path)
        image.thumbnail((640,640),Image.Resampling.LANCZOS)

        print('=================The '+str(i)+'-th test question===============================')
        print ('------------------question------------------',question)
        print ('------------------image_path------------------', image_path) 

        init_state = dict(
            IMAGE=image.convert('RGB')
        )

        #################inference phase#################
        can_run, subq, prog, index, result, prog_state,_,_ =CLOVA_model.inference(question, init_state)
        if can_run==False:
            failed_prog=failed_prog+1
            print('program bug')

        else:
            print ('the program correctly can')
            print ('--------prog_state---------',prog_state)
            try:
                result.save(result_save_path+image_path+question[:-1]+'_replace_test_before.png')
            except:
                print ('final result is not an image.')
            print ('--------prog_state---------')
        i=i+1



#################start train#################
with open(dataset_train_path, 'r') as f:   
    lines = f.readlines()
    n_data= len(lines)      
    pbar = tqdm(lines)
    i=0

    for line in pbar:
        print ('\n=====================train=====================train=====================train=====================train=====================train=============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================')

        total_count=total_count+1
        image_path,question,correct,feedback=line.split(';')
        human_feedback=feedback[:-1]
        image = Image.open(image_folder+image_path)
        image.thumbnail((640,640),Image.Resampling.LANCZOS)

        print('=================The '+str(i)+'-th training question===============================')
        print ('------------------question------------------',question)
        print ('------------------image_path------------------', image_path) 
        print ('------------------Does this question be correctly solved?------------------', int(correct)==1) 
        print ('------------------human_feedback------------------', human_feedback) 

        init_state = dict(
            IMAGE=image.convert('RGB')
        )


        #################inference phase#################
        can_run, subq, prog, index, result, prog_state, _, _ =CLOVA_model.inference(question, init_state)
        if int(correct)==1:
            correct_count=correct_count+1
            print ('result is correct')


        if can_run==False:
            failed_prog=failed_prog+1
            print('program bug')
        else:
            print ('the program correctly can')
            print ('------------------is the question correctedly answered?------------------', int(correct)==1)  
            try:
                result.save(result_save_path+image_path+question[:-1]+'_replace_train.png')
            except:
                print ('the results are not images')


        #################reflection process#################
        inference_results=dict(can_run=can_run, correct=correct, index=index, init_state=init_state, prog_state=prog_state, question=question, subq=subq, prog=prog, human_feedback=human_feedback, answer='None')
        state, reflection_outputs = CLOVA_model.reflection(inference_results)
        print ('------------------reflection result result------------------')   
        print ('state',state)
        print ('reflection_outputs',reflection_outputs)  


        #################learning process#################
        if 'no_need_reflection' in state:
            learning_inputs=dict(
            question=question,
            answer='None',
            subq=subq,
            prog=prog,
            location='None',
            reason='None',
            init_state=init_state,
            prog_state=prog_state)

            CLOVA_model.learning(learning_inputs)

        elif 'failed' not in state:
            if 'function' in reflection_outputs['reason']:
                learning_inputs=dict(
                question=question,
                answer='None',
                subq=subq,
                prog=prog,
                location=reflection_outputs['location'],
                reason=reflection_outputs['reason'],
                init_state=init_state,
                prog_state=prog_state)


            else:
                learning_inputs=dict(
                question=question,
                answer='None',
                subq=reflection_outputs['new_subq'],
                prog=reflection_outputs['new_prog'],
                location=reflection_outputs['location'],
                reason=reflection_outputs['reason'],
                incorrect_subq=subq,
                incorrect_prog=prog,
                init_state=init_state,
                prog_state=reflection_outputs['new_prog_state'])

            CLOVA_model.learning(learning_inputs)


        #################report#################
        accuracy=float(correct_count/total_count)
        prog_success_ration=float(failed_prog/total_count)
        i=i+1
        pbar.set_postfix(train_accuracy=accuracy, prog_bug_ration=prog_success_ration)







#################start test after train#################
with open(dataset_test_path, 'r') as f:   
    lines = f.readlines()
    n_data= len(lines)      
    pbar = tqdm(lines)
    i=0

    for line in pbar:
        print ('\n=====================test=====================test=====================test=====================test=====================test=============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================')

        total_count=total_count+1
        image_path,question=line.split(';')
        image = Image.open(image_folder+image_path)
        image.thumbnail((640,640),Image.Resampling.LANCZOS)

        print('=================The '+str(i)+'-th test question===============================')
        print ('------------------question------------------',question)
        print ('------------------image_path------------------', image_path) 

        init_state = dict(
            IMAGE=image.convert('RGB')
        )

        #################inference phase#################
        can_run, subq, prog, index, result, prog_state, _, _ =CLOVA_model.inference(question, init_state)
        if can_run==False:
            failed_prog=failed_prog+1
            print ('the program has bug')

        else:
            print ('the program correctly can')
            try:
                result.save(result_save_path+image_path+question[:-1]+'_replace_test_after.png')
            except:
                print('results are not images')

        i=i+1