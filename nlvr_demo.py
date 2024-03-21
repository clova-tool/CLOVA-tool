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
from framework.clova import CLOVA

LLM_config_path='configs/LLM_config.yaml'
LLM_config=  yaml.load(open(LLM_config_path, 'r'), Loader=yaml.Loader)
LLM_config['Task_type']='nlvr'
#####create the model
CLOVA_model=CLOVA(LLM_config)
#####create the model


#################dataset construction#################
with open(LLM_config['NLVR']['Dataset_path']+LLM_config['NLVR']['train_file'], 'rb') as file:
    dataset = pickle.load(file)
n_batches=len(dataset)


with open(LLM_config['NLVR']['Dataset_path']+LLM_config['NLVR']['test_file'], 'rb') as test_file:
    test_dataset = pickle.load(test_file)
test_n_batches=len(test_dataset)
#################dataset construction#################
train_data_num=LLM_config['NLVR']['train_data_num']
test_data_num=LLM_config['NLVR']['test_data_num']
interval=LLM_config['NLVR']['interval']



#################start train#################
i=0
correct_count=0
total_count=0
failed_prog=0
loop = tqdm(dataset)

for data in loop:
    print ('\n=====================train=====================train=====================train=====================train=====================train=============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================')

    total_count=total_count+1

    statement=data['sentence']
    left_image_path=LLM_config['NLVR']['Dataset_path']+'image/train/'+data["left_image"]
    right_image_path=LLM_config['NLVR']['Dataset_path']+'image/train/'+data["right_image"]
    answer=data["label"]
    answer=str(answer).lower().lstrip(' ').lstrip(' ')
    sample_id=str(data["identifier"])

    left_image = Image.open(left_image_path)
    left_image.thumbnail((640,640),Image.Resampling.LANCZOS)
    right_image = Image.open(right_image_path)
    right_image.thumbnail((640,640),Image.Resampling.LANCZOS)
    init_state = dict(
        LEFT=left_image.convert('RGB'),
        RIGHT=right_image.convert('RGB'),
    )


    print('=================The '+str(i)+'-th training question===============================')
    print ('------------------question------------------',statement)
    print ('------------------left_image_path------------------',left_image_path)
    print ('------------------right_image_path------------------',right_image_path)    
    print ('------------------graound truth answer------------------', answer)
    human_feedback= f'the correct answer should be {answer}' 


    #################inference phase#################
    can_run, subq, prog, index, result, prog_state, _, _ =CLOVA_model.inference(statement, init_state)
    result=str(result)

    try:
        result=str(result).lstrip('\n').rstrip('\n').lower()
    except:
        print('the answer seems wrong')

    if result.lstrip('\n').rstrip('\n').lower()==answer.lower():
        correct_count=correct_count+1
    if can_run==False:
        failed_prog=failed_prog+1
        print ('the program has bug')
    print ('------------------prediction result------------------', result.lstrip('\n').rstrip('\n').lower())
    print ('------------------is the question correctedly answered?------------------', (result==answer))  

    #################reflection process#################
    inference_results=dict(can_run=can_run, correct=(result==answer), index=index, init_state=init_state, prog_state=prog_state, question=statement, subq=subq, prog=prog, human_feedback=human_feedback, answer=answer)
    state, reflection_outputs = CLOVA_model.reflection(inference_results)
    print ('------------------reflection result result------------------')   
    print ('state',state)
    print ('reflection_outputs',reflection_outputs)  

    #################learning process#################
    if 'no_need_reflection' in state:
        learning_inputs=dict(
        question=statement,
        answer=answer,
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
            question=statement,
            answer=answer,
            subq=subq,
            prog=prog,
            location=reflection_outputs['location'],
            reason=reflection_outputs['reason'],
            init_state=init_state,
            prog_state=prog_state)

            # CLOVA_model.learning(learning_inputs)

        else:
            learning_inputs=dict(
            question=statement,
            answer=answer,
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
    i=i+1
    accuracy=float(correct_count/total_count)
    prog_success_ration=float(failed_prog/total_count)
    loop.set_postfix(train_accuracy=accuracy, prog_bug_ration=prog_success_ration)



    if (i+1)%interval==0:
        print ('\n=====================test=====================test=====================test=====================test=====================test=============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================')
        test_correct_count=0
        test_failed_prog=0
        test_total_count=0            
        j=0
        loop_test = tqdm(test_dataset)
        for test_data in loop_test:
            if j>test_data_num:
                break

            test_total_count=test_total_count+1
            statement=test_data['sentence']
            left_image_path=LLM_config['NLVR']['Dataset_path']+'image/dev/'+test_data["left_image"]
            right_image_path=LLM_config['NLVR']['Dataset_path']+'image/dev/'+test_data["right_image"]
            answer=test_data["label"]
            answer=str(answer).lower().lstrip(' ').lstrip(' ')
            sample_id=str(test_data["identifier"])

            left_image = Image.open(left_image_path)
            left_image.thumbnail((640,640),Image.Resampling.LANCZOS)
            right_image = Image.open(right_image_path)
            right_image.thumbnail((640,640),Image.Resampling.LANCZOS)
            init_state = dict(
                LEFT=left_image.convert('RGB'),
                RIGHT=right_image.convert('RGB'),
            )

            print ('------------------question------------------',statement)
            print ('------------------left_image_path------------------',left_image_path)
            print ('------------------right_image_path------------------',right_image_path)
            print ('------------------graound truth answer------------------', answer)  

            #################test inference phase#################
            can_run, subq, prog, index, result, prog_state, _, _ =CLOVA_model.inference(statement, init_state)
            result=str(result)
            if result.lstrip('\n').rstrip('\n').lower()==answer.lower():
                test_correct_count=test_correct_count+1
            if can_run==False:
                test_failed_prog=test_failed_prog+1
            print ('------------------prediction result------------------', result)   

            #################test report#################
            test_accuracy=float(test_correct_count/test_total_count)
            test_prog_success_ration=float(test_failed_prog/test_total_count)
            loop_test.set_postfix(test_accuracy=test_accuracy, test_prog_success_ration=test_prog_success_ration)