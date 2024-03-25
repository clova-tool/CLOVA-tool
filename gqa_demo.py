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
from framework.clova import CLOVA
from Datasets.loaders import GQADataset

LLM_config_path='configs/LLM_config.yaml'
LLM_config=  yaml.load(open(LLM_config_path, 'r'), Loader=yaml.Loader)
LLM_config['Task_type']='gqa'
#####create the model
CLOVA_model=CLOVA(LLM_config)
#####create the model

#################dataset construction#################
train_dataset = GQADataset(split="train", balanced=True, data_path=LLM_config['GQA']['Dataset_path'], testing=False)
train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=False)
train_n_batches = len(train_dataset)

dataset = GQADataset(split="testdev", balanced=False, data_path=LLM_config['GQA']['Dataset_path'], testing=False)
dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
n_batches = len(dataset)

loop = tqdm(enumerate(train_dataloader), total =train_n_batches)
#################dataset construction#################


train_data_num=LLM_config['GQA']['train_data_num']
test_data_num=LLM_config['GQA']['test_data_num']
interval=LLM_config['GQA']['interval']


#################start train#################
correct_count=0
total_count=0
failed_prog=0
for i, data in loop:
    print ('\n=====================train=====================train=====================train=====================train=====================train=============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================')

    if i > train_data_num:
        break

    total_count=total_count+1
    sample_id=data['sample_id'][0]
    image_id=data['image_id'][0]
    image_path=data['img'][0]
    image=Image.open(image_path)
    question=data['question'][0]
    answer=data['answer'][0]

    image.thumbnail((640,640),Image.Resampling.LANCZOS)
    init_state = dict(IMAGE=image.convert('RGB'))

    print('=================The '+str(i)+'-th training question===============================')
    print ('------------------question------------------',question)
    print ('------------------image_id------------------',str(image_id))
    print ('------------------graound truth answer------------------', answer) 
    human_feedback= f'the correct answer should be {answer}' 


    #################inference phase#################
    can_run, subq, prog, index, result, prog_state, _, _ =CLOVA_model.inference(question, init_state)
    try:
        result=str(result).lstrip('\n').rstrip('\n').lower()
    except:
        print('the answer seems wrong')
    if result==answer.lower():
        correct_count=correct_count+1
    if can_run==False:
        failed_prog=failed_prog+1
        print ('the program has bug')        
    print ('------------------prediction result------------------', result)   
    print ('------------------is the question correctedly answered?------------------', (result==answer))  

    #################reflection process#################
    inference_results=dict(can_run=can_run, correct=(result==answer), index=index, init_state=init_state, prog_state=prog_state, question=question, subq=subq, prog=prog, human_feedback=human_feedback, answer=answer)
    state, reflection_outputs = CLOVA_model.reflection(inference_results)
    print ('------------------reflection result result------------------')   
    print ('state',state)
    print ('reflection_outputs',reflection_outputs)  

    #################learning process#################
    if 'no_need_reflection' in state:
        learning_inputs=dict(
        question=question,
        answer=answer,
        subq=subq,
        prog=prog,
        location='None',
        reason='None',
        init_state=init_state,
        prog_state=prog_state)

        CLOVA_model.learning(learning_inputs)

    elif 'failed' not in state:
        if 'function' in state:
            learning_inputs=dict(
            question=question,
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
            question=question,
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
    accuracy=float(correct_count/total_count)
    prog_success_ration=float(failed_prog/total_count)
    loop.set_postfix(train_accuracy=accuracy, prog_bug_ration=prog_success_ration)



    if (i+1)%interval==0:
        print ('\n=====================test=====================test=====================test=====================test=====================test=============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================')
        test_correct_count=0
        test_total_count=0
        test_failed_prog=0
        loop_test = tqdm(enumerate(dataloader), total =n_batches)
        for j, test_data in loop_test:
            if j>test_data_num:
                break
            test_total_count=test_total_count+1
            sample_id=test_data['sample_id'][0]
            image_id=test_data['image_id'][0]
            image_path=test_data['img'][0]
            image=Image.open(image_path)
            question=test_data['question'][0]
            answer=test_data['answer'][0]

            image.thumbnail((640,640),Image.Resampling.LANCZOS)
            init_state = dict(
                IMAGE=image.convert('RGB')
            )

            print('=================The '+str(i)+'-th test question===============================')
            print ('------------------question------------------',question)
            print ('------------------image_id------------------',str(image_id))
            print ('------------------prediction result------------------', result)
            print ('------------------graound truth answer------------------', answer) 


            #################test inference phase#################
            can_run, subq, prog, index, result, prog_state, _, _ =CLOVA_model.inference(question, init_state)

            try:
                result=str(result).lstrip('\n').rstrip('\n').lower()
            except:
                print('the answer seems wrong')

            if result==answer.lower():
                test_correct_count=test_correct_count+1
            if can_run==False:
                test_failed_prog=test_failed_prog+1
            print ('------------------prediction result------------------', result)   

            #################test report#################
            test_accuracy=float(test_correct_count/test_total_count)
            test_prog_success_ration=float(test_failed_prog/test_total_count)
            loop_test.set_postfix(test_accuracy=test_accuracy, test_prog_success_ration=test_prog_success_ration)