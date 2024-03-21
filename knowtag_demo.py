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
import json
from framework.clova import CLOVA
from knowtag_fl_score.ok_tag_res import judge_all_right, calculate_f1_score_single, calculate_f1_score_tag

LLM_config_path='configs/LLM_config.yaml'
LLM_config=  yaml.load(open(LLM_config_path, 'r'), Loader=yaml.Loader)


#####create the model
CLOVA_model=CLOVA(LLM_config)
#####create the model

#################dataset construction#################

dataset_train_path=LLM_config['KNOWTAG']['dataset_train_path']
dataset_test_path=LLM_config['KNOWTAG']['dataset_test_path']
gt_path=LLM_config['KNOWTAG']['gt_path']
result_save_path=LLM_config['KNOWTAG']['result_save_path']


with open(gt_path, 'r') as file:
    data_real = file.readlines()
print ('ground truth of training data has been loaded.')

if os.path.exists(result_save_path):
    print('file exists')
else:
    print('results file not exists')
    os.mkdir(result_save_path)

before_json_train_file_name= result_save_path+'testresult_before_.json'
before_ans_file = open(before_json_train_file_name, 'a')

after_json_train_file_name= result_save_path+'testresult_after_.json'
after_ans_file = open(after_json_train_file_name, 'a')
#################dataset construction#################


train_data_num=LLM_config['GQA']['train_data_num']
test_data_num=LLM_config['GQA']['test_data_num']
interval=LLM_config['GQA']['interval']


correct_count=0
total_count=0
failed_prog=0

#################start test before train#################
with open(dataset_test_path+'test.txt', 'r') as f:   
    lines = f.readlines()
    n_data= len(lines)      
    # pbar = tqdm(lines)
    real_count=0
    for line in lines:

        print ('real_count',real_count)
        real=data_real[real_count]
        real_json = json.loads(real.strip())
        real_box_all = real_json['real']
        real_json = json.loads(real.strip())
        image_name=real_json['image']
        real_box_all = real_json['real']

        print ('\n================beforelearning=========================================================================================================================================================================================================================================================================================================================================================================================================================beforelearning=============================')
        image_path,instruction=line.split(';')
        instruction=instruction[:-1]
        print ('image_path:',image_path)
        print ('Instruction:',instruction)
        image = Image.open(dataset_test_path+image_path)

        image.thumbnail((640,640),Image.ANTIALIAS)
        init_state = dict(
            IMAGE=image.convert('RGB')
        )

        #################inference phase#################
        can_run, subq, prog, index, result, prog_state, before_extra_out, before_real_loc =CLOVA_model.inference(instruction, init_state)
        if can_run==False or before_extra_out==None:
            failed_prog=failed_prog+1
            result=''
            before_extra_out=[]
            cur_js= {}
            cur_js['image'] = image_path
            cur_js['res'] = before_extra_out
            before_ans_file.write(json.dumps(cur_js) + '\n')
            before_ans_file.flush()
            real_count=real_count+1
        else:
            cur_js= {}
            cur_js['image'] = image_path
            if len(before_extra_out) > 0:
                for data in before_extra_out:
                    if 'category' in data.keys():
                        del data['category']
                    if 'inst_id' in data.keys():
                        del data['inst_id']
                    if 'mask' in data.keys():
                        del data['mask']
                    if 'class_score' in data.keys():
                        del data['class_score']
            cur_js['res'] = before_extra_out
            before_ans_file.write(json.dumps(cur_js) + '\n')
            before_ans_file.flush()
            real_count=real_count+1

            if 'Image' in str(type(result)):
                result.save(result_save_path+image_path+instruction+'_before_result.png')





#################start train#################
with open(dataset_train_path+'train.txt', 'r') as f:   
    lines = f.readlines()
    n_data= len(lines)      
    # pbar = tqdm(lines)
    real_count=0
    for line in lines:

        print ('real_count',real_count)
        real=data_real[real_count]
        real_json = json.loads(real.strip())
        real_box_all = real_json['real']
        real_json = json.loads(real.strip())
        image_name=real_json['image']
        real_box_all = real_json['real']

        print ('\n================training=========================================================================================================================================================================================================================================================================================================================================================================================================================beforelearning=============================')
        image_path,instruction=line.split(';')
        instruction=instruction[:-1]
        print ('image_path:',image_path)
        print ('Instruction:',instruction)
        image = Image.open(dataset_train_path+image_path)

        image.thumbnail((640,640),Image.ANTIALIAS)
        init_state = dict(
            IMAGE=image.convert('RGB')
        )

        #################inference phase#################
        can_run, subq, prog, index, result, prog_state, extra_out, real_loc =CLOVA_model.inference(instruction, init_state)
        if can_run==False or extra_out==None:
            failed_prog=failed_prog+1
            is_all_right=False
            human_feedback='This program has bug'
            result=''
            extra_out=[]
            cur_js= {}
            cur_js['image'] = image_path
            cur_js['res'] = extra_out

        else:
            cur_js= {}
            cur_js['image'] = image_path
            if len(extra_out) > 0:
                for data in extra_out:
                    if 'category' in data.keys():
                        del data['category']
                    if 'inst_id' in data.keys():
                        del data['inst_id']
                    if 'mask' in data.keys():
                        del data['mask']
                    if 'class_score' in data.keys():
                        del data['class_score']
            cur_js['res'] = extra_out

            new_format_pred =[]
            new_format_real = []
            for data in extra_out:
                new_format_pred.append((data['box'], data['class']))
            for data in real_box_all:
                new_format_real.append((data['box'], data['class']))

            is_all_right = judge_all_right(new_format_pred, new_format_real) ### judge each image correct or incorrect
            print("signle image predictiobn: ", is_all_right)   
            f1_single = calculate_f1_score_single(new_format_pred, new_format_real) ### 计算单张图f1 score   
            print("single image f1_socre", f1_single)                             
            true_positives, false_positives, false_negatives = calculate_f1_score_tag(new_format_pred, new_format_real)

            if true_positives + false_positives==0:
                precision=0
            else:
                precision = true_positives / (true_positives + false_positives)
                
            if true_positives + false_negatives==0:
                recall=0
            else:
                recall = true_positives / (true_positives + false_negatives)

            F1_score=f1_single
            dict_gt=new_format_real.copy()
            dict_our=new_format_pred.copy()
            num_gt=len(dict_gt)
            num_our=len(dict_our)
            Our_prediction=''
            Ground_truth=f'There are {str(num_gt)} objects should be tagged, while our method tags {str(num_our)} objects. The details of desirable prediction is {str(dict_gt)}, while our prediction is {str(dict_our)}.'
            human_feedback=Our_prediction+Ground_truth

            real_count=real_count+1


        #################reflection process#################
        inference_results=dict(can_run=can_run, correct=is_all_right, index=index, init_state=init_state, prog_state=prog_state, question=instruction, subq=subq, prog=prog, human_feedback=human_feedback, answer='None')
        state, reflection_outputs = CLOVA_model.reflection(inference_results)
        print ('------------------reflection result result------------------')   
        print ('state',state)
        print ('reflection_outputs',reflection_outputs)  



        #################learning process#################
        if 'failed' in state:
            continue

        elif 'no_need_reflection' in state:
            learning_inputs=dict(
            question=instruction,
            answer='None',
            subq=subq,
            prog=prog,
            location='None',
            reason='None',
            init_state=init_state,
            prog_state=prog_state)

            CLOVA_model.learning(learning_inputs)

        else:
            if 'function' in reflection_outputs['reason']:
                learning_inputs=dict(
                question=instruction,
                answer='None',
                subq=subq,
                prog=prog,
                location=reflection_outputs['location'],
                reason=reflection_outputs['reason'],
                init_state=init_state,
                prog_state=prog_state)

            else:
                learning_inputs=dict(
                question=instruction,
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





#################start test after train#################
with open(dataset_test_path+'test.txt', 'r') as f:   
    lines = f.readlines()
    n_data= len(lines)      
    # pbar = tqdm(lines)
    real_count=0
    for line in lines:

        print ('real_count',real_count)
        real=data_real[real_count]
        real_json = json.loads(real.strip())
        real_box_all = real_json['real']
        real_json = json.loads(real.strip())
        image_name=real_json['image']
        real_box_all = real_json['real']

        print ('\n================afterelearning=========================================================================================================================================================================================================================================================================================================================================================================================================================beforelearning=============================')
        image_path,instruction=line.split(';')
        instruction=instruction[:-1]
        print ('image_path:',image_path)
        print ('Instruction:',instruction)
        image = Image.open(dataset_test_path+image_path)

        image.thumbnail((640,640),Image.ANTIALIAS)
        init_state = dict(
            IMAGE=image.convert('RGB')
        )

        #################inference phase#################
        can_run, subq, prog, index, result, prog_state, after_extra_out, after_real_loc =CLOVA_model.inference(instruction, init_state)
        if can_run==False or after_extra_out==None:
            failed_prog=failed_prog+1
            result=''
            after_extra_out=[]
            cur_js= {}
            cur_js['image'] = image_path
            cur_js['res'] = after_extra_out
            after_ans_file.write(json.dumps(cur_js) + '\n')
            after_ans_file.flush()
            real_count=real_count+1

        else:
            cur_js= {}
            cur_js['image'] = image_path
            if len(after_extra_out) > 0:
                for data in after_extra_out:
                    if 'category' in data.keys():
                        del data['category']
                    if 'inst_id' in data.keys():
                        del data['inst_id']
                    if 'mask' in data.keys():
                        del data['mask']
                    if 'class_score' in data.keys():
                        del data['class_score']
            cur_js['res'] = after_extra_out
            after_ans_file.write(json.dumps(cur_js) + '\n')
            after_ans_file.flush()

            if 'Image' in str(type(result)):
                result.save(result_save_path+image_path+instruction+'_after_result.png')









