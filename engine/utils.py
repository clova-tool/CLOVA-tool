import os
from PIL import Image
import openai
import numpy as np
import copy
import ruamel.yaml as yaml
import time

from llama import Llama
from .step_interpreters import register_step_interpreters, parse_step

from tools.image2text import I2T_function
from tools.image2text import I2T_model
from prompts.intermediate_result import INTERMEDIATE_FUNC

image2text=I2T_model()

LLM_config_path='configs/LLM_config.yaml'
LLM_config=  yaml.load(open(LLM_config_path, 'r'), Loader=yaml.Loader)

ckpt_dir= LLM_config['LLaMA']['ckpt_dir_path']
tokenizer_path= LLM_config['LLaMA']['tokenizer_path']
temperature = LLM_config['LLaMA']['temperature']
top_p = LLM_config['LLaMA']['top_p']
max_seq_len = LLM_config['LLaMA']['max_seq_len']
max_gen_len = LLM_config['LLaMA']['max_gen_len']
max_batch_size = LLM_config['LLaMA']['max_batch_size']



llama_generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)


def concent_location_answer(input):

    index_s=input.find('\n\n')
    if index_s>0:
        return input[:index_s+1]
    else:
        return input


def concent_location(input):

    index_s=input.find('\n\n')
    if index_s>2:
        return input[:index_s+1]
    else:
        return input


class Program:
    def __init__(self,prog_str,init_state=None):
        self.prog_str = prog_str
        self.state = init_state if init_state is not None else dict()
        self.instructions = self.prog_str.split('\n')


class ProgramInterpreter:
    def __init__(self,task='nlvr'):
        self.step_interpreters = register_step_interpreters(task,llama_generator)
        self.task=task

    def execute_step(self,prog_step, is_face):
        step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
        # 执行
        if step_name == "SELECT":
            return self.step_interpreters[step_name].execute(prog_step,is_face)  # 执行模块
        elif step_name =='SEG':
            return self.step_interpreters[step_name].execute(prog_step,query_name=self.seg_query)  # 执行模块        
        else:
            return self.step_interpreters[step_name].execute(prog_step)  # 执行模块

    def execute(self,prog, init_state, is_face=False):

        if 'SEG' in prog:
            select_list=self.search_select(prog,init_state)
            self.seg_query=select_list[0]['query']

        if isinstance(prog,str):
            prog = Program(prog,init_state)
        else:
            assert(isinstance(prog,Program))

        prog_steps = [Program(instruction,init_state=prog.state) \
            for instruction in prog.instructions]

        extra_out = None
        real_loc = None

        for prog_step in prog_steps:
            step_output = self.execute_step(prog_step,is_face)

            if self.task == 'knowtag':
                if prog_step.prog_str.find("CLASSIFY") > -1:
                    extra_out = step_output.copy()  
                if prog_step.prog_str.find("LOC") > -1:
                    real_loc = step_output            

        return step_output, prog.state, extra_out, real_loc




    def parse_vqa(self,prog_step,i):
        parse_result = parse_step(prog_step.prog_str)
        if parse_result['step_name']=='VQA':
            args = parse_result['args']
            img_var = args['image']
            question = eval(args['question'])
            output_var = parse_result['output_var']            

            return dict(img_var=img_var,question=question,output_var=output_var, step=i)
        else:
            return None

    def search_vqa(self,prog,init_state):
        vqa_list=[]

        if isinstance(prog,str):
            prog = Program(prog,init_state)
        else:
            assert(isinstance(prog,Program))

        prog_steps = [Program(instruction,init_state=prog.state) \
            for instruction in prog.instructions]

        step=1
        for prog_step in prog_steps:
            step_var = self.parse_vqa(prog_step, step)
            if step_var !=None:
                vqa_list.append(step_var)
            step=step+1

        return vqa_list


    def parse_loc(self,prog_step,i):
        # step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
        parse_result = parse_step(prog_step.prog_str)
        if parse_result['step_name']=='LOC':
            args = parse_result['args']
            img_var = args['image']
            object1 = args['object']
            output_var = parse_result['output_var']            

            return dict(img_var=img_var,object=object1,output_var=output_var, step=i)
        else:
            return None

    def search_loc(self,prog,init_state):
        vqa_list=[]

        if isinstance(prog,str):
            prog = Program(prog,init_state)
        else:
            assert(isinstance(prog,Program))

        prog_steps = [Program(instruction,init_state=prog.state) \
            for instruction in prog.instructions]

        step=1
        for prog_step in prog_steps:
            step_var = self.parse_loc(prog_step, step)
            if step_var !=None:
                vqa_list.append(step_var)
            step=step+1

        return vqa_list

    def parse_select(self,prog_step,i,mod="SELECT"):
        # step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
        parse_result = parse_step(prog_step.prog_str)
        if mod == "SELECT":
            if parse_result['step_name']=='SELECT':
                args = parse_result['args']
                img_var = args['image']
                object1 = args['object']
                query = args['query']
                category = args['category']
                output_var = parse_result['output_var']            

                return dict(img_var=img_var, object=object1, query=query, category=category, output_var=output_var, step=i)
            else:
                return None
        elif mod == "REPLACE":
            if parse_result['step_name']=='REPLACE':
                args = parse_result['args']
                img_var = args['image']
                object1 = args['object']
                query = args['query']
                category = args['category']
                output_var = parse_result['output_var']            

                return dict(img_var=img_var, object=object1, query=query, category=category, output_var=output_var, step=i)
            else:
                return None


    def search_select(self,prog,init_state, mod="SELECT"):
        select_list=[]

        if isinstance(prog,str):
            prog = Program(prog,init_state)
        else:
            assert(isinstance(prog,Program))

        prog_steps = [Program(instruction,init_state=prog.state) \
            for instruction in prog.instructions]

        step=1
        for prog_step in prog_steps:
          
            step_var = self.parse_select(prog_step, step, mod=mod)
            
            if step_var !=None:
                select_list.append(step_var)
            step=step+1

        return select_list


    def parse_seg(self,prog_step,i):
        # step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
        parse_result = parse_step(prog_step.prog_str)
        if parse_result['step_name']=='SEG':
            args = parse_result['args']
            img_var = args['image']
            output_var = parse_result['output_var']            

            return dict(img_var=img_var, output_var=output_var, step=i)
        else:
            return None

    def search_seg(self,prog,init_state):
        seg_list=[]

        if isinstance(prog,str):
            prog = Program(prog,init_state)
        else:
            assert(isinstance(prog,Program))

        prog_steps = [Program(instruction,init_state=prog.state) \
            for instruction in prog.instructions]

        step=1
        for prog_step in prog_steps:
            step_var = self.parse_select(prog_step, step)
            if step_var !=None:
                seg_list.append(step_var)
            step=step+1

        return seg_list

    def parse_replace(self,prog_step,i):
        # step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
        parse_result = parse_step(prog_step.prog_str)
    
        if parse_result['step_name']=='REPLACE':
            args = parse_result['args']
            img_var = args['image']
            object1 = args['object']
            query = args['prompt']
            # category = args['category']
            output_var = parse_result['output_var']            

            return dict(img_var=img_var, object=object1, query=query, output_var=output_var, step=i)
        else:
            return None


    def search_replace(self,prog,init_state):
        select_list=[]

        if isinstance(prog,str):
            prog = Program(prog,init_state)
        else:
            assert(isinstance(prog,Program))

        prog_steps = [Program(instruction,init_state=prog.state) \
            for instruction in prog.instructions]

        step=1
        for prog_step in prog_steps:
          
            step_var = self.parse_replace(prog_step, step)
            
            if step_var !=None:
                select_list.append(step_var)
            step=step+1

        return select_list


    def parse_list(self,prog_step,i):
        # step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
        parse_result = parse_step(prog_step.prog_str)
        if parse_result['step_name']=='LIST':
            args = parse_result['args']
            query = args['query']
            list_max = args['max']
            output_var = parse_result['output_var']            

            return dict(query=query,list_max=list_max,output_var=output_var,step=i)
        else:
            return None

    def search_list(self,prog,init_state):
        list_list=[]

        if isinstance(prog,str):
            prog = Program(prog,init_state)
        else:
            assert(isinstance(prog,Program))

        prog_steps = [Program(instruction,init_state=prog.state) \
            for instruction in prog.instructions]

        step=1
        for prog_step in prog_steps:
            step_var = self.parse_list(prog_step, step)
            if step_var !=None:
                list_list.append(step_var)
            step=step+1

        return list_list


    def parse_classify(self,prog_step,i):
        # step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
        parse_result = parse_step(prog_step.prog_str)
        if parse_result['step_name']=='CLASSIFY':
            args = parse_result['args']
            image = args['image']
            object1 = args['object']
            categories = args['categories']
            output_var = parse_result['output_var']            

            return dict(image=image,object=object1,categories=categories, output_var=output_var,step=i)
        else:
            return None

    def search_classify(self,prog,init_state):
        list_classify=[]

        if isinstance(prog,str):
            prog = Program(prog,init_state)
        else:
            assert(isinstance(prog,Program))

        prog_steps = [Program(instruction,init_state=prog.state) \
            for instruction in prog.instructions]

        step=1
        for prog_step in prog_steps:
            step_var = self.parse_list(prog_step, step)
            if step_var !=None:
                list_classify.append(step_var)
            step=step+1

        return list_classify




    def update_vqavisualmodel(self, model_name, correct, data):
        print ('------------------start update the visual model:', model_name)
        self.step_interpreters[model_name].update(correct,data)

    def update_locvisualmodel(self, model_name, data):
        print ('------------------start update the LOC visual model:', model_name)
        self.step_interpreters[model_name].update(data)

    def update_selectvisualmodel(self, model_name, data):
        print ('------------------start update the SELECT visual model:', model_name)
        self.step_interpreters[model_name].update(data)

    def update_segvisualmodel(self, model_name, data):
        print ('------------------start update the SEG visual model:', model_name)
        self.step_interpreters[model_name].update(data)

    def update_replacevisualmodel(self, model_name, data):
        print ('------------------start update the REPLACE visual model:', model_name)
        self.step_interpreters[model_name].update(data)       

    def update_classifyvisualmodel(self, model_name, result, category_name):
        print ('------------------start update the CLASSIFY visual model:', model_name)
        self.step_interpreters[model_name].update(result, category_name)      


class ProgramGenerator():
    def __init__(self, subquestion_prompter, program_prompter, temperature=0.7, top_p=0.5, prob_agg='mean'):

        self.subquestion_prompter = subquestion_prompter
        self.program_prompter = program_prompter

        self.temperature = temperature
        self.top_p = top_p
        self.prob_agg = prob_agg

    def compute_prob(self,response):
        eos = '<|endoftext|>'
        for i,token in enumerate(response.choices[0]['logprobs']['tokens']):
            if token==eos:
                break

        if self.prob_agg=='mean':
            agg_fn = np.mean
        elif self.prob_agg=='sum':
            agg_fn = np.sum
        else:
            raise NotImplementedError

        return np.exp(agg_fn(
            response.choices[0]['logprobs']['token_logprobs'][:i]))


    def generate(self,inputs):

        subquestion_prompt, subq_correct_index, sub_failed_index=self.subquestion_prompter(inputs, dict(question=inputs['question']), index=True)
        print('------------------subquestion_prompt start------------------')
        print(subquestion_prompt)
        print('------------------subquestion_prompt end------------------')     



        try:
            subquestion_prompt_list=[]
            subquestion_prompt_list.append(subquestion_prompt)
            response = llama_generator.text_completion(
                subquestion_prompt_list,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            subquestion = response[0]['generation']
            subquestion=concent_location(subquestion)
            subquestion = subquestion.lstrip('\n').rstrip('\n')  
        except:
            subquestion='some errors'



        program_prompt, prog_correct_index, prog_failed_index=self.program_prompter(dict(question=inputs['question'], subquestion=subquestion),index=True)
        print('------------------program_prompt start------------------')
        print(program_prompt)
        print('------------------program_prompt end------------------')  


        try:
            program_prompt_list=[]
            program_prompt_list.append(program_prompt)
            program_reresponse = llama_generator.text_completion(
                program_prompt_list,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            prog = program_reresponse[0]['generation']
            prog=concent_location(prog)
            prog = prog.lstrip('\n').rstrip('\n')
        except:
            prog = 'some errors'

            
        print ('-------------subquestion-------------prog-------------')
        print (subquestion)
        print (prog)


        index=[subq_correct_index, sub_failed_index, prog_correct_index, prog_failed_index]

        return subquestion, prog, index
    



class PartReflectioner():
    def __init__(self, part_reflection_prompter_step, part_reflection_prompter_stepbystep, part_reflection_prompter_interrupt, temperature=0.7, top_p=0.5, prob_agg='mean'):
        self.part_reflection_prompter_step = part_reflection_prompter_step
        self.part_reflection_prompter_stepbystep = part_reflection_prompter_stepbystep
        self.part_reflection_prompter_interrupt = part_reflection_prompter_interrupt
        self.temperature = temperature
        self.top_p = top_p
        self.prob_agg = prob_agg

        self.intermediate=INTERMEDIATE_FUNC[LLM_config['Task_type']]

    def compute_prob(self,response):
        eos = '<|endoftext|>'
        for i,token in enumerate(response.choices[0]['logprobs']['tokens']):
            if token==eos:
                break

        if self.prob_agg=='mean':
            agg_fn = np.mean
        elif self.prob_agg=='sum':
            agg_fn = np.sum
        else:
            raise NotImplementedError

        return np.exp(agg_fn(
            response.choices[0]['logprobs']['token_logprobs'][:i]))



    def analyze_interrupt(self,inputs):

        reflection_prompt=self.part_reflection_prompter_interrupt(dict(question=inputs['question'], human_feedback=inputs['human_feedback'], subquestion=inputs['subquestion'], program=inputs['program']))
        
        print('------------------interrupt reflection_prompt start------------------')
        print (reflection_prompt.lstrip('\n').rstrip('\n'))
        print('------------------interrupt reflection_prompt end------------------')              


        try:
            reflection_prompt_list=[]
            reflection_prompt_list.append(reflection_prompt)
            reflection = llama_generator.text_completion(
                reflection_prompt_list,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            reflection = reflection[0]['generation']
            reflection=concent_location(reflection)
            reflection = reflection.lstrip('\n').rstrip('\n')     
        except:
            reflection='some errors'


        return reflection



    def analyze_step(self,inputs,prog_state):

        intermediate_output=self.intermediate(prog_state)
        reflection_prompt=self.part_reflection_prompter_step(dict(question=inputs['question'], human_feedback=inputs['human_feedback'], subquestion=inputs['subquestion'], program=inputs['program'],intermediate_output=intermediate_output),prog_state)
        
        print('------------------step reflection_prompt start------------------')
        print (reflection_prompt.lstrip('\n').rstrip('\n'))
        print('------------------step reflection_prompt end------------------')              


        try:
            reflection_prompt_list=[]
            reflection_prompt_list.append(reflection_prompt)
            reflection = llama_generator.text_completion(
                reflection_prompt_list,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            reflection = reflection[0]['generation']
            reflection=concent_location(reflection)
            reflection = reflection.lstrip('\n').rstrip('\n')   
        except:
            reflection='some errors'

        return reflection




    def analyze_stepbystep(self,inputs,prog_state):

        reason_find=False
        reason=None
        error_step=None

        intermediate_output=self.intermediate(prog_state)
        d_subquestion=inputs['subquestion']
        d_subquestion=d_subquestion.split('\n')
        total_number_step=len(d_subquestion)

        for i in range(total_number_step):
            reflection_prompt=self.part_reflection_prompter_stepbystep(dict(question=inputs['question'], human_feedback=inputs['human_feedback'], subquestion=inputs['subquestion'], program=inputs['program'],intermediate_output=intermediate_output),prog_state,i+1)
            
            print('------------------stepbystep reflection_prompt start------------------')
            print (reflection_prompt.lstrip('\n').rstrip('\n'))
            print('------------------stepbystep reflection_prompt end------------------')              


            try:
                reflection_prompt_list=[]
                reflection_prompt_list.append(reflection_prompt)
                reflection = llama_generator.text_completion(
                    reflection_prompt_list,
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )

                reflection = reflection[0]['generation']
                reflection=concent_location(reflection)
                reflection = reflection.lstrip('\n').rstrip('\n')   
            except:
                reflection='some errors'


            if 'yes' in reflection or 'Yes' in reflection:
                continue
            else:
                reason_find=True
                no_location=reflection.find('No')
                reason=reflection[no_location+3:]
                error_step='Error is in Step'+str(i+1)+'. '
                break

        return reason_find, reason, error_step






def reason_locate(reflection):

    reason_index=reflection.find('Reason:')

    if reason_index <2: 
        reason_end_index=reflection.find('\n\n')
        return reflection[:reason_end_index], reflection[:reason_end_index]
    else:

        location = reflection[:reason_index]
        reason_end_index=reflection.find('\n\n')
        reason=reflection[reason_index+7:reason_end_index]

        return location, reason




class Program_React_Generator():
    def __init__(self, subquestion_prompter, program_prompter, subquestion_react_prompter, program_react_prompter, part_inference_prompter, temperature=0.7, top_p=0.5, prob_agg='mean'):

        self.subquestion_prompter = subquestion_prompter
        self.program_prompter = program_prompter
        self.subquestion_react_prompter = subquestion_react_prompter
        self.program_react_prompter = program_react_prompter        

        self.answer_inference_prompter = part_inference_prompter

        self.intermediate=INTERMEDIATE_FUNC[LLM_config['Task_type']]

        self.temperature = temperature
        self.top_p = top_p
        self.prob_agg = prob_agg

    def compute_prob(self,response):
        eos = '<|endoftext|>'
        for i,token in enumerate(response.choices[0]['logprobs']['tokens']):
            if token==eos:
                break

        if self.prob_agg=='mean':
            agg_fn = np.mean
        elif self.prob_agg=='sum':
            agg_fn = np.sum
        else:
            raise NotImplementedError

        return np.exp(agg_fn(
            response.choices[0]['logprobs']['token_logprobs'][:i]))



    def generate(self,inputs):

        errorlocation=inputs['errorlocation']

        if ('subquestion' in errorlocation) or ('SubQuestion' in errorlocation) or ('Subquestion' in errorlocation):
            subquestion_prompt=self.subquestion_react_prompter(inputs)
        else:
            subquestion_prompt=self.subquestion_prompter(inputs, dict(question=inputs['question'], pre_subq=inputs['pre_subq']))
        print('------------------react_subquestion_prompt start------------------')
        print(subquestion_prompt)
        print('------------------react_subquestion_prompt end------------------')    

        try:
            subquestion_prompt_list=[]
            subquestion_prompt_list.append(subquestion_prompt)
            response = llama_generator.text_completion(
                subquestion_prompt_list,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            subquestion = response[0]['generation']
            subquestion=concent_location(subquestion)
            subquestion = subquestion.lstrip('\n').rstrip('\n')  
        except:
            subquestion= 'some errors'


        if 'program' in errorlocation:
            program_prompt=self.program_react_prompter(dict(question=inputs['question'], subquestion=inputs['subquestion'], program=inputs['program'], errorlocation=inputs['errorlocation'], reason=inputs['reason'], newsubquestion=subquestion))
        else:
            program_prompt=self.program_prompter(dict(question=inputs['question'], subquestion=subquestion),dict(question=inputs['question'], pre_subq=inputs['pre_subq'], pre_prog=inputs['pre_prog']))
        print('------------------react_program_prompt start------------------')
        print(program_prompt)
        print('------------------react_program_prompt end------------------')  


        try:
            program_prompt_list=[]
            program_prompt_list.append(program_prompt)
            program_reresponse = llama_generator.text_completion(
                program_prompt_list,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            prog = program_reresponse[0]['generation']
            prog=concent_location(prog)
            prog = prog.lstrip('\n').rstrip('\n')  
        except:
            prog='some errors'

        return subquestion, prog
    

    def answer_inference(self,inputs,prog_state):

        intermediate_output=self.intermediate(prog_state)
        inference_prompt=self.answer_inference_prompter(dict(question=inputs['question'], human_feedback=inputs['human_feedback'], subquestion=inputs['subquestion'], program=inputs['program'],intermediate_output=intermediate_output,errorlocation=inputs['errorlocation'], reason=inputs['reason']),prog_state)
        
        print('------------------answer_inference_prompt start------------------')
        print (inference_prompt.lstrip('\n').rstrip('\n'))
        print('------------------answer_inference_prompt end------------------')              

        try:
            inference_prompt_list=[]
            inference_prompt_list.append(inference_prompt)
            inference = llama_generator.text_completion(
                inference_prompt_list,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            inference = inference[0]['generation']
            inference=concent_location_answer(inference)
            inference = inference.lstrip('\n').rstrip('\n')   
        except:
            inference='some errors'

        return inference