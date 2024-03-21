# %%
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# %%

# %%
from PIL import Image
from IPython.core.display import HTML
from functools import partial
import torch
import ruamel.yaml as yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from engine.utils import ProgramGenerator, ProgramInterpreter, PartReflectioner, Program_React_Generator
from engine.utils import reason_locate

from prompts.prompt_engineering import create_subquestion_prompt
from prompts.prompt_engineering import create_program_prompt

from prompts.prompt_engineering import create_part_reflection_prompt_step
from prompts.prompt_engineering import create_part_reflection_prompt_stepbystep
from prompts.prompt_engineering import create_part_reflection_prompt_interrupt

from prompts.prompt_engineering import create_subquestion_react_prompt
from prompts.prompt_engineering import create_program_react_prompt
from prompts.prompt_engineering import create_part_reflection_prompt_inference

from prompts.prompt_engineering import experience_summzerize, experience_filter
from prompts.prompt_engineering import experience_store_incorrect, experience_store_correct
from tools.model_updating import update_vqa_model, update_loc_model, update_seg_model, update_select_model, update_replace_model, update_classify_model



class CLOVA():
    def __init__(self,LLM_config):

        self.LLM_config=LLM_config
        self.incontext_num=self.LLM_config['GQA']['incontext_num']    

        #inference
        self.subquestion_prompter = partial(create_subquestion_prompt,method='retrieval',num_prompts=int(self.incontext_num))
        self.program_prompter = partial(create_program_prompt,method='retrieval',num_prompts=int(self.incontext_num))
        self.generator = ProgramGenerator(subquestion_prompter=self.subquestion_prompter, program_prompter=self.program_prompter)
        self.interpreter = ProgramInterpreter(task=self.LLM_config['Task_type'])

        #learning
        self.part_inference_prompter = partial(create_part_reflection_prompt_inference, method='random', num_prompts=int(self.incontext_num))

        #reflection
        self.part_reflection_prompter_step = partial(create_part_reflection_prompt_step,method='random',num_prompts=6)
        self.part_reflection_prompter_stepbystep = partial(create_part_reflection_prompt_stepbystep,method='random',num_prompts=int(self.incontext_num))
        self.part_reflection_prompter_interrupt = partial(create_part_reflection_prompt_interrupt,method='all',num_prompts=int(self.incontext_num))
        self.part_reflectioner= PartReflectioner(part_reflection_prompter_step=self.part_reflection_prompter_step, part_reflection_prompter_stepbystep=self.part_reflection_prompter_stepbystep, part_reflection_prompter_interrupt=self.part_reflection_prompter_interrupt)
        self.subquestion_react_prompter = partial(create_subquestion_react_prompt,method='random',num_prompts=int(self.incontext_num))
        self.program_react_prompter = partial(create_program_react_prompt,method='random',num_prompts=int(self.incontext_num))  
        self.generator_react= Program_React_Generator(subquestion_prompter=self.subquestion_prompter, program_prompter=self.program_prompter, subquestion_react_prompter=self.subquestion_react_prompter, program_react_prompter=self.program_react_prompter, part_inference_prompter=self.part_inference_prompter)


        


    #inference
    def inference(self, question, init_state):

        subq, prog, index = self.generator.generate(dict(question=question))
        can_run=True

        if 'FACEDET' in prog:
            is_face = True
        else:
            is_face = False

        try:
            result, prog_state, extra_out, real_loc = self.interpreter.execute(prog, init_state, is_face=is_face) 
        except:
            can_run=False
            result=''
            prog_state=None
            extra_out=None
            real_loc=None

        return can_run, subq, prog, index, result, prog_state, extra_out, real_loc
 

    #######reflection
    def LLMs_react(self,inputs):

        re_subq, re_prog = self.generator_react.generate(dict(question=inputs['question'], subquestion=inputs['subq'], program=inputs['prog'], errorlocation=inputs['errorlocation'], reason=inputs['reason'], pre_subq=inputs['subq'], pre_prog=inputs['prog']))
        return re_subq, re_prog

    def intermediate(self, prog_state):

        intermediate_results=self.generator_react.intermediate(prog_state)
        return intermediate_results
        
    def plan_program_correct_store(self, correct, index):
        experience_summzerize(correct, index)
        experience_filter()


    def program_interrupt_reflection(self, inputs):

        analysis=self.part_reflectioner.analyze_interrupt(dict(question=inputs['question'], human_feedback=inputs['human_feedback'], subquestion=inputs['subq'], program=inputs['prog']))
        print('-------------interrupt_analysis-------------',analysis)
        location, reason=reason_locate(analysis)
        location=location.lstrip('\n').rstrip('\n')
        reason=reason.lstrip('\n').rstrip('\n')

        return location, reason
    
    
    def plan_program_fast_reflection(self, inputs, prog_state):

        fast_analysis=self.part_reflectioner.analyze_step(dict(question=inputs['question'], human_feedback=inputs['human_feedback'], subquestion=inputs['subq'], program=inputs['prog']), prog_state)
        print('-------------fast_analysis-------------',fast_analysis)
        fast_location, fast_reason=reason_locate(fast_analysis)
        fast_location=fast_location.lstrip('\n').rstrip('\n')
        fast_reason=fast_reason.lstrip('\n').rstrip('\n')

        return fast_location, fast_reason
    

    def plan_program_slow_reflection(self, inputs, prog_state):

        slow_reason_find, slow_analysis, slow_error_step=self.part_reflectioner.analyze_stepbystep(dict(question=inputs['question'], human_feedback=inputs['human_feedback'], subquestion=inputs['subq'], program=inputs['prog']), prog_state)
        print('-------------slow_analysis-------------',slow_analysis)
        slow_location, slow_reason=reason_locate(slow_analysis)
        slow_location=slow_location.lstrip('\n').rstrip('\n')
        slow_reason=slow_reason.lstrip('\n').rstrip('\n')
        slow_reason=slow_error_step+slow_reason

        return slow_reason_find, slow_location, slow_reason


    def reflection(self, inputs):

        if inputs['can_run']==False or inputs['correct']==False:
            self.plan_program_correct_store('incorrect', inputs['index'])
        else:
            self.plan_program_correct_store('correct', inputs['index'])

        if inputs['can_run']==False:
            errorlocation='program'   
            interrupt_location, interrupt_reason = self.program_interrupt_reflection(dict(question=inputs['question'], human_feedback=inputs['human_feedback'], subq=inputs['subq'], prog=inputs['prog']))   
            interrupt_subq, interrupt_prog = self.LLMs_react(dict(question=inputs['question'], subq=inputs['subq'], prog=inputs['prog'], errorlocation=errorlocation, reason=interrupt_reason, pre_subq=inputs['subq'], pre_prog=inputs['prog']))   
            print ('--------------interrupt_reason--------------',interrupt_reason)
            try:
                if 'FACEDET' in interrupt_prog:
                    is_face = True
                else:
                    is_face = False                 
                interrupt_result, interrupt_prog_state, interrupt_extra_out, interrupt_real_loc = self.interpreter.execute(interrupt_prog, inputs['init_state'], is_face=is_face)
            except:
                interrupt_result='program has bug'
                reflection_outputs=dict(location='None', reason='None', new_subq='None', new_prog='None', new_prog_state='None')
                print('bad interrupt reflection, program still has bug')
                return 'failed_interrupt_reflection', reflection_outputs
            
            try:
                interrupt_result=str(interrupt_result).lstrip('\n').rstrip('\n').lower()
            except:
                print ('interrupt_result is not a string')

            print ('-------------interrupt_result-------------',interrupt_result)
            print ('-------------ground truth-------------',inputs['answer'])
            if interrupt_result== inputs['answer']:
                reflection_outputs=dict(location='program', reason=interrupt_reason, new_subq=interrupt_subq, new_prog=interrupt_prog, new_prog_state=interrupt_prog_state)
                print('good interrupt reflection, correct prediction')
                return 'successful_inerrupt_reflection', reflection_outputs
            else:
                reflection_outputs=dict(location='None', reason='None', new_subq='None', new_prog='None', new_prog_state='None')
                print('bad interrupt reflection, wrong prediction')         
                return 'failed_interrupt_reflection', reflection_outputs
            
        else:
            if inputs['correct']==True:
                reflection_outputs=dict(location='None', reason='None', new_subq='None', new_prog='None', new_prog_state='None')
                return 'no_need_reflection', reflection_outputs
            else:
                ####fast adaptation
                fast_location, fast_reason = self.plan_program_fast_reflection(dict(question=inputs['question'], subq=inputs['subq'], prog=inputs['prog'], human_feedback=inputs['human_feedback']), inputs['prog_state'])
                print ('--------------fast_location--------------',fast_location)
                print ('--------------fast_reason--------------',fast_reason)
                if 'function' in fast_location: 
                    reflection_outputs=dict(location=fast_location, reason=fast_reason, new_subq='None', new_prog='None', new_prog_state='None')
                    return 'successful_fast_reflection', reflection_outputs
                else:
                    fast_reflection_flag=0
                    fast_subq, fast_prog = self.LLMs_react(dict(question=inputs['question'], subq=inputs['subq'], prog=inputs['prog'], errorlocation=fast_location, reason=fast_reason, pre_subq=inputs['subq'], pre_prog=inputs['prog']))
                    print ('----------------re-generation plan/program based on fast reflection----------------')

                    fast_run=0
                    try:
                        if 'FACEDET' in fast_prog:
                            is_face = True
                        else:
                            is_face = False
                        fast_result, fast_prog_state, fast_extra_out, fast_real_loc = self.interpreter.execute(fast_prog, inputs['init_state'], is_face=is_face) 

                    except:
                        fast_reflection_flag=1
                        fast_run=1
                        fast_result='program bug'
                        print('fast reflection is bad, the regenerated program has bug')

                    try:
                        fast_result=str(fast_result).lstrip('\n').rstrip('\n').lower()
                    except:
                        print ('fast_result is not a string')                        
                    print ('-------------fast result-------------',fast_result)
                    print ('-------------ground truth-------------',inputs['answer'])
                    if fast_run==0:    
                        if fast_result== inputs['answer']:
                            reflection_outputs=dict(location=fast_location, reason=fast_reason, new_subq=fast_subq, new_prog=fast_prog, new_prog_state=fast_prog_state)
                            print('fast reflection is good, the answer is correct')
                            return 'successful_fast_reflection', reflection_outputs
                        else:
                            fast_reflection_flag=1
                            print('fast reflection is bad, the answer is incorrect')   
                                          
                ####slow adaptation
                if fast_reflection_flag!=0:
                    slow_reason_find, slow_location, slow_reason=self.plan_program_slow_reflection(dict(question=inputs['question'], subq=inputs['subq'], prog=inputs['prog'], human_feedback=inputs['human_feedback']), inputs['prog_state'])
                    print ('--------------slow_location--------------',slow_location)
                    print ('--------------slow_reason--------------',slow_reason)
                if slow_reason_find==False:
                    reflection_outputs=dict(location='None', reason='None', new_subq='None', new_prog='None', new_prog_state='None')
                    print('slow reflection cannot find the error')
                    return 'failed_slow_reflection', reflection_outputs
                else:
                    if 'function' in slow_location:
                        print ('----------------error is caused by tools----------------')
                        reflection_outputs=dict(location=slow_location, reason=slow_reason, new_subq='None', new_prog='None', new_prog_state='None')
                        return 'successful_slow_reflection', reflection_outputs
                    else:
                        print ('----------------error is caused by plans/programs----------------')
                        slow_reflection_flag=0
                        slow_subq, slow_prog = self.LLMs_react(dict(question=inputs['question'], subq=inputs['subq'], prog=inputs['prog'], errorlocation=slow_location, reason=slow_reason, pre_subq=inputs['subq'], pre_prog=inputs['prog']))
                        print ('----------------re-generation plan/program based on slow reflection----------------')
                        print(slow_subq)
                        print(slow_prog)

                        slow_run=0
                        try:
                            if 'FACEDET' in slow_prog:
                                is_face = True
                            else:
                                is_face = False
                            slow_result, slow_prog_state, slow_extra_out, slow_real_loc = self.interpreter.execute(slow_prog, inputs['init_state'], is_face=is_face)    
                        except:
                            slow_run=0
                            slow_result='program has bug'
                            reflection_outputs=dict(location='None', reason='None', new_subq='None', new_prog='None', new_prog_state='None')
                            print('slow reflection is bad, the regenerated program has bug')
                            return 'failed_slow_reflection', reflection_outputs
                        
                        try:
                            slow_result=str(slow_result).lstrip('\n').rstrip('\n').lower()
                        except:
                            print ('slow_result is not a string')  

                        print ('-------------slow_result-------------',slow_result)
                        print ('-------------ground truth-------------',inputs['answer'])
                        if slow_result== inputs['answer']:
                            print('slow reflection is good, the program can run, and the prediction is correct')
                            reflection_outputs=dict(location=slow_location, reason=slow_reason, new_subq=slow_subq, new_prog=slow_prog, new_prog_state=slow_prog_state)
                            return 'successful_slow_reflection', reflection_outputs
                        else:
                            reflection_outputs=dict(location='None', reason='None', new_subq='None', new_prog='None', new_prog_state='None')
                            print('slow reflection is bad, the program can run, but the prediction is wrong')    
                            return 'failed_slow_reflection', reflection_outputs        




    def LLMs_updating(self,inputs):

        experience_store_correct(dict(question=inputs['question'], subquestion=inputs['subq'], program=inputs['prog']))
        if 'incorrect_subq' in inputs.keys():
            experience_store_incorrect(inputs['location'],dict(question=inputs['question'], subquestion=inputs['incorrect_subq'], program=inputs['incorrect_prog'], reason=inputs['reason']))


    def tools_updating(self, inputs):
        
        init_state=inputs['init_state']
        prog_state=inputs['prog_state']
        subq=inputs['subq']
        prog=inputs['prog']
        location=inputs['location']
        reason=inputs['reason']
        answer=inputs['answer']

        if 'VQA' in prog:
            update_vqa_model(self.interpreter, self.generator_react, subq, prog, init_state, prog_state, location, reason, answer)

        if 'function' in location:
            if 'LOC' in reason:
                update_loc_model(self.interpreter, self.generator_react, subq, prog, init_state, prog_state, location, reason, answer)
            elif 'SEG' in reason: 
                update_seg_model(self.interpreter, self.generator_react, subq, prog, init_state, prog_state, location, reason, answer)
            elif 'SELECT' in reason: 
                update_select_model(self.interpreter, self.generator_react, subq, prog, init_state, prog_state, location, reason, answer)        
            elif 'REPLACE' in reason: 
                update_replace_model(self.interpreter, self.generator_react, subq, prog, init_state, prog_state, location, reason, answer)     
            elif 'CLASSIFY' in reason: 
                update_classify_model(self.interpreter, self.generator_react, subq, prog, init_state, prog_state, location, reason, answer)   


    def learning (self, inputs):

        self.tools_updating(inputs)
        if 'function' not in inputs['location']:
            self.LLMs_updating(inputs)
