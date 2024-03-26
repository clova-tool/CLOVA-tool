from tools.image2text import I2T_function
from tools.image2text import I2T_model
image2text=I2T_model()




def intermediate_gqa(prog_state):
    intermediate_output =''
    new_state=prog_state.copy()
    for k,v in new_state.items():
        if 'IMAGE' in k and 'BOX' in k:
            continue
        elif 'IMAGE' in k and 'BOX' not in k and ('Image' in str(type(v))):
            new_state[k]=image2text.forward(v)
            intermediate_output = intermediate_output + 'The description of ' + k + ': ' + str(new_state[k]) + '\n'
        elif 'BOX' in k and 'IMAGE' not in k:
            if 'Image' in str(type(v)):
                new_state[k]=image2text.forward(v)
                intermediate_output = intermediate_output + 'The description of ' + k + ': ' + str(new_state[k]) + '\n'                    
            elif 'list' in str(type(v)): 
                if len (v)>0:
                    intermediate_output = intermediate_output + 'The coordinate of ' +  k + ': ' + str(new_state[k]) + '\n'
                else:
                    intermediate_output = intermediate_output+ k + ' is empty ' + '\n'
            else:
                intermediate_output = intermediate_output + k + ': ' + str(new_state[k]) + '\n'
        else:
            intermediate_output = intermediate_output + k + ': ' + str(new_state[k]) + '\n'
    return intermediate_output


def intermediate_knowtag(prog_state):
    intermediate_output =''
    new_state=prog_state.copy()
    for k,v in new_state.items():
        
        if 'IMAGE' in k and 'OBJ' in k:
            continue
        elif 'IMAGE' in k and 'OBJ' not in k and ('Image' in str(type(v))):
            new_state[k]=image2text.forward(v)
            intermediate_output = intermediate_output + 'The description of ' + k + ': ' + str(new_state[k]).replace('\n', ' ') + '\n'
        elif 'FINAL' in k and 'OBJ' not in k and ('Image' in str(type(v))):
            new_state[k]=image2text.forward(v)
            intermediate_output = intermediate_output + 'The description of ' + k + ': ' + str(new_state[k]).replace('\n', ' ') + '\n'                
        elif 'OBJ' in k and 'IMAGE' not in k:
            if 'Image' in str(type(v)):
                new_state[k]=image2text.forward(v)
                intermediate_output = intermediate_output + 'The description of ' + k + ': ' + str(new_state[k]).replace('\n', ' ') + '\n'                    
            elif 'list' in str(type(v)): 
                if len (v)>0:
                    temp_v=[]
                    for i in range (len(v)):
                        if 'box' in v[i].keys():
                            temp_v.append(v[i]['box'])
                        else:
                            temp_v.append(v[i])
                    intermediate_output = intermediate_output + 'The coordinate of ' +  k + ': ' + str(temp_v).replace('\n', ' ') + '\n'
                else:
                    intermediate_output = intermediate_output+ k + ' is empty ' + '\n'
            else:
                intermediate_output = intermediate_output + k + ': ' + str(new_state[k]).replace('\n', ' ') + '\n'
        else:
            try:
                if 'list' in str(type(v)): 
                    if len (v)>0:
                        temp_v=[]
                        for i in range (len(v)):
                            if 'box' in v[i].keys():
                                temp_v.append(v[i]['box'])
                            else:
                                temp_v.append(v[i])
                        intermediate_output = intermediate_output + 'The coordinate of ' +  k + ': ' + str(temp_v).replace('\n', ' ') + '\n'
                    else:
                        intermediate_output = intermediate_output+ k + ' is empty ' + '\n'
                else:
                    intermediate_output = intermediate_output + k + ': ' + str(new_state[k]).replace('\n', ' ') + '\n'
            except:
                intermediate_output = intermediate_output + k + ': ' + str(new_state[k]).replace('\n', ' ') + '\n'
    return intermediate_output




def intermediate_nlvr(prog_state):
    intermediate_output =''
    new_state=prog_state.copy()
    for k,v in new_state.items():
        if ('LEFT' in k and 'BOX' in k) or ('RIGHT' in k and 'BOX' in k):
            continue
        elif ('LEFT' in k or 'RIGHT' in k) and 'BOX' not in k and ('Image' in str(type(v))):
            new_state[k]=image2text.forward(v)
            intermediate_output = intermediate_output + 'The description of ' + k + ': ' + str(new_state[k]) + '\n'
        elif 'BOX' in k and ('LEFT' not in k or 'RIGHT' not in k):
            if 'Image' in str(type(v)):
                new_state[k]=image2text.forward(v)
                intermediate_output = intermediate_output + 'The description of ' + k + ': ' + str(new_state[k]) + '\n'                    
            elif 'list' in str(type(v)): 
                if len (v)>0:
                    intermediate_output = intermediate_output + 'The coordinate of ' +  k + ': ' + str(new_state[k]) + '\n'
                else:
                    intermediate_output = intermediate_output+ k + ' is empty ' + '\n'
            else:
                intermediate_output = intermediate_output + k + ': ' + str(new_state[k]) + '\n'
        else:
            intermediate_output = intermediate_output + k + ': ' + str(new_state[k]) + '\n'
    return intermediate_output




def intermediate_imgedit(prog_state):
    intermediate_output =''
    new_state=prog_state.copy()
    for k,v in new_state.items():
        if 'IMAGE' in k and 'OBJ' in k:
            continue
        elif 'IMAGE' in k and 'OBJ' not in k and ('Image' in str(type(v))):
            new_state[k]=image2text.forward(v)
            intermediate_output = intermediate_output + 'The description of ' + k + ': ' + str(new_state[k]).replace('\n', ' ') + '\n'
        elif 'FINAL' in k and 'OBJ' not in k and ('Image' in str(type(v))):
            new_state[k]=image2text.forward(v)
            intermediate_output = intermediate_output + 'The description of ' + k + ': ' + str(new_state[k]).replace('\n', ' ') + '\n'
        elif 'OBJ' in k and 'IMAGE' not in k:
            if 'Image' in str(type(v)):
                new_state[k]=image2text.forward(v)
                intermediate_output = intermediate_output + 'The description of ' + k + ': ' + str(new_state[k]).replace('\n', ' ') + '\n'                    
            elif 'list' in str(type(v)): 
                if len (v)>0:
                    temp_v=[]
                    for i in range (len(v)):
                        if 'box' in v[i].keys():
                            temp_v.append(v[i]['box'])
                        else:
                            temp_v.append(v[i])
                    intermediate_output = intermediate_output + 'The coordinate of ' +  k + ': ' + str(temp_v).replace('\n', ' ') + '\n'
                else:
                    intermediate_output = intermediate_output+ k + ' is empty ' + '\n'
            else:
                intermediate_output = intermediate_output + k + ': ' + str(new_state[k]).replace('\n', ' ') + '\n'
        else:
            intermediate_output = intermediate_output + k + ': ' + str(new_state[k]).replace('\n', ' ') + '\n'

    return intermediate_output




def divide_by_step_gqa(question, human_feedback, subquestion, program, results, prog_step):
    # d_subquestion=subquestion.split('\n')
    d_program=program.split('\n')
    d_results=results.split('\n')

    prompt = 'Question: ' + question+ '\n'

    input_image_description=d_results[0]
    input_image_description_index=input_image_description.find(': ')
    input_image_description=input_image_description[input_image_description_index+2:]
    prompt = prompt+'Description of the Input Image: ' + input_image_description + '\n'

    prompt = prompt + 'Human Feedback: ' + human_feedback + '\n'

    our_result=d_results[len(d_program)]
    index=our_result.find('FINAL_RESULT: ')
    prompt = prompt + 'Our Wrong Answer: ' + our_result[index+14:] + '\n'

    prompt = prompt + 'Following are the decomposed subquestion, used program, and obtained result in each step. \n'

    prompt = prompt + 'subquestion: \n' + subquestion+ '\n' + 'Program and obtained result in each step: \n'

    for i in range(len(d_program)):
        prompt = prompt + 'Step'+str(i+1)+'\n'
        prog=d_program[i]
        prompt = prompt + 'Program: ' + prog +'\n'

        if ('The description of' in d_results[i+1]) or ('The coordinate of' in d_results[i+1]):
            prompt = prompt + d_results[i+1] +'\n'
        else:
            prompt = prompt + 'Result of ' + d_results[i+1] +'\n'

    return prompt



def divide_by_stepbystep_gqa(question, human_feedback, subquestion, program, results, prog_step, step_num):
    d_subquestion=subquestion.split('\n')
    d_program=program.split('\n')
    d_results=results.split('\n')

    prompt_question = 'Question: ' + question+ '\n'

    input_image_description=d_results[0]
    input_image_description_index=input_image_description.find(': ')
    input_image_description=input_image_description[input_image_description_index+2:]

    prompt_question = prompt_question+'Description of the Input Image: ' + input_image_description + '\n'

    prompt_question = prompt_question + 'Human Feedback: ' + human_feedback + '\n'

    our_result=d_results[len(d_program)]
    index=our_result.find('FINAL_RESULT: ')
    prompt_question = prompt_question + 'Our Wrong Answer: ' + our_result[index+14:] + '\n\n'

    if step_num==1:
        prompt_checked=None
        prompt_unchecked='Step1'+'\n'
        prompt_unchecked = prompt_unchecked + 'SubQuestion: ' + d_subquestion[0] +'\n'
        prompt_unchecked = prompt_unchecked + 'Program: ' + d_program[0] +'\n'
        if 'The description of' in d_results[1]:
            prompt_unchecked = prompt_unchecked + d_results[1] +'\n'
        else:
            prompt_unchecked = prompt_unchecked + 'Result of ' + d_results[1] +'\n'
    else:
        prompt_checked=''
        i=0
        while(i<step_num-1):
            prompt_checked = prompt_checked + 'Step'+str(i+1)+'\n'
            prompt_checked = prompt_checked + 'SubQuestion: ' + d_subquestion[i] +'\n'
            prompt_checked = prompt_checked + 'Program: ' + d_program[i] +'\n'
            if 'The description of' in d_results[i+1]:
                prompt_checked = prompt_checked + d_results[i+1] +'\n'
            else:
                prompt_checked = prompt_checked + 'Result of ' + d_results[i+1] +'\n'    
            i=i+1   
        prompt_unchecked='Step'+str(step_num)+'\n'
        prompt_unchecked = prompt_unchecked + 'SubQuestion: ' + d_subquestion[step_num-1] +'\n'
        prompt_unchecked = prompt_unchecked + 'Program: ' + d_program[step_num-1] +'\n'
        if 'The description of' in d_results[step_num]:
            prompt_unchecked = prompt_unchecked + d_results[step_num] +'\n'
        else:
            prompt_unchecked = prompt_unchecked + 'Result of ' + d_results[step_num] +'\n'

    return prompt_question, prompt_checked, prompt_unchecked




def divide_by_step_nlvr(question, human_feedback, subquestion, program, results, prog_step):
    # d_subquestion=subquestion.split('\n')
    d_program=program.split('\n')
    d_results=results.split('\n')

    prompt = 'Question: ' + question+ '\n'

    left_image_description=d_results[0]
    left_image_description_index=left_image_description.find(': ')
    left_image_description=left_image_description[left_image_description_index+2:]
    prompt = prompt+'Description of the Left Image: ' + left_image_description + '\n'


    right_image_description=d_results[1]
    right_image_description_index=right_image_description.find(': ')
    right_image_description=right_image_description[right_image_description_index+2:]
    prompt = prompt+'Description of the Right Image: ' + right_image_description + '\n'


    prompt = prompt + 'Human Feedback: ' + human_feedback + '\n'

    our_result=d_results[len(d_program)+1]
    index=our_result.find('FINAL_ANSWER: ')
    prompt = prompt + 'Our Wrong Answer: ' + our_result[index+14:] + '\n'

    prompt = prompt + 'Following are the decomposed subquestion, used program, and obtained result in each step. \n'

    prompt = prompt + 'subquestion: \n' + subquestion+ '\n' + 'Program and obtained result in each step: \n'

    for i in range(len(d_program)):
        prompt = prompt + 'Step'+str(i+1)+'\n'
        prog=d_program[i]

        prompt = prompt + 'Program: ' + prog +'\n'

        if ('The description of' in d_results[i+2]) or ('The coordinate of' in d_results[i+2]):
            prompt = prompt + d_results[i+2] +'\n'
        elif ('result of' in d_results[i+2]) or ('Result of' in d_results[i+2]):
            prompt = prompt + d_results[i+2] +'\n'
        else:
            prompt = prompt + 'Result of ' + d_results[i+2] +'\n'

    return prompt



def divide_by_stepbystep_nlvr(question, human_feedback, subquestion, program, results, prog_step, step_num):
    d_subquestion=subquestion.split('\n')
    d_program=program.split('\n')
    d_results=results.split('\n')

    prompt_question = 'Question: ' + question+ '\n'

    left_image_description=d_results[0]
    left_image_description_index=left_image_description.find(': ')
    left_image_description=left_image_description[left_image_description_index+2:]
    prompt_question = prompt_question+'Description of the Left Image: ' + left_image_description + '\n'


    right_image_description=d_results[1]
    right_image_description_index=right_image_description.find(': ')
    right_image_description=right_image_description[right_image_description_index+2:]
    prompt_question = prompt_question+'Description of the Right Image: ' + right_image_description + '\n'

    prompt_question = prompt_question + 'Human Feedback: ' + human_feedback + '\n'

    our_result=d_results[len(d_program)+1]
    index=our_result.find('FINAL_ANSWER: ')
    prompt_question = prompt_question + 'Our Wrong Answer: ' + our_result[index+14:] + '\n\n'


    if step_num==1:
        prompt_checked=None
        prompt_unchecked='Step1'+'\n'
        prompt_unchecked = prompt_unchecked + 'SubQuestion: ' + d_subquestion[0] +'\n'
        prompt_unchecked = prompt_unchecked + 'Program: ' + d_program[0] +'\n'
        if 'The description of' in d_results[2]:
            prompt_unchecked = prompt_unchecked + d_results[2] +'\n'
        else:
            prompt_unchecked = prompt_unchecked + 'Result of ' + d_results[2] +'\n'
    else:
        prompt_checked=''
        i=0
        while(i<step_num-1):
            prompt_checked = prompt_checked + 'Step'+str(i+1)+'\n'
            prompt_checked = prompt_checked + 'SubQuestion: ' + d_subquestion[i] +'\n'
            prompt_checked = prompt_checked + 'Program: ' + d_program[i] +'\n'
            if 'The description of' in d_results[i+2]:
                prompt_checked = prompt_checked + d_results[i+2] +'\n'
            else:
                prompt_checked = prompt_checked + 'Result of ' + d_results[i+2] +'\n'    
            i=i+1   
        prompt_unchecked='Step'+str(step_num)+'\n'
        prompt_unchecked = prompt_unchecked + 'SubQuestion: ' + d_subquestion[step_num-1] +'\n'
        prompt_unchecked = prompt_unchecked + 'Program: ' + d_program[step_num-1] +'\n'
        if 'The description of' in d_results[step_num]:
            prompt_unchecked = prompt_unchecked + d_results[step_num] +'\n'
        else:
            prompt_unchecked = prompt_unchecked + 'Result of ' + d_results[step_num] +'\n'

    return prompt_question, prompt_checked, prompt_unchecked



def divide_by_step_imgedit(question, feedback, subquestion, program, results, prog_step):
    # d_subquestion=subquestion.split('\n')
    d_program=program.split('\n')
    d_results=results.split('\n')

    prompt = 'Question: ' + question+ '\n'

    input_image_description=d_results[0]
    input_image_description_index=input_image_description.find(': ')
    input_image_description=input_image_description[input_image_description_index+2:]
    prompt = prompt+'Description of the Input Image: ' + input_image_description + '\n'

    prompt = prompt + 'Human Feedback: ' + feedback + '\n'

    our_result=d_results[len(d_program)]
    index=our_result.find('FINAL_RESULT: ')
    # prompt = prompt + 'Our Wrong Answer: ' + our_result[index+14:] + '\n'

    prompt = prompt + 'Following are the decomposed subquestion, used program, and obtained result in each step. \n'

    prompt = prompt + 'subquestion: \n' + subquestion+ '\n' + 'Program and obtained result in each step: \n'

    for i in range(len(d_program)):
        prompt = prompt + 'Step'+str(i+1)+'\n'
        prog=d_program[i]
        prompt = prompt + 'Program: ' + prog +'\n'

        if ('The description of' in d_results[i+1]) or ('The coordinate of' in d_results[i+1]):
            prompt = prompt + d_results[i+1] +'\n'
        else:
            prompt = prompt + 'Result of ' + d_results[i+1] +'\n'

    return prompt



def divide_by_stepbystep_imgedit(question, human_feedback, subquestion, program, results, prog_step, step_num):
    d_subquestion=subquestion.split('\n')
    d_program=program.split('\n')
    d_results=results.split('\n')

    prompt_question = 'Question: ' + question+ '\n'

    input_image_description=d_results[0]
    input_image_description_index=input_image_description.find(': ')
    input_image_description=input_image_description[input_image_description_index+2:]

    prompt_question = prompt_question+'Description of the Input Image: ' + input_image_description + '\n'

    prompt_question = prompt_question + 'Human Feedback: ' + human_feedback + '\n'

    our_result=d_results[len(d_program)]
    index=our_result.find('FINAL_RESULT: ')
    if 'str' in str(type(our_result)):
        prompt_question = prompt_question + 'Our Wrong Answer: ' + our_result[index+14:] + '\n\n'

    if step_num==1:
        prompt_checked=None
        prompt_unchecked='Step1'+'\n'
        prompt_unchecked = prompt_unchecked + 'SubQuestion: ' + d_subquestion[0] +'\n'
        prompt_unchecked = prompt_unchecked + 'Program: ' + d_program[0] +'\n'
        if 'The description of' in d_results[1]:
            prompt_unchecked = prompt_unchecked + d_results[1] +'\n'
        else:
            prompt_unchecked = prompt_unchecked + 'Result of ' + d_results[1] +'\n'
    else:
        prompt_checked=''
        i=0
        while(i<step_num-1):
            prompt_checked = prompt_checked + 'Step'+str(i+1)+'\n'
            prompt_checked = prompt_checked + 'SubQuestion: ' + d_subquestion[i] +'\n'
            prompt_checked = prompt_checked + 'Program: ' + d_program[i] +'\n'
            if 'The description of' in d_results[i+1]:
                prompt_checked = prompt_checked + d_results[i+1] +'\n'
            else:
                prompt_checked = prompt_checked + 'Result of ' + d_results[i+1] +'\n'    
            i=i+1   
        prompt_unchecked='Step'+str(step_num)+'\n'
        prompt_unchecked = prompt_unchecked + 'SubQuestion: ' + d_subquestion[step_num-1] +'\n'
        prompt_unchecked = prompt_unchecked + 'Program: ' + d_program[step_num-1] +'\n'
        if 'The description of' in d_results[step_num]:
            prompt_unchecked = prompt_unchecked + d_results[step_num] +'\n'
        else:
            prompt_unchecked = prompt_unchecked + 'Result of ' + d_results[step_num] +'\n'

    return prompt_question, prompt_checked, prompt_unchecked



def divide_by_step_knowtag(question, feedback, subquestion, program, results, prog_step):
    # d_subquestion=subquestion.split('\n')
    d_program=program.split('\n')
    d_results=results.split('\n')

    prompt = 'Question: ' + question+ '\n'

    input_image_description=d_results[0]
    input_image_description_index=input_image_description.find(': ')
    input_image_description=input_image_description[input_image_description_index+2:]

    prompt = prompt+'Description of the Input Image: ' + input_image_description + '\n'

    prompt = prompt + 'Human Feedback: ' + feedback + '\n'

    our_result=d_results[len(d_program)]

    prompt = prompt + 'Following are the decomposed subquestion, used program, and obtained result in each step. \n'

    prompt = prompt + 'subquestion: \n' + subquestion+ '\n' + 'Program and obtained result in each step: \n'

    for i in range(len(d_program)):
        prompt = prompt + 'Step'+str(i+1)+'\n'
        prog=d_program[i]
        prompt = prompt + 'Program: ' + prog +'\n'

        if ('The description of' in d_results[i+1]) or ('The coordinate of' in d_results[i+1]):
            prompt = prompt + d_results[i+1] +'\n'
        else:
            prompt = prompt + 'Result of ' + d_results[i+1] +'\n'

    return prompt




def divide_by_stepbystep_knowtag(question, human_feedback, subquestion, program, results, prog_step, step_num):
    d_subquestion=subquestion.split('\n')
    d_program=program.split('\n')
    d_results=results.split('\n')

    prompt_question = 'Question: ' + question+ '\n'

    input_image_description=d_results[0]
    input_image_description_index=input_image_description.find(': ')
    input_image_description=input_image_description[input_image_description_index+2:]

    prompt_question = prompt_question+'Description of the Input Image: ' + input_image_description + '\n'

    prompt_question = prompt_question + 'Human Feedback: ' + human_feedback + '\n'

    our_result=d_results[len(d_program)]
    index=our_result.find('FINAL_RESULT: ')
    if 'str' in str(type(our_result)):
        prompt_question = prompt_question + 'Our Wrong Answer: ' + our_result[index+14:] + '\n\n'

    if step_num==1:
        prompt_checked=None
        prompt_unchecked='Step1'+'\n'
        prompt_unchecked = prompt_unchecked + 'SubQuestion: ' + d_subquestion[0] +'\n'
        prompt_unchecked = prompt_unchecked + 'Program: ' + d_program[0] +'\n'
        if 'The description of' in d_results[1]:
            prompt_unchecked = prompt_unchecked + d_results[1] +'\n'
        else:
            prompt_unchecked = prompt_unchecked + 'Result of ' + d_results[1] +'\n'
    else:
        prompt_checked=''
        i=0
        while(i<step_num-1):
            prompt_checked = prompt_checked + 'Step'+str(i+1)+'\n'
            prompt_checked = prompt_checked + 'SubQuestion: ' + d_subquestion[i] +'\n'
            prompt_checked = prompt_checked + 'Program: ' + d_program[i] +'\n'
            if 'The description of' in d_results[i+1]:
                prompt_checked = prompt_checked + d_results[i+1] +'\n'
            else:
                prompt_checked = prompt_checked + 'Result of ' + d_results[i+1] +'\n'    
            i=i+1   
        prompt_unchecked='Step'+str(step_num)+'\n'
        prompt_unchecked = prompt_unchecked + 'SubQuestion: ' + d_subquestion[step_num-1] +'\n'
        prompt_unchecked = prompt_unchecked + 'Program: ' + d_program[step_num-1] +'\n'
        if 'The description of' in d_results[step_num]:
            prompt_unchecked = prompt_unchecked + d_results[step_num] +'\n'
        else:
            prompt_unchecked = prompt_unchecked + 'Result of ' + d_results[step_num] +'\n'

    return prompt_question, prompt_checked, prompt_unchecked




INTERMEDIATE_FUNC={
"gqa": intermediate_gqa, 
"nlvr": intermediate_nlvr,
"imgedit": intermediate_imgedit,
"knowtag": intermediate_knowtag
    }


DIVIDE_BY_STEP_FUNC={
"gqa": divide_by_step_gqa, 
"nlvr": divide_by_step_nlvr,
"imgedit": divide_by_step_imgedit,
"knowtag": divide_by_step_knowtag   
}



DIVIDE_BY_STEPBYSTEP_FUNC={
"gqa": divide_by_stepbystep_gqa, 
"nlvr": divide_by_stepbystep_nlvr,
"imgedit": divide_by_stepbystep_imgedit,
"knowtag": divide_by_stepbystep_knowtag   
}