
def update_vqa_model(interpreter, generator_react, subq, prog, init_state, prog_state, error_location, reason, gt_answer):

    vqa_list=interpreter.search_vqa(prog, init_state)

    print ('!!!!!!!prog!!!!!!!')
    print (prog)
    print ('!!!!!!!prog!!!!!!!')

    print ('!!!!!!!vqa_list!!!!!!!')
    print (vqa_list)
    print ('!!!!!!!vqa_list!!!!!!!')

    print ('!!!!!!!prog_state!!!!!!!')
    print (prog_state)
    print ('!!!!!!!prog_state!!!!!!!')

    #######---------------code to update the visual model
    if 'function' not in error_location:
        for i in range (len(vqa_list)):
            img=prog_state[vqa_list[i]['img_var']]
            question=vqa_list[i]['question']
            answer=prog_state[vqa_list[i]['output_var']]
            try:
                interpreter.update_vqavisualmodel('VQA', correct=True, data=dict(image=img, question=question, answer=answer))
            except:
                print ('no image in vqa update')

    else:
        if 'VQA' in reason:
            step_location=reason.find('Step')
            start_location=step_location+4
            end_location=start_location
            while (1):
                if reason[end_location]>='0' and reason[end_location]<='9':
                    end_location=end_location+1
                else:
                    break
            if start_location==end_location:
                errorstep=vqa_list[0]['step']
            else:
                errorstep=int(reason[start_location:end_location])
            print ('error step:', errorstep)


            for i in range (len(vqa_list)):
                if int(vqa_list[i]['step'])!=errorstep:
                    img=prog_state[vqa_list[i]['img_var']]
                    question=vqa_list[i]['question']
                    answer=prog_state[vqa_list[i]['output_var']]
                    try:
                        interpreter.update_vqavisualmodel('VQA', correct=True, data=dict(image=img, question=question, answer=answer))
                    except:
                        print ('no image in vqa update')                        
                else:
                    img=prog_state[vqa_list[i]['img_var']]
                    question=vqa_list[i]['question']
                    answer=prog_state[vqa_list[i]['output_var']]

                    inferenced_answer=generator_react.answer_inference(dict(question=question, human_feedback=f'the correct answer should be {gt_answer}', subquestion=subq, program=prog, errorlocation=error_location, reason=reason), prog_state)
                    inferenced_answer=inferenced_answer.lstrip(' ').rstrip(' ')
                    print ('----------------------re-answer start----------------------')
                    print (inferenced_answer)
                    print ('----------------------re-answer end----------------------')

                    vqa_react_prog_state=prog_state.copy()
                    for i in range (len(vqa_list)):
                        if int(vqa_list[i]['step'])==errorstep:
                            vqa_react_prog_state[vqa_list[i]['output_var']]=inferenced_answer

                    vqa_react_prog=prog
                    vqa_react_prog=vqa_react_prog.split('\n')
                    vqa_react_prog=vqa_react_prog[errorstep:]
                    vqa_react_prog = '\n'.join(vqa_react_prog)
                    vqa_react_prog=vqa_react_prog.rstrip('\n')

                    print ('----------------------vqa_react start----------------------')
                    print ('vqa_react_prog', vqa_react_prog)
                    print ('vqa_react_prog_state', vqa_react_prog_state)
                    print ('----------------------vqa_react end----------------------')


                    if 'FACEDET' in vqa_react_prog:
                        is_face = True
                    else:
                        is_face = False
                    try:
                        vqa_react_result, _ = interpreter.execute(vqa_react_prog, vqa_react_prog_state, inspect=False, is_face=is_face)
                    except:
                        vqa_react_result=''

                    try:
                        vqa_react_result=vqa_react_result.lstrip('\n').rstrip('\n').lower()
                    except:
                        print ('vqa_react_result is not a string')  

                    print ('vqa_react_result',vqa_react_result)
                    if (vqa_react_result==gt_answer):
                        try:
                            print ('update LOC immediately by prompt tuning')
                            interpreter.update_vqavisualmodel('VQA', correct=False, data=dict(image=img, question=question, answer=inferenced_answer))
                        except:
                            print ('no image in vqa update')



def update_loc_model(interpreter, part_reflectioner, subq, prog, init_state, prog_state, error_location, reason, gt_answer):

    loc_list=interpreter.search_loc(prog, init_state)

    #######---------------code to update the visual model

    step_location=reason.find('Step')
    start_location=step_location+4
    end_location=start_location
    while (1):
        if reason[end_location]>='0' and reason[end_location]<='9':
            end_location=end_location+1
        else:
            break
    if start_location==end_location:
        errorstep=loc_list[0]['step']
    else:
        errorstep=int(reason[start_location:end_location])
    print ('error step:', errorstep)

    for i in range (len(loc_list)):
        if (loc_list[i]['step'])==errorstep:
            img=prog_state[loc_list[i]['img_var']]
            object1=loc_list[i]['object']
            interpreter.update_locvisualmodel('LOC', object1.strip("'"))
            print ('----------update LOC finish----------')



def update_seg_model(interpreter, part_reflectioner, subq, prog, init_state, prog_state, error_location, reason, gt_answer):

    replace_list=interpreter.search_replace(prog, init_state)
    print ('replace_list',replace_list)
    print ('----------start udpate the SEG model----------')

    # find which step
    step_location=reason.find('Step')
    start_location=step_location+4
    end_location=start_location
    while (1):
        if reason[end_location]>='0' and reason[end_location]<='9':
            end_location=end_location+1
        else:
            break
    if start_location==end_location:
        errorstep=replace_list[0]['step']
    else:
        errorstep=int(reason[start_location:end_location])
    print ('error step:', errorstep)

    query=replace_list[0]['query']
    try:
        interpreter.update_segvisualmodel('SEG', query.strip("'"))
    except:
        print ("something wrong when updating SEG")   



def update_select_model(interpreter, part_reflectioner, subq, prog, init_state, prog_state, error_location, reason, gt_answer):

    select_list=interpreter.search_select(prog, init_state)
    print ('select_list',select_list)
    print ('-------------we well update the SELECT tool----------')

    # find which step
    step_location=reason.find('Step')
    start_location=step_location+4
    end_location=start_location
    while (1):
        if reason[end_location]>='0' and reason[end_location]<='9':
            end_location=end_location+1
        else:
            break
    if start_location==end_location:
        errorstep=select_list[0]['step']
    else:
        errorstep=int(reason[start_location:end_location])
    print ('error step:', errorstep)


    for i in range (len(select_list)):
        if (select_list[i]['step'])==errorstep:
            query=select_list[i]['query']
            print ('----------start udpate the SELECT model----------')
            print ('----------query----------',query)
            try:
                interpreter.update_selectvisualmodel('SELECT', query.strip("'"))
            except:
                print ("something wrong when updating SELECT")



def update_replace_model(interpreter, part_reflectioner, subq, prog, init_state, prog_state, error_location, reason, gt_answer):

    replace_list=interpreter.search_replace(prog, init_state)
    print ('replace_list',replace_list)
    print ('----------start udpate the REPLACE model----------')


    # find which step
    step_location=reason.find('Step')
    start_location=step_location+4
    end_location=start_location
    while (1):
        if reason[end_location]>='0' and reason[end_location]<='9':
            end_location=end_location+1
        else:
            break
    if start_location==end_location:
        errorstep=replace_list[0]['step']
    else:
        errorstep=int(reason[start_location:end_location])
    print ('error step:', errorstep)


    for i in range (len(replace_list)):
        if (replace_list[i]['step'])==errorstep:
            query=replace_list[i]['query']
            try:
                interpreter.update_replacevisualmodel('REPLACE', query.strip("'"))
            except:
                print ("something wrong when updating REPLACE") 



def update_classify_model(interpreter, part_reflectioner, subq, prog, init_state, prog_state, error_location, reason, gt_answer):

    if 'FACEDET' in prog:
        category_name=''
    elif 'LOC' in prog:
        loc_list=interpreter.search_loc(prog, init_state)
        category_name=loc_list[0]['object']
    else:
        category_name=''

    list_list=interpreter.search_list(prog, init_state)
    
    if 'LIST' in prog:
        print ('-------------list is in updaing CLASSIFY----------')
        for list_step in list_list:
            query=list_step['query']
            list_max = list_step['list_max']
            result=interpreter.step_interpreters['LIST'].get_list(query.strip("'"),list_max)
            print ('list result in update classifiy', result)
            interpreter.update_classifyvisualmodel('CLASSIFY', result, category_name)
    else:
        print ('-------------list is not in updaing CLASSIFY----------')
        classify_list=interpreter.search_classify(prog, init_state)
        for classify_step in classify_list:
            categories=classify_step['categories']
            interpreter.update_classifyvisualmodel('CLASSIFY', [categories], category_name)