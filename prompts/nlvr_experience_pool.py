FAILED_SUBQUESTION=[
"""Question: There are two pairs of hands wearing gloves.
SubQuesion:
Step1, Asking the left image, 'How many pairs of hands are in the image?'.
Step2, Asking the right image, 'How many pairs of hands are in the image?'.
Step3, Asking the left image, 'Are the hands wearing gloves?'.
Step4, Asking the right image, 'Are the hands wearing gloves?'.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number obtained in Step1 and answers obtained in Step3. If the number is equal to '2' and the answer is equal to 'yes', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step6, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number obtained in Step2 and answers obtained in Step4. If the number is equal to '2' and the answer is equal to 'yes', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step7, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step5 and Step6. If one of the two answers is equal to 'yes' and the rest one answer is equal to 'no', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step8, Visualize results.
Reason:
In Step5 of the subquestions, the subquestions judge whether there are two pairs of hands in the left image. It is wrong, the subquestions should judge whether the sum of pairs of hands in the two images are equal to two, and then judge whether the two pairs of hands are wearing gloves.
""",
]


FAILED_PROGRAM=[
"""
Question: A mitten is being worn in one image and the mittens are not being worn in the other image.
SubQuesion:
Step1, Asking the left image, 'Is a mitten being worn in the image?'.
Step2, Asking the right image, 'Is a mitten being worn in the image?'.
Step3, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step1 and Step2. If one of the two answers is equal to 'yes' and the rest one answer is equal to 'no', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step4, Visualize results.
Program:
ANSWER0=VQA(image=LEFT,question='Is a mitten being worn in the image?')
ANSWER1=VQA(image=RIGHT,question='Is a mitten being worn in the image?')
ANSWER2=EVAL(expr='{ANSWER0} and {ANSWER1}')
FINAL_ANSWER=RESULT(var=ANSWER2)
Reason: The subquestions are correct, and can address the given question. But the Step3 of the program does not match the subquestion. The program should use 'xor' instead of 'and'.
""",
"""
Question: A mitten is being worn in one image and the mittens are not being worn in the other image.
SubQuesion:
Step1, Asking the left image, 'Is a mitten being worn in the image?'.
Step2, Asking the right image, 'Is a mitten being worn in the image?'.
Step3, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step1 and Step2. If one of the two answers is equal to 'yes' and the rest one answer is equal to 'no', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step4, Visualize results.
Program:
ANSWER0=VQA(image=LEFT,question='Is a mitten being worn in the image?')
ANSWER1=VQA(image=RIGHT,question='Are the mittens being worn in the image?')
ANSWER2=EVAL(expr='{ANSWER0} xor {ANSWER1}')
FINAL_ANSWER=RESULT(var=ANSWER2)
Reason: The subquestions are correct, and can address the given question. But the Step2 of the program does not match the subquestion. The program should ask 'Is a mitten being worn in the image?' to correspond to the subquestion in Step2.
""",
]


CURATED_SUBQUESTION=[
"""Question: An image shows one bare hand with the thumb on the right holding up a belly-first, head-up crab, with water in the background.
SubQuestion:
Step1, Asking the left image, 'Does the image shows one bare hand with the thumb on the right holding a crab?'.
Step2, Asking the right image, 'Does the image shows one bare hand with the thumb on the right holding a crab?'.
Step3, Asking the left image, 'Is the crab belly-first and head-ups?'.
Step4, Asking the right image, 'Is the crab belly-first and head-ups?'.
Step5, Asking the left image, 'Is there water in the background?'.
Step6, Asking the right image, 'Is there water in the background?'.
Step7, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step1, Step3, and Step5. If all answers obtained in Step1, Step3, and Step5 are 'yes', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step8, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step2, Step4, and Step6. If all answers obtained in Step2, Step4, and Step6 are 'yes', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step9, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step7 and Step8. If one of the two answers is equal to 'yes' and the rest one answer is equal to 'no', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step10, Visualize results.
""",
"""Question: One dog is laying down.
SubQuestion:
Step1, Asking the left image, 'How many dogs are laying down?'.
Step2, Asking the right image, 'How many dogs are laying down?'.
Step3, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the numbers of dogs obtained in Step1 and Step2. If the sum of the two numbers obtained in Step1 and Step2 is equal to '1', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step4, Visualize results.
""",
"""Question: There are two blue and yellow birds
SubQuestion:
Step1, Asking the left image, 'How many blue and yellow birds are in the image?'.
Step2, Asking the right image, 'How many blue and yellow birds are in the image?'.
Step3, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the numbers of birds obtained in Step1 and Step2. If the sum of the two numbers obtained in Step1 and Step2 is equal to '2', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step4, Visualize results.
""",
"""Question: A single wolf is howling and silhouetted by the moon in one of the images.
SubQuestion:
Step1, Asking the left image, 'How many wolves are in the image?'.
Step2, Asking the right image, 'How many wolves are in the image?'.
Step3, Asking the left image, 'Is the wolf howling and silhouetted by the moon?'.
Step4, Asking the right image, 'Is the wolf howling and silhouetted by the moon?'.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number obtained in Step1 and answers obtained in Step3. If the number is equal to '1' and the answer is equal to 'yes', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step6, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number obtained in Step2 and answers obtained in Step4. If the number is equal to '1' and the answer is equal to 'yes', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step7, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step5 and Step6. If one of the two answers is equal to 'yes' and the rest one answer is equal to 'no', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step8, Visualize results.
""",
"""Question: One of the two images has a bag with the characters from Disney's Frozen on it.
SubQuestion:
Step1, Asking the left image, 'Does the image have a bag with the characters from Disney's Frozen on it?'.
Step2, Asking the right image, 'Does the image have a bag with the characters from Disney's Frozen on it?'.
Step3, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step1 and Step2. If one of the two answers is equal to 'yes' and the rest one answer is equal to 'no', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step4, Visualize results.
""",
"""Question: An image shows broccoli growing in soil, with leaves surrounding the florets.
SubQuestion:
Step1, Asking the left image, 'Does the image show broccoli growing in soil?'.
Step2, Asking the right image, 'Does the image show broccoli growing in soil?'.
Step3, Asking the left image, 'Are leaves surrounding the floret?'.
Step4, Asking the right image, 'Are leaves surrounding the floret?'.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step1 and Step3. If the two answers are both equal to 'yes', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step2 and Step4. If the two answers are both equal to 'yes', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step7, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step5 and Step6. If one of the two answers is equal to 'yes' and the rest one answer is equal to 'no', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step8, Visualize results.
""",
"""Question: An image shows exactly two seals in direct contact, posed face to face.
SubQuestion:
Step1, Asking the left image, 'How many seals are in the image?'.
Step2, Asking the right image, 'How many seals are in the image?'.
Step3, Asking the left image, 'Are the seals in direct contact?'.
Step4, Asking the right image, 'Are the seals in direct contact?'.
Step5, Asking the left image, 'Are the seals posed face to face?'.
Step6, Asking the right image, 'Are the seals posed face to face?'.
Step7, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step1, Step3, and Step5. If the answer obtained in Step1 is equal to '2', and the two answers obtained in Step3 and Step5 are both equal to 'yes', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step8, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step2, Step4, and Step6. If the answer obtained in Step2 is equal to '2', and the two answers obtained in Step4 and Step6 are both equal to 'yes', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step9, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step7 and Step8. If one of the two answers is equal to 'yes' and the rest one answer is equal to 'no', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step10, Visualize results.
""",
"""Question: There is at least two parrots in the right image.
SubQuestion:
Step1, Asking the left image, 'How many parrots are in the image?'.
Step2, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answer obtained in Step1. If the answer obtained in Step1 is equal to or greater than '2', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step3, Visualize results.
""",
"""Question: There are two wolves in each image.
SubQuestion:
Step1, Asking the left image, 'How many wolves are in the image?'.
Step2, Asking the left image, 'How many wolves are in the image?'.
Step3, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answer obtained in Step1 and Step2. If the two answers obtained in Step1 and Step2 are both equal to '2', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step4, Visualize results.
"""
]


CURATED_PROGRAMS=[
"""Question: There is a red convertible in one image.
SubQuestion:
Step1, Asking the left image, 'Is there a red convertible in the image?'.
Step2, Asking the right image, 'Is there a red convertible in the image?'.
Step3, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step1 and Step2. If one of the two answers is equal to 'yes' and the rest one answer is equal to 'no', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step4, Visualize results.
Program:
ANSWER0=VQA(image=LEFT,question='Is there a red convertible in the image?')
ANSWER1=VQA(image=RIGHT,question='Is there a red convertible in the image?')
ANSWER2=EVAL(expr='{ANSWER0} xor {ANSWER1}')
FINAL_ANSWER=RESULT(var=ANSWER2)
""",
"""Question: One dog is laying down.
SubQuestion:
Step1, Asking the left image, 'How many dogs are laying down?'.
Step2, Asking the right image, 'How many dogs are laying down?'.
Step3, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the numbers of dogs obtained in Step1 and Step2. If the sum of the two numbers obtained in Step1 and Step2 is equal to '1', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step4, Visualize results.
Program:
ANSWER0=VQA(image=LEFT,question='How many dogs are laying down?')
ANSWER1=VQA(image=RIGHT,question='How many dogs are laying down?')
ANSWER2=EVAL(expr='{ANSWER0} + {ANSWER1} == 1')
FINAL_ANSWER=RESULT(var=ANSWER2)
""",
"""Question: There are two blue and yellow birds
SubQuestion:
Step1, Asking the left image, 'How many blue and yellow birds are in the image?'.
Step2, Asking the right image, 'How many blue and yellow birds are in the image?'.
Step3, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the numbers of birds obtained in Step1 and Step2. If the sum of the two numbers obtained in Step1 and Step2 is equal to '2', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step4, Visualize results.
Program:
ANSWER0=VQA(image=LEFT,question='How many blue and yellow birds are in the image?')
ANSWER1=VQA(image=RIGHT,question='How many blue and yellow birds are in the image?')
ANSWER2=EVAL(expr='{ANSWER0} + {ANSWER1} == 2')
FINAL_ANSWER=RESULT(var=ANSWER2)
""",
"""Question: One of the two images has a bag with the characters from Disney's Frozen on it.
SubQuestion:
Step1, Asking the left image, 'Does the image have a bag with the characters from Disney's Frozen on it?'.
Step2, Asking the right image, 'Does the image have a bag with the characters from Disney's Frozen on it?'.
Step3, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step1 and Step2. If one of the two answers is equal to 'yes' and the rest one answer is equal to 'no', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step4, Visualize results.
Program:
ANSWER0=VQA(image=LEFT,question='Does the image have a bag with the characters from Disney's Frozen on it?')
ANSWER1=VQA(Image=RIGHT,question='Does the image have a bag with the characters from Disney's Frozen on it?')
ANSWER2=EVAL(expr='{ANSWER0} xor {ANSWER1}')
FINAL_ANSWER=RESULT(var=ANSWER2)
""",
"""Question: There is at least two parrots in the right image.
SubQuestion:
Step1, Asking the left image, 'How many parrots are in the image?'.
Step2, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answer obtained in Step1. If the answer obtained in Step1 is equal to or greater than '2', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step3, Visualize results.
Program:
ANSWER0=VQA(image=RIGHT,question='How many parrots are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} >= 2')
FINAL_ANSWER=RESULT(var=ANSWER1)
""",
"""Question: There are two wolves in each image.
SubQuestion:
Step1, Asking the left image, 'How many wolves are in the image?'.
Step2, Asking the left image, 'How many wolves are in the image?'.
Step3, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answer obtained in Step1 and Step2. If the two answers obtained in Step1 and Step2 are both equal to '2', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step4, Visualize results.
Program:
ANSWER0=VQA(image=LEFT,question='How many wolves are in the image?')
ANSWER1=VQA(image=RIGHT,question='How many wolves are in the image?')
ANSWER2=EVAL(expr='{ANSWER0} == 2 and {ANSWER1} == 2')
FINAL_ANSWER=RESULT(var=ANSWER2)
"""
]


REFLECTION_STEP=[
"""Question: There are two pairs of hands wearing gloves.
The description of left image: a photography of a pair of hands wearing pink gloves.
The description of right image: a photography of a pair of hands wearing white gloves.
Human Feedback: the correct answer is True
Our Wrong Answer: False
SubQuesion:
Step1, Asking the left image, 'How many pairs of hands are in the image?'.
Step2, Asking the right image, 'How many pairs of hands are in the image?'.
Step3, Asking the left image, 'Are the hands wearing gloves?'.
Step4, Asking the right image, 'Are the hands wearing gloves?'.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number obtained in Step1 and answers obtained in Step3. If the number is equal to '1' and the answer is equal to 'yes', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step6, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number obtained in Step2 and answers obtained in Step4. If the number is equal to '1' and the answer is equal to 'yes', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step7, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step5 and Step6. If one of the two answers is equal to 'yes' and the rest one answer is equal to 'no', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step8, Visualize results.
Program and obtained result in each step:
Step1
Program: ANSWER0=VQA(image=LEFT,question='How many pairs of hands are in the image?')
Results of ANSWER0: 1
Step2
Program: ANSWER1=VQA(image=RIGHT,question='How many pairs of hands are in the image?')
Results of ANSWER1: 1
Step3
Program: ANSWER2=VQA(image=LEFT,question='Are the hands wearing gloves?')
Results of ANSWER2: yes
Step4
Program: ANSWER3=VQA(image=RIGHT,question='Are the hands wearing gloves?')
Results of ANSWER3: no
Step5
Program: ANSWER4=EVAL(expr='{ANSWER0} == 1 and {ANSWER2}')
Results of ANSWER4: True
Step6
Program: ANSWER5=EVAL(expr='{ANSWER1} == 1 and {ANSWER3}')
Results of ANSWER5: False
Step7
Program: ANSWER6=EVAL(expr='{ANSWER4} xor {ANSWER5}')
Results of ANSWER6: False
Step8
Program: FINAL_ANSWER=RESULT(var=ANSWER6)
Results of FINAL_ANSWER: False
Error Location: functions called by programs
Reason: In the Step4 of the program, the used function 'VQA' failed to recognize hands wearing gloves correctly, as the obtained result of ANSWER6 is 'no' instead of 'yes'.
""",
"""Question: There are two pairs of hands wearing gloves.
The description of left image: a photography of a pair of hands wearing pink gloves.
The description of right image: a photography of a pair of hands wearing white gloves.
Human Feedback: the correct answer is True
Our Wrong Answer: False
SubQuesion:
Step1, Asking the left image, 'How many pairs of hands are in the image?'.
Step2, Asking the right image, 'How many pairs of hands are in the image?'.
Step3, Asking the left image, 'Are the hands wearing gloves?'.
Step4, Asking the right image, 'Are the hands wearing gloves?'.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number obtained in Step1 and answers obtained in Step3. If the number is equal to '2' and the answer is equal to 'yes', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step6, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number obtained in Step2 and answers obtained in Step4. If the number is equal to '2' and the answer is equal to 'yes', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step7, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step5 and Step6. If one of the two answers is equal to 'yes' and the rest one answer is equal to 'no', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step8, Visualize results.
Program and obtained result in each step:
Step1
Program: ANSWER0=VQA(image=LEFT,question='How many pairs of hands are in the image?')
Results of ANSWER0: 1
Step2
Program: ANSWER1=VQA(image=RIGHT,question='How many pairs of hands are in the image?')
Results of ANSWER1: 1
Step3
Program: ANSWER2=VQA(image=LEFT,question='Are the hands wearing gloves?')
Results of ANSWER2: yes
Step4
Program: ANSWER3=VQA(image=RIGHT,question='Are the hands wearing gloves?')
Results of ANSWER3: yes
Step5
Program: ANSWER4=EVAL(expr='{ANSWER0} == 2 and {ANSWER2}')
Results of ANSWER4: False
Step6
Program: ANSWER5=EVAL(expr='{ANSWER1} == 2 and {ANSWER3}')
Results of ANSWER5: False
Step7
Program: ANSWER6=EVAL(expr='{ANSWER4} xor {ANSWER5}')
Results of ANSWER6: False
Step8
Program: FINAL_ANSWER=RESULT(var=ANSWER6)
Results of FINAL_ANSWER: False
Error Location: subquestions
Reason: In Step5 of the subquestions, the subquestions judge whether there are two pairs of hands in the left image. It is wrong, the subquestions should judge whether the sum of pairs of hands in the two images are equal to two, and then judge whether the two pairs of hands are wearing gloves.
""",
"""
Question: A mitten is being worn in one image and the mittens are not being worn in the other image.
The description of left image: a photography of a pair of pink gloves.
The description of right image: a photography of a pair of hands wearing white gloves.
Human Feedback: the correct answer is True
Our Wrong Answer: False
SubQuesion:
Step1, Asking the left image, 'Is a mitten being worn in the image?'.
Step2, Asking the right image, 'Is a mitten being worn in the image?'.
Step3, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step1 and Step2. If one of the two answers is equal to 'yes' and the rest one answer is equal to 'no', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step4, Visualize results.
Program and obtained result in each step:
Step1
ANSWER0=VQA(image=LEFT,question='Is a mitten being worn in the image?')
Result of ANSWER0: no
Step2
ANSWER1=VQA(image=RIGHT,question='Is a mitten being worn in the image?')
Result of ANSWER1: yes
Step3
ANSWER2=EVAL(expr='{ANSWER0} and {ANSWER1}')
Result of ANSWER2: False
Step4
FINAL_ANSWER=RESULT(var=ANSWER2)
Result of FINAL_ANSWER: False
Error Location: program
Reason: The subquestions are correct, and can address the given question. But the Step3 of the program does not match the subquestion. The program should use 'xor' instead of 'and'.
""",
"""
Question: A mitten is being worn in one image and the mittens are not being worn in the other image.
The description of left image: a photography of a pair of pink gloves.
The description of right image: a photography of a pair of hands wearing white gloves.
Human Feedback: the correct answer is True
Our Wrong Answer: False
SubQuesion:
Step1, Asking the left image, 'Is a mitten being worn in the image?'.
Step2, Asking the right image, 'Is a mitten being worn in the image?'.
Step3, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step1 and Step2. If one of the two answers is equal to 'yes' and the rest one answer is equal to 'no', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step4, Visualize results.
Program and obtained result in each step:
Step1
ANSWER0=VQA(image=LEFT,question='Is a mitten being worn in the image?')
Result of ANSWER0: no
Step2
ANSWER1=VQA(image=RIGHT,question='Are there mittens being worn in the image?')
Result of ANSWER1: no
Step3
ANSWER2=EVAL(expr='{ANSWER0} xor {ANSWER1}')
Result of ANSWER2: False
Step4
FINAL_ANSWER=RESULT(var=ANSWER2)
Result of FINAL_ANSWER: False
Error Location: program
Reason: The subquestions are correct, and can address the given question. But the Step2 of the program does not match the subquestion. The program should ask 'Is a mitten being worn in the image?' to correspond to the subquestion in Step2.
""",
]


REFLECTION_INTERRUPT=[
"""Question: A mitten is being worn in one image and the mittens are not being worn in the other image.
SubQuesion:
Step1, Asking the left image, 'Is a mitten being worn in the image?'.
Step2, Asking the right image, 'Is a mitten being worn in the image?'.
Step3, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step1 and Step2. If one of the two answers is equal to 'yes' and the rest one answer is equal to 'no', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step4, Visualize results.
Program:
ANSWER0=VQA(image=LEFT,question='Is a mitten being worn in the image?')
ANSWER1=VQA(image=RIGHT,question='Are the mittens being worn in the image?')
ANSWER2=EVAL(expr='{ANSWER0} xor {ANSWER3}')
FINAL_ANSWER=RESULT(var=ANSWER2)
Reason: 
The bug in the program is in the third line: ANSWER2=EVAL(expr='{ANSWER0} xor {ANSWER3}'), where 'ANSWER3' is not defined. It should be 'ANSWER1'.
""",
]


INFERENCE=[
"""Question: There are two pairs of hands wearing gloves.
The description of left image: a photography of a pair of hands wearing pink gloves.
The description of right image: a photography of a pair of hands wearing white gloves.
Human Feedback: the correct answer is True
Our Wrong Answer: False
Following are the decomposed subquestion, used program, and obtained result in each step. 
SubQuesion:
Step1, Asking the left image, 'How many pairs of hands are in the image?'.
Step2, Asking the right image, 'How many pairs of hands are in the image?'.
Step3, Asking the left image, 'Are the hands wearing gloves?'.
Step4, Asking the right image, 'Are the hands wearing gloves?'.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number obtained in Step1 and answers obtained in Step3. If the number is equal to '1' and the answer is equal to 'yes', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step6, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number obtained in Step2 and answers obtained in Step4. If the number is equal to '1' and the answer is equal to 'yes', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step7, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the answers obtained in Step5 and Step6. If one of the two answers is equal to 'yes' and the rest one answer is equal to 'no', the answer in this step is 'yes'; On the contrary, the answer is 'no'.
Step8, Visualize results.
Program and obtained result in each step:
Step1
Program: ANSWER0=VQA(image=LEFT,question='How many pairs of hands are in the image?')
Results of ANSWER0: 1
Step2
Program: ANSWER1=VQA(image=RIGHT,question='How many pairs of hands are in the image?')
Results of ANSWER1: 1
Step3
Program: ANSWER2=VQA(image=LEFT,question='Are the hands wearing gloves?')
Results of ANSWER2: yes
Step4
Program: ANSWER3=VQA(image=RIGHT,question='Are the hands wearing gloves?')
Results of ANSWER3: no
Step5
Program: ANSWER4=EVAL(expr='{ANSWER0} == 1 and {ANSWER2}')
Results of ANSWER4: True
Step6
Program: ANSWER5=EVAL(expr='{ANSWER1} == 1 and {ANSWER3}')
Results of ANSWER5: False
Step7
Program: ANSWER6=EVAL(expr='{ANSWER4} xor {ANSWER5}')
Results of ANSWER6: False
Step8
Program: FINAL_ANSWER=RESULT(var=ANSWER6)
Results of FINAL_ANSWER: False
Error Location: functions called by programs
Reason: In the Step4 of the program, the used function 'VQA' failed to recognize hands wearing gloves correctly, as the obtained result of ANSWER6 is 'no' instead of 'yes'.
Correct answer of the wrong step: yes
""",
]





