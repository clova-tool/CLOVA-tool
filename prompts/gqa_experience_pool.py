FAILED_SUBQUESTION=[
"""Question: Are there any windows in the picture that are not rectangular?
SubQuesion:
Step1, Locate windows in the given image, and obtain bounding boxes of windows.
Step2, Crop the image region of windows from the given image, based on bounding boxes of windows. The bounding boxes are obtained in Step1.
Step3, Asking the image region of windows, 'What shape is the window?'. The image region of windows is obtained in Step2.
Step4, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the intermediate answers obtained in Step3. If the shape is not equal to 'rectangular', the answer is 'yes'; On the contrary, the answer is 'no'.
Step5, Visualize results.
Reason:
In Step3 of the subquestions, the subquestions should ask 'Are the windows rectangular?', instead of asking 'What shape is the window?'. Then, the subquestions should further determine whether the answer is 'yes' or 'no' by executing Python expression, based on the intermediate answers obtained in Step3. If the answer is 'yes', the answer is 'no'; On the contrary, the answer is 'yes'.
""",
]


FAILED_PROGRAM=[
"""
Question: Is the lamp different in color than the shirt?
SubQuesion:
Step1, Locate the lamp in the given image, and obtain bounding boxes of lamp.
Step2, Crop the region of the lamp from the given image, based on bounding boxes of lamp. The bounding boxes are obtained in Step1.
Step3, Asking the image region of lamp, 'What color is the lamp?'. The image region of lamp is cropped in Step2.
Step4, Locate the shirt in the given image, and obtain bounding boxes of shirt. 
Step5, Crop the region of the shirt from the given image, based on bounding boxes of shirt. The bounding boxes are obtained in Step4.
Step6, Asking the image region of shirt, 'What color is the shirt?'. The image region of lamp is cropped in Step5.
Step7, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the color of lamp and the color of shirt. The color of lamp and shirt is obtained in Step3 and Step6, respectively. If their color are the same, the answer is 'yes'; On the contrary, the answer is 'no'.
Step8, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='lamp')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='shirt')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
Reason: The subquestions are correct, and can address the given question. But the Step3-Step6 of the program does not match the subquestion. The subquestions locate, crop, and ask color of the lamp and shirt, but the program counts the number of lamp.
""",
"""
Question: Is the lamp different in color than the shirt?
SubQuesion:
Step1, Locate the lamp in the given image, and obtain bounding boxes of lamp.
Step2, Crop the region of the lamp from the given image, based on bounding boxes of lamp. The bounding boxes are obtained in Step1.
Step3, Asking the image region of lamp, 'What color is the lamp?'. The image region of lamp is cropped in Step2.
Step4, Locate the shirt in the given image, and obtain bounding boxes of shirt. 
Step5, Crop the region of the shirt from the given image, based on bounding boxes of shirt. The bounding boxes are obtained in Step4.
Step6, Asking the image region of shirt, 'What color is the shirt?'. The image region of lamp is cropped in Step5.
Step7, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the color of lamp and the color of shirt. The color of lamp and shirt is obtained in Step3 and Step6, respectively. If their color are the same, the answer is 'yes'; On the contrary, the answer is 'no'.
Step8, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='lamp')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='shirt')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
Reason: The subquestions are correct, and can address the given question. But the Step3-Step6 of the program does not match the subquestion. The subquestions locate, crop, and ask color of the lamp and shirt, but the program counts the number of lamp.
""",
"""
Question: Is the lamp different in color than the shirt?
SubQuesion:
Step1, Locate the lamp in the given image, and obtain bounding boxes of lamp.
Step2, Crop the region of the lamp from the given image, based on bounding boxes of lamp. The bounding boxes are obtained in Step1.
Step3, Asking the image region of lamp, 'What color is the lamp?'. The image region of lamp is cropped in Step2.
Step4, Locate the shirt in the given image, and obtain bounding boxes of shirt. 
Step5, Crop the region of the shirt from the given image, based on bounding boxes of shirt. The bounding boxes are obtained in Step4.
Step6, Asking the image region of shirt, 'What color is the shirt?'. The image region of lamp is cropped in Step5.
Step7, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the color of lamp and the color of shirt. The color of lamp and shirt is obtained in Step3 and Step6, respectively. If their color are the same, the answer is 'yes'; On the contrary, the answer is 'no'.
Step8, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='lamp')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='shirt')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
Reason: The subquestions are correct, and can address the given question. But the Step3-Step6 of the program does not match the subquestion. The subquestions locate, crop, and ask color of the lamp and shirt, but the program counts the number of lamp.
""",
]


CURATED_SUBQUESTION=[
"""Question: Is the vehicle in the top of the image?
SubQuesion:
Step1, Locate the upper region of the given image since the question asks the top of the image, and obtain bounding boxes of the upper region.
Step2, Crop the upper region from the given image, based on bounding boxes of the upper region. The bounding boxes are obtained in Step1.
Step3, Locate vehicle in the upper region of the given image, and obtain bounding boxes of vehicle. The upper region is cropped in Step2.
Step4, Count the number of vehicle, based on bounding boxes of vehicle. The bounding boxes are obtained in Step3.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number of vehicles. The number is obtained in Step4. If the number is greater than zero, the answer is 'yes'; On the contrary, the answer is 'no'.
Step6, Visualize results.
""",
"""Question: What type of candy is in the bowl that the pizza cutter is to the right of?
Subquestion: 
Step1, Locate the pizza cutter, and obtain bounding boxes of the pizza cutter.
Step2, Crop the left part of the pizza cutter since the question is asking what is to the right of the pizza cutter. The bounding boxes are obtained in Step1.
Step3, Since the question is asking what type, then ask the image region, 'What candy is it?' image. The image is cropped in Step2.
Step4, Visualize results.
""",
"""Question: Are the glass bowls to the left of a book?
Subquestion: 
Step1, Locate the book, and obtain bounding boxes of the book.
Step2, Crop the left part of the book since the question is asking what is to the left of the book. The bounding boxes are obtained in Step1.
Step3, Try locate glass bowls in the cropped image. The image is cropped in Step2.
Step4, Count the number of bounding boxes. The bounding box is from Step3.
Step5, This is a yes or no question, so determine whether the answer is 'yes' or 'no' by executing Python expression.
Step6, Visualize results.
""",
"""Question: Does the cup that is to the right of the skateboarder look red?
Subquestion: 
Step1, Locate skateboarder in the given image, and obtain bounding boxes of skateboarder.
Step2, Crop the region right to the skateboarder, based on bounding boxes of skateboarder. The bounding boxes are obtained in Step1.
Step3, Asking the image region, 'What color is the cup?'. The image region of cup is obtained in Step2.
Step4, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the intermediate answers obtained in Step3. If the color is equal to 'red', the answer is 'yes'; On the contrary, the answer is 'no'.
Step5, Visualize results.
""",
"""Question: What's on the table?
Subquestion: 
Step1, Locate the table, and obtain bounding boxes of the table.
Step2, Crop the table since the question is asking what is on the table. The bounding boxes are obtained in Step1.
Step3, Ask the image region "what's on the table?". The image is cropped in Step2.
Step4, Visualize results.
""",
"""Question: Is the street light standing behind a truck?
SubQuesion:
Step1, Locate truck in the given image, and obtain bounding boxes of truck.
Step2, Crop the image region behind the truck from the given image, based on bounding boxes of truck. The bounding boxes are obtained in Step1.
Step3, Locate street light in the region behind the truck, and obtain bounding boxes of street light. The region behind the truck is cropped in Step2.
Step4, Count the number of street light, based on bounding boxes of street light. The bounding boxes are obtained in Step3.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number of street light. The number is obtained in Step4. If the number is greater than zero, the answer is 'yes'; On the contrary, the answer is 'no'.
Step6, Visualize results.
""",
"""Question: Is the log in front or behind the human in the center?
Subquestion: 
Step1, Locate the human in the center of the given image, and obtain bounding boxes of the animal.
Step2, Crop the image region in front of the human, based on bounding boxes of the human. The bounding boxes are obtained in Step1.
Step3, Crop the image region in front of the human, based on bounding boxes of the human. The bounding boxes are obtained in Step1.
Step4, Locate log in the image region in front of the human, and obtain bounding boxes of log. The image region in front of the human is cropped in Step2.
Step5, Count the number of log, based on bounding boxes of log. The bounding boxes are obtained in Step4.
Step6, Locate log in the image region in behind of the human, and obtain bounding boxes of log. The image region in front of the human is cropped in Step3.
Step7, Count the number of log, based on bounding boxes of log. The bounding boxes are obtained in Step5.
Step8, the question ask front or behind, so that determine whether the answer is 'front' or 'behind' by executing Python expression, based on the number of logs. The number is obtained in Step5 and Step 7 and. Based on the number, answer 'front' or 'behind' or 'neigher'.
Step9, Visualize results.
""",
"""Question: Are both the clouds and the pants the same color?
Subquestion: 
Step1, Locate the pants, and obtain bounding boxes of the pants.
Step2, Locate the clouds, and obtain bounding boxes of the clouds.
Step3, Crop the pants since the question requires knowing color of the pants. The bounding boxes are obtained in Step1.
Step4, Crop the clouds since the question requires knowing color of the clouds. The bounding boxes are obtained in Step2.
Step5, Ask image 'what color is the pants'. The image is cropped in Step3.
Step6, Ask image 'what color is the clouds'. The image is from Step4.
Step7, Determine whether colors are the same by executing Python expression.
Step6, Visualize results.
""",
"""Question: Is the car on the right side?
Subquestion: 
Step1, Locate the right side of the given image, and obtain bounding boxes.
Step2, Crop the right region based on bounding boxes obtained in Step1.
Step3, Locate the car of the cropped image from Step2.
Step4, Count the number of boxes from Step3.
Step5, Determine if the car is on the right side by Python expression. Answer yes if the result from Step4 are greater than 0.
Step6, Visualize results.
""",
"""Question: What color is the appliance above the bananas?
Subquestion: 
Step1, Locate the bananas, and obtain bounding boxes of the bananas.
Step2, Crop the above part of the bananas since the question is asking what is above the bananas. The bounding boxes are obtained in Step1.
Step3, Ask the image region, 'What color is it?' image. The image is cropped in Step2.
Step4, Visualize results.
""",
"""Question: What do both the pancake and the coffee mug have in common?
Subquestion: 
Step1, This question ask what two objects have in common without any detail information, therefore asking image region directly by the question 'What do both the pancake and the coffee mug have in common?'.
Step2, Visualize results.
""",
"""Question: How thick are the clouds the birds are flying in?
Subquestion: 
Step1, This question asks how thick are the clouds without any detail information, therefore asking image region directly by the question 'How thick are the clouds the birds are flying in?'.
Step2, Visualize results.
""",
"""Question: What material is the bath tub?
Subquestion: 
Step1, Locate the bath tub, and obtain bounding boxes of the bath tub.
Step2, Crop the bath tub since the question is asking what is the bath tub. The bounding boxes are obtained in Step1.
Step3, Ask the image region "what material is it?". The image is cropped in Step2.
Step4, Visualize results.
""",
"""Question: Who is standing?
Subquestion: 
Step1, This question asks who is standing without specifying location, species, and any other information, therefore asking image region directly by the question 'Who is standing?'.
Step2, Visualize results.
"""

]


CURATED_PROGRAMS=[
"""Question: Is the vehicle in the top of the image?
SubQuesion:
Step1, Locate the upper region of the given image since the question asks the top of the image, and obtain bounding boxes of the upper region.
Step2, Crop the upper region from the given image, based on bounding boxes of the upper region. The bounding boxes are obtained in Step1.
Step3, Locate vehicle in the upper region of the given image, and obtain bounding boxes of vehicle. The upper region is cropped in Step2.
Step4, Count the number of vehicle, based on bounding boxes of vehicle. The bounding boxes are obtained in Step3.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number of vehicles. The number is obtained in Step4. If the number is greater than zero, the answer is 'yes'; On the contrary, the answer is 'no'.
Step6, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='TOP')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='vehicle')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: What type of candy is in the bowl that the pizza cutter is to the right of?
Subquestion: 
Step1, Locate the pizza cutter, and obtain bounding boxes of the pizza cutter.
Step2, Crop the left part of the pizza cutter since the question is asking what is to the right of the pizza cutter. The bounding boxes are obtained in Step1.
Step3, Since the question is asking what type, then ask the image region, 'What candy is it?' image. The image is cropped in Step2.
Step4, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='pizza cutter')
IMAGE0=CROP_LEFTOF(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0, question='What candy is it?')
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Are the glass bowls to the left of a book?
Subquestion: 
Step1, Locate the book, and obtain bounding boxes of the book.
Step2, Crop the left part of the book since the question is asking what is to the left of the book. The bounding boxes are obtained in Step1.
Step3, Try locate glass bowls in the cropped image. The image is cropped in Step2.
Step4, Count the number of bounding boxes. The bounding box is from Step3.
Step5, This is a yes or no question, so determine whether the answer is 'yes' or 'no' by executing Python expression.
Step6, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='book')
IMAGE0=CROP_LEFTOF(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0, object='glass bowls')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: Does the cup that is to the right of the skateboarder look red?
Subquestion: 
Step1, Locate skateboarder in the given image, and obtain bounding boxes of skateboarder.
Step2, Crop the region right to the skateboarder, based on bounding boxes of skateboarder. The bounding boxes are obtained in Step1.
Step3, Asking the image region, 'What color is the cup?'. The image region of cup is obtained in Step2.
Step4, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the intermediate answers obtained in Step3. If the color is equal to 'red', the answer is 'yes'; On the contrary, the answer is 'no'.
Step5, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='skateboarder')
IMAGE0=CROP_RIGHTOF(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='What color is the cup?')
ANSWER1=EVAL(expr="'yes' if {ANSWER0} == 'red' else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: What's on the table?
Subquestion: 
Step1, Locate the table, and obtain bounding boxes of the table.
Step2, Crop the table since the question is asking what is on the table. The bounding boxes are obtained in Step1.
Step3, Ask the image region "what's on the table?". The image is cropped in Step2.
Step4, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='table')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question="what's on the table?")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Is the street light standing behind a truck?
SubQuesion:
Step1, Locate truck in the given image, and obtain bounding boxes of truck.
Step2, Crop the image region behind the truck from the given image, based on bounding boxes of truck. The bounding boxes are obtained in Step1.
Step3, Locate street light in the region behind the truck, and obtain bounding boxes of street light. The region behind the truck is cropped in Step2.
Step4, Count the number of street light, based on bounding boxes of street light. The bounding boxes are obtained in Step3.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number of street light. The number is obtained in Step4. If the number is greater than zero, the answer is 'yes'; On the contrary, the answer is 'no'.
Step6, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='truck')
IMAGE0=CROP_BEHIND(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='street light')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: Is the log in front or behind the human in the center?
Subquestion: 
Step1, Locate the human in the center of the given image, and obtain bounding boxes of the animal.
Step2, Crop the image region in front of the human, based on bounding boxes of the human. The bounding boxes are obtained in Step1.
Step3, Crop the image region in front of the human, based on bounding boxes of the human. The bounding boxes are obtained in Step1.
Step4, Locate log in the image region in front of the human, and obtain bounding boxes of log. The image region in front of the human is cropped in Step2.
Step5, Count the number of log, based on bounding boxes of log. The bounding boxes are obtained in Step4.
Step6, Locate log in the image region in behind of the human, and obtain bounding boxes of log. The image region in front of the human is cropped in Step3.
Step7, Count the number of log, based on bounding boxes of log. The bounding boxes are obtained in Step5.
Step8, the question ask front or behind, so that determine whether the answer is 'front' or 'behind' by executing Python expression, based on the number of logs. The number is obtained in Step5 and Step 7 and. Based on the number, answer 'front' or 'behind' or 'neigher'.
Step9, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='human')
IMAGE0=CROP_FRONTOF(image=IMAGE,box=BOX0)
IMAGE1=CROP_BEHIND(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='log')
ANSWER0=COUNT(box=BOX1)
BOX2=LOC(image=IMAGE1,object='log')
ANSWER1=COUNT(box=BOX2)
ANSWER2=EVAL(expr="'front' if {ANSWER0} > 0 else 'behind' if {ANSWER1} > 0 else 'neither'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Are both the clouds and the pants the same color?
Subquestion: 
Step1, Locate the pants, and obtain bounding boxes of the pants.
Step2, Locate the clouds, and obtain bounding boxes of the clouds.
Step3, Crop the pants since the question requires knowing color of the pants. The bounding boxes are obtained in Step1.
Step4, Crop the clouds since the question requires knowing color of the clouds. The bounding boxes are obtained in Step2.
Step5, Ask image 'what color is the pants'. The image is cropped in Step3.
Step6, Ask image 'what color is the clouds'. The image is from Step4.
Step7, Determine whether colors are the same by executing Python expression.
Step6, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='pants')
BOX1=LOC(image=IMAGE,object='clouds')
IMAGE0=CROP(image=IMAGE,box=BOX0)
IMAGE1=CROP(image=IMAGE,box=BOX1)
ANSWER0=VQA(image=IMAGE0,question='what color is the pants?')
ANSWER1=VQA(image=IMAGE1,question='what color is the clouds?')
ANSWER2=EVAL(expr="'yes' if {ANSWER0} == {ANSWER1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Is the car on the right side?
Subquestion: 
Step1, Locate the right side of the given image, and obtain bounding boxes.
Step2, Crop the right region based on bounding boxes obtained in Step1.
Step3, Locate the car of the cropped image from Step2.
Step4, Count the number of boxes from Step3.
Step5, Determine if the car is on the right side by Python expression. Answer yes if the result from Step4 are greater than 0.
Step6, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='RIGHT')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='car')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: What color is the appliance above the bananas?
Subquestion: 
Step1, Locate the bananas, and obtain bounding boxes of the bananas.
Step2, Crop the above part of the bananas since the question is asking what is above the bananas. The bounding boxes are obtained in Step1.
Step3, Ask the image region, 'What color is it?' image. The image is cropped in Step2.
Step4, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='bananas')
IMAGE0=CROP_ABOVE(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0, question='What color is it?')
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: What do both the pancake and the coffee mug have in common?
Subquestion: 
Step1, This question ask what two objects have in common without any detail information, therefore asking image region directly by the question 'What do both the pancake and the coffee mug have in common?'.
Step2, Visualize results.
Program:
ANSWER0=VQA(image=IMAGE,question='What do both the pancake and the coffee mug have in common?')
FINAL_RESULT=RESULT(var=ANSWER0)
""","""Question: How thick are the clouds the birds are flying in?
Subquestion: 
Step1, This question asks how thick are the clouds without any detail information, therefore asking image region directly by the question 'How thick are the clouds the birds are flying in?'.
Step2, Visualize results.
Program:
ANSWER1=VQA(image=IMAGE,question='How thick are the clouds the birds are flying in?')
FINAL_RESULT=RESULT(var=ANSWER1)
""","""Question: What material is the bath tub?
Subquestion: 
Step1, Locate the bath tub, and obtain bounding boxes of the bath tub.
Step2, Crop the bath tub since the question is asking what is the bath tub. The bounding boxes are obtained in Step1.
Step3, Ask the image region "what material is it?". The image is cropped in Step2.
Step4, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='bath tub')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question="what material is it?")
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Who is standing?
Subquestion: 
Step1, This question asks who is standing without specifying location, species, and any other information, therefore asking image region directly by the question 'Who is standing?'.
Step2, Visualize results.
Program:
ANSWER1=VQA(image=IMAGE,question='Who is standing?')
FINAL_RESULT=RESULT(var=ANSWER1)
"""

]


REFLECTION_STEP=[
"""Question: What's the window made of?
Description of the Input Image: a photography of a cat sitting on a table watching tv
Human Feedback: the correct answer should be glass
Our Wrong Answer: metal
subQuesion:
Step1, Locate window in the given image, and obtain bounding boxes of window.
Step2, Crop the region of window from the given image, based on bounding boxes of window. The bounding boxes are obtained in Step1.
Step3, Asking the image region of window, 'What's the window made of?'. The image region of window is obtained in Step2.
Step4, Visualize results.
Program and obtained result in each step:
Step1
Program: BOX0=LOC(image=IMAGE,object='window')
Result of BOX0 is empty 
Step2
Program: IMAGE0=CROP(image=IMAGE,box=BOX0)
Result of The description of IMAGE0: a photography of a cat sitting on a table watching tv
Step3
Program: ANSWER0=VQA(image=IMAGE0,question='What's the window made of?')
Result of ANSWER0: cat
Step4
Program: FINAL_RESULT=RESULT(var=ANSWER0)
Result of FINAL_RESULT: metal
Error Location: functions called by programs
Reason: In the Step1 of the program, the used function 'LOC' failed to locate the window in the given image, as the obtained result of BOX0 is empty.
""",
"""Question: Is the lamp different in color than the shirt?
Description of the Input Image: a photography of a couple of people on a snowboard in the snow
Human Feedback: the correct answer should be yes
Our Wrong Answer: no
subquestion: 
Step1, Locate the lamp in the given image, and obtain bounding boxes of lamp.
Step2, Crop the region of the lamp from the given image, based on bounding boxes of lamp. The bounding boxes are obtained in Step1.
Step3, Asking the image region of lamp, 'What color is the lamp?'. The image region of lamp is cropped in Step2.
Step4, Locate the shirt in the given image, and obtain bounding boxes of shirt. 
Step5, Crop the region of the shirt from the given image, based on bounding boxes of shirt. The bounding boxes are obtained in Step4.
Step6, Asking the image region of shirt, 'What color is the shirt?'. The image region of lamp is cropped in Step5.
Step7, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the color of lamp and the color of shirt. The color of lamp and shirt is obtained in Step3 and Step6, respectively. If their color are the same, the answer is 'yes'; On the contrary, the answer is 'no'.
Step8, Visualize results.
Program and obtained result in each step:
Step1
Program: BOX0=LOC(image=IMAGE,object='lamp')
Result of The coordinate of BOX0: [[45, 78, 245, 345]]
Step2
Program: IMAGE0=CROP(image=IMAGE,box=BOX0)
Result of The description of IMAGE0: a photography of a couple of people on a snowboard in the snow
Step3
Program: BOX1=LOC(image=IMAGE0,object='shirt')
Result of BOX1 is empty 
Step4
Program: ANSWER0=COUNT(box=BOX1)
Result of ANSWER0: 0
Step5
Program: ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
Result of ANSWER1: no
Step6
Program: FINAL_RESULT=RESULT(var=ANSWER1)
Result of FINAL_RESULT: no
Error Location: program
Reason: The subquestions are correct, and can address the given question. But the Step3-Step6 of the program does not match the subquestion. The subquestions locate, crop, and ask color of the lamp and shirt, but the program counts the number of lamp.
""",
"""Question: Is the pipe made of the same material as the bed sheet?
Description of the Input Image: a photography of a kitchen with a refrigerator and a stove
Human Feedback: the correct answer should be yes
Our Wrong Answer: metal
subquestion: 
Step1, Locate pipe in the given image, and obtain bounding boxes of pipe.
Step2, Crop the region of pipe from the given image, based on bounding boxes of pipe. The bounding boxes are obtained in Step1.
Step3, Asking the image region of pipe, 'What material is the pipe made of?'. The image region of pipe is obtained in Step2.
Step4, Visualize results.
Program and obtained result in each step:
Step1
Program: BOX0=LOC(image=IMAGE,object='pipe')
Result of The coordinate of BOX0: [[67, 28, 98, 69]]
Step2
Program: IMAGE0=CROP(image=IMAGE,box=BOX0)
Result of The description of IMAGE0: a photography of a man is holding a knife in his hand
Step3
Program: ANSWER0=VQA(image=IMAGE0,question='What material is the pipe made of?')
Result of ANSWER0: metal
Step4
Program: FINAL_RESULT=RESULT(var=ANSWER0)
Result of FINAL_RESULT: metal
Error Location: subquestions
Reason: In Step1-Step3, the subquestions only identify the material of the pipe. The subquestions should then to identify the material of the bed sheet, and then compare the material of the pipe and the bed sheet.
""",
"""Question: What type of animal is to the left of the people?
Description of the Input Image: a photography of a group of giraffes and zebras in a field
Human Feedback: the correct answer should be zebras
Our Wrong Answer: dog
subquestion: 
Step1, Locate people in the given image, and obtain bounding boxes of people.
Step2, Crop the region of people from the given image, based on bounding boxes of people. The bounding boxes are obtained in Step1.
Step3, Asking the image region of people, 'What type of animal is to the left of the people?'. The image region of people is obtained in Step2.
Step4, Visualize results.
Program and obtained result in each step:
Step1
Program: BOX0=LOC(image=IMAGE,object='people')
Result of The coordinate of BOX0: [[545, 241, 562, 266], [592, 245, 606, 279]]
Step2
Program: IMAGE0=CROP(image=IMAGE,box=BOX0)
Result of The description of IMAGE0: a photography of a man standing next to a white refrigerator
Step3
Program: ANSWER0=VQA(image=IMAGE0,question='What type of animal is to the left of the people?')
Result of ANSWER0: dog
Step4
Program: FINAL_RESULT=RESULT(var=ANSWER0)
Result of FINAL_RESULT: dog
Error Location: subquestions
Reason: In Step2 of the subquestions, the subquestions should crop the image region on the left side of people, instead of cropping the region of people, since the question asks 'What type of animal is to the left of the people?'.
""",
"""Question: Is the water dark and wet?
Description of the Input Image: a photography of a woman walking down a street holding an umbrella
Human Feedback: the correct answer should be yes
Our Wrong Answer: no
subquestion: 
Step1, Locate water in the given image, and obtain bounding boxes of water.
Step2, Crop the region of water from the given image, based on bounding boxes of water. The bounding boxes are obtained in Step1.
Step3, Asking the image region of water, 'Is the water dark?'. The image region of water is obtained in Step2.
Step4, Asking the image region of water, 'Is the water wet?'. The image region of water is obtained in Step2.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the intermediate answers obtained in Step3 and Step4. If the answer of Step3 is 'yes' and the answer of Step4 is 'yes', the answer is 'yes'; On the contrary, the answer is 'no'.
Step6, Visualize results.
Program and obtained result in each step:
Step1
Program: BOX0=LOC(image=IMAGE,object='water')
Result of The coordinate of BOX0: [[136, 45, 606, 279]]
Step2
Program: IMAGE0=CROP(image=IMAGE,box=BOX0)
Result of The description of IMAGE0: a photography of a street
Step3
Program: ANSWER0=VQA(image=IMAGE0,question='Is the water dark?')
Result of ANSWER0: no
Step4
Program: ANSWER1=VQA(image=IMAGE0,question='Is the water wet?')
Result of ANSWER1: yes
Step5
Program: ANSWER2=EVAL(expr="'yes' if {ANSWER0} and {ANSWER1} else 'no'")
Result of ANSWER2: no
Step6
Program: FINAL_RESULT=RESULT(var=ANSWER2)
Result of FINAL_RESULT: no
Error Location: functions called by programs
Reason: In the Step3 of the program, the used function 'VQA' failed failed to identify whether the water is dark correctly, as the obtained result of ANSWER0 is 'no' instead of 'yes'.
""",
"""Question: Are there either any windows or trains in this image?
Description of the Input Image: a photography of a woman and a man playing a video game
Human Feedback: the correct answer should be yes
Our Wrong Answer: no
subquestion: 
Step1, Locate any windows in the given image, and obtain bounding boxes of any windows.
Step2, Crop the region of any windows from the given image, based on bounding boxes of any windows. The bounding boxes are obtained in Step1.
Step3, Asking the image region of any windows, 'Are there either any windows or trains in this image?'. The image region of any windows is obtained in Step2.
Step4, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the intermediate answers obtained in Step3. If the answer is 'yes', the answer is 'yes'; On the contrary, the answer is 'no'.
Step5, Visualize results.
Program and obtained result in each step:
Step1
Program: BOX0=LOC(image=IMAGE,object='window')
Result of The coordinate of BOX0: [[274, 93, 408, 327], [25, 59, 192, 331]]
Step2
Program: IMAGE0=CROP(image=IMAGE,box=BOX0)
Result of The description of IMAGE0: a photography of a man in a plaid shirt is holding a wii controller
Step3
Program: ANSWER0=VQA(image=IMAGE0,question='Are there either any windows or trains in this image?')
Result of ANSWER0: yes
Step4
Program: ANSWER1=EVAL(expr="'yes' if {ANSWER0} == 'yes' else 'no'")
Result of ANSWER1: no
Step5
Program: FINAL_RESULT=RESULT(var=ANSWER1)
Result of FINAL_RESULT: no
Error Location: subquestions
Reason: In Step2 of the subquestions, the subquestions should count the number of boxes of windows', instead of cropping the image and asking question. Then, the subquestions should further detect trains and count the number of trains.
""",
# """Question: Do you think the table is rectangular?
# The description of Input image: a photography of a restaurant with a table set for a meal
# Human Feedback: the correct answer should be yes
# Our Wrong Answer: no
# Following are the decomposed subquestion, used program, and obtained result in each step. 
# subquestion: 
# Step1, Locate table in the given image, and obtain bounding boxes of table.
# Step2, Crop the region of table from the given image, based on bounding boxes of table. The bounding boxes are obtained in Step1.
# Step3, Asking the image region of table, 'What shape is the table?'. The image region of table is obtained in Step2.
# Step4, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the intermediate answers obtained in Step3. If the shape is equal to 'rectangular', the answer is 'yes'; On the contrary, the answer is 'no'.
# Step5, Visualize results.
# Program and obtained result in each step:
# Step1
# Program: BOX0=LOC(image=IMAGE,object='table')
# The coordinate of BOX0: [[40, 240, 581, 479], [180, 200, 258, 243]]
# Step2
# Program: IMAGE0=CROP(image=IMAGE,box=BOX0)
# The description of IMAGE0: a photography of a table set with a white table cloth and red plates
# Step3
# Program: ANSWER0=VQA(image=IMAGE0,question='What shape is the table?')
# Results of ANSWER0: circular
# Step4
# Program: ANSWER1=EVAL(expr="'yes' if {ANSWER0} == 'rectangular' else 'no'")
# Result of ANSWER1: no
# Step5
# Program: INAL_RESULT=RESULT(var=ANSWER1)
# Result of FINAL_RESULT: no
# Error Location: functions called by programs
# Reason: In the Step3 of the program, the used function 'VQA' failed to recognize the shape of the table correctly, as the obtained result of ANSWER0 is 'rectangle' instead of 'circular'.
# """,
]


REFLECTION_INTERRUPT=[
"""Question: Is there a plate to the left of the food tray in the top?
SubQuesion:
Step1, Locate food tray in the given image, and obtain bounding boxes of food tray.
Step2, Crop the image region on the left side of food tray from the given image, based on bounding boxes of food tray. The bounding boxes are obtained in Step1.
Step3, Locate plate in the image region on the left side of the food tray, and obtain bounding boxes of plate. The image region on the left side of the food tray is cropped in Step2.
Step4, Count the number of plate, based on bounding boxes of plate. The bounding boxes are obtained in Step3.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number of plate. The number is obtained in Step4. If the number is greater than zero, the answer is 'yes'; On the contrary, the answer is 'no'.
Step6, Visualize results.
Program: 
BOX0=LOC(image=IMAGE,object='food tray')
IMAGE0=CROP_LEFT(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='plate')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
Error Location: 
Second line of the program: IMAGE0=CROP_LEFT(image=IMAGE,box=BOX0)
Reason: 
The bug in the program is in the second line: IMAGE0=CROP_LEFT(image=IMAGE,box=BOX0), where the function 'CROP_LEFT' is called. It should be 'CROP_LEFTOF'.
""",
"""Question: Is the vehicle in the top of the image?
SubQuesion:
Step1, Locate the upper region of the given image, and obtain bounding boxes of the upper region.
Step2, Crop the upper region from the given image, based on bounding boxes of the upper region. The bounding boxes are obtained in Step1.
Step3, Locate vehicle in the upper region of the given image, and obtain bounding boxes of vehicle. The upper region is cropped in Step2.
Step4, Count the number of vehicle, based on bounding boxes of vehicle. The bounding boxes are obtained in Step3.
Step5, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the number of vehicles. The number is obtained in Step4. If the number is greater than zero, the answer is 'yes'; On the contrary, the answer is 'no'.
Step6, Visualize results.
Program:
BOX0=LOC(image=IMAGE,object='TOP')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE1,object='vehicle')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
Error Location: 
Third line of the program: BOX1=LOC(image=IMAGE1,object='vehicle')
Reason:
The bug in the program is in the third line: BOX1=LOC(image=IMAGE1,object='vehicle'), where the variable 'IMAGE1' is called. It should be 'IMAGE'.
""",
]


INFERENCE=[
"""Question: Do you think the table is rectangular?
The description of Input image: a photography of a restaurant with a table set for a meal
Human Feedback: the correct answer should be yes
Our Wrong Answer: no
Following are the decomposed subquestion, used program, and obtained result in each step. 
subquestion: 
Step1, Locate table in the given image, and obtain bounding boxes of table.
Step2, Crop the region of table from the given image, based on bounding boxes of table. The bounding boxes are obtained in Step1.
Step3, Asking the image region of table, 'What shape is the table?'. The image region of table is obtained in Step2.
Step4, Determine whether the answer is 'yes' or 'no' by executing Python expression, based on the intermediate answers obtained in Step3. If the shape is equal to 'rectangular', the answer is 'yes'; On the contrary, the answer is 'no'.
Step5, Visualize results.
Program and obtained result in each step:
Step1
Program: BOX0=LOC(image=IMAGE,object='table')
The coordinate of BOX0: [[40, 240, 581, 479], [180, 200, 258, 243]]
Step2
Program: IMAGE0=CROP(image=IMAGE,box=BOX0)
The description of IMAGE0: a photography of a table set with a white table cloth and red plates
Step3
Program: ANSWER0=VQA(image=IMAGE0,question='What shape is the table?')
Results of ANSWER0: circular
Step4
Program: ANSWER1=EVAL(expr="'yes' if {ANSWER0} == 'rectangular' else 'no'")
Result of ANSWER1: no
Step5
Program: INAL_RESULT=RESULT(var=ANSWER1)
Result of FINAL_RESULT: no
Error Location: functions called by programs
Reason: In the Step3 of the program, the used function 'VQA' failed to recognize the shape of the table correctly, as the obtained result of ANSWER0 is 'rectangle' instead of 'circular'.
Correct answer of the wrong step: rectangle.
""",
]
