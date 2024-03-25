FAILED_SUBQUESTION=[""" """]


FAILED_PROGRAM=[""" """]


CURATED_SUBQUESTION=[
"""
Question: Hide the face of Nicole Kidman with face_with_tongue.
SubQuetion:
Step1, Detect face regions from the given image, and obtain bounding boxes of face regions.
Step2, Select the face region of Nicole Kidman, based on the bounding boxes obtained in Step1.
Step3, Add the emoji face_with_tongue to the face region of Nicole Kidman in the given image, where the face region of Nicole Kidman is obtained in Step2.
Step4, Visualize results.
""",
"""
Question: Hide the faces of Nicole Kidman and Brad Pitt with winking_face and smiling_face_with_sunglasses.
SubQuetion:
Step1, Detect face regions from the given image, and obtain bounding boxes of face regions.
Step2, Select the face region of Nicole Kidman, based on the bounding boxes obtained in Step1.
Step3, Add the emoji winking_face to the face region of Nicole Kidman in the given image, where the face region of Nicole Kidman is obtained in Step2.
Step4, Select the face region of Brad Pitt, based on the bounding boxes obtained in Step1.
Step5, Add the emoji smiling_face_with_sunglasses to the face region of Brad Pitt in the image obtained at Step3, and the face region of Brad Pitt is obtained in Step4.
Step6, Visualize results.
""",
"""
Question: Create a color pop of Amy and Daphne.
SubQuetion:
Step1, Perform segmentation for the given image, and obtain image regions of each object.
Step2, Select the objects of Amy and Daphne, based on the image regions obtained in Step1.
Step3, Perform the COLORPOP operation for image regions of Amy and Daphne in the given image, where the image regions of Amy and Daphne are obtained in Step2.
Step4, Visualize results.
""",
"""
Question: Create a color pop of the girl and the umbrella.
SubQuetion:
Step1, Perform segmentation for the given image, and obtain image regions of each object.
Step2, Select the objects of girl and umbrella, based on the image regions obtained in Step1.
Step3, Perform the COLORPOP operation for image regions of girl and umbrella in the given image, where the image regions of girl and umbrella are obtained in Step2.
Step4, Visualize results.
""",
"""
Question: Create a color pop of the dog, frisbee, and grass.
SubQuetion:
Step1, Perform segmentation for the given image, and obtain image regions of each object.
Step2, Select the objects of dog, frisbee, and grass, based on the image regions obtained in Step1.
Step3, Perform the COLORPOP operation for image regions of dog, frisbee, and grass in the given image, where the image regions of dog, frisbee, and grass are obtained in Step2.
Step4, Visualize results.
""",
"""
Question: Create a color pop of the man wearing a red suit (person).
SubQuetion:
Step1, Perform segmentation for the given image, and obtain image regions of each object.
Step2, Select the object of the man wearing a red suit (person), based on the image regions obtained in Step1.
Step3, Perform the COLORPOP operation for image region of the man wearing a red suit (person) in the given image, where the image region of the man wearing a red suit (person) is obtained in Step2.
Step4, Visualize results.
""",
"""
Question: Select the red bus and blur the background.
SubQuetion:
Step1, Perform segmentation for the given image, and obtain image regions of each object.
Step2, Select the object of the red bus, based on the image regions obtained in Step1.
Step3, Blur the background for given image, except for the image region of the red bus, where the image region of the red bus is obtained in Step2.
Step4, Visualize results.
""",
"""
Question: Replace the red bus with a blue bus.
SubQuetion:
Step1, Perform segmentation for the given image, and obtain image regions of each object.
Step2, Select the object of the red bus, based on the image regions obtained in Step1.
Step3, Generate a blue bus to replace the red bus in the given image, where the image region of the red bus is obtained in Step2.
Step4, Visualize results.
""",
"""
Question: Replace the red bus with blue bus and the road with dirt road.
SubQuetion:
Step1, Perform segmentation for the given image, and obtain image regions of each object.
Step2, Select the object of the red bus, based on the image regions obtained in Step1.
Step3, Generate a blue bus to replace the red bus in the given image, where the image region of the red bus is obtained in Step2.
Step4, Perform segmentation for the image obtained in Step3, and obtain image regions of each object.
Step5, Select the object of the road, based on the image regions obtained in Step4.
Step6, Generate a dirt road to replace the road in the image obtained in Step3, where the image region of the road is obtained in Step5.
Step7, Visualize results.
""",
"""
Question: Replace the red bus with a truck.
SubQuetion:
Step1, Perform segmentation for the given image, and obtain image regions of each object.
Step2, Select the object of the red bus, based on the image regions obtained in Step1.
Step3, Generate a truck to replace the red bus in the given image, where the image region of the red bus is obtained in Step2.
Step4, Visualize results.
""",
"""
Question: Replace Barack Obama with Joe Biden.
SubQuetion:
Step1, Detect face regions from the given image, and obtain bounding boxes of face regions.
Step2, Select the face region of Barack Obama, based on the bounding boxes obtained in Step1.
Step3, Generate Joe Biden to replace Barack Obama in the given image, where the image region of Barack Obama is obtained in Step2.
Step4, Visualize results.
""",
"""
Question: Replace Donald Trump with a panda.
SubQuetion:
Step1, Detect face regions from the given image, and obtain bounding boxes of face regions.
Step2, Select the face region of Donald Trump, based on the bounding boxes obtained in Step1.
Step3, Generate a panda to replace Donald Trump in the given image, where the image region of Donald Trump is obtained in Step2.
Step4, Visualize results.
""",]


CURATED_PROGRAMS=[
"""
Question: Hide the face of Nicole Kidman with face_with_tongue.
SubQuetion:
Step1, Detect face regions from the given image, and obtain bounding boxes of face regions.
Step2, Select the face region of Nicole Kidman, based on the bounding boxes obtained in Step1.
Step3, Add the emoji face_with_tongue to the face region of Nicole Kidman in the given image, where the face region of Nicole Kidman is obtained in Step2.
Step4, Visualize results.
Program:
OBJ0=FACEDET(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='Nicole Kidman',category=None)
IMAGE0=EMOJI(image=IMAGE,object=OBJ1,emoji='face_with_tongue')
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Hide the faces of Nicole Kidman and Brad Pitt with winking_face and smiling_face_with_sunglasses.
SubQuetion:
Step1, Detect face regions from the given image, and obtain bounding boxes of face regions.
Step2, Select the face region of Nicole Kidman, based on the bounding boxes obtained in Step1.
Step3, Add the emoji winking_face to the face region of Nicole Kidman in the given image, where the face region of Nicole Kidman is obtained in Step2.
Step4, Select the face region of Brad Pitt, based on the bounding boxes obtained in Step1.
Step5, Add the emoji smiling_face_with_sunglasses to the face region of Brad Pitt in the image obtained at Step3, and the face region of Brad Pitt is obtained in Step4.
Step6, Visualize results.
Program:
OBJ0=FACEDET(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='Nicole Kidman',category=None)
IMAGE0=EMOJI(image=IMAGE,object=OBJ1,emoji='winking_face')
OBJ2=SELECT(image=IMAGE,object=OBJ0,query='Brad Pitt',category=None)
IMAGE1=EMOJI(image=IMAGE0,object=OBJ1,emoji='smiling_face_with_sunglasses')
FINAL_RESULT=RESULT(var=IMAGE1)
""",
"""
Question: Create a color pop of Amy and Daphne.
SubQuetion:
Step1, Perform segmentation for the given image, and obtain image regions of each object.
Step2, Select the objects of Amy and Daphne, based on the image regions obtained in Step1.
Step3, Perform the COLORPOP operation for image regions of Amy and Daphne in the given image, where the image regions of Amy and Daphne are obtained in Step2.
Step4, Visualize results.
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='Amy,Daphne',category=None)
IMAGE0=COLORPOP(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Create a color pop of the girl and the umbrella.
SubQuetion:
Step1, Perform segmentation for the given image, and obtain image regions of each object.
Step2, Select the objects of girl and umbrella, based on the image regions obtained in Step1.
Step3, Perform the COLORPOP operation for image regions of girl and umbrella in the given image, where the image regions of girl and umbrella are obtained in Step2.
Step4, Visualize results.
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='girl,umbrella',category=None)
IMAGE0=COLORPOP(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Create a color pop of the dog, frisbee, and grass.
SubQuetion:
Step1, Perform segmentation for the given image, and obtain image regions of each object.
Step2, Select the objects of dog, frisbee, and grass, based on the image regions obtained in Step1.
Step3, Perform the COLORPOP operation for image regions of dog, frisbee, and grass in the given image, where the image regions of dog, frisbee, and grass are obtained in Step2.
Step4, Visualize results.
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='dog,frisbee,grass',category=None)
IMAGE0=COLORPOP(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Create a color pop of the man wearing a red suit (person).
SubQuetion:
Step1, Perform segmentation for the given image, and obtain image regions of each object.
Step2, Select the object of the man wearing a red suit (person), based on the image regions obtained in Step1.
Step3, Perform the COLORPOP operation for image region of the man wearing a red suit (person) in the given image, where the image region of the man wearing a red suit (person) is obtained in Step2.
Step4, Visualize results.
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='man wearing a red suit',category='person')
IMAGE0=COLORPOP(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Select the red bus and blur the background.
SubQuetion:
Step1, Perform segmentation for the given image, and obtain image regions of each object.
Step2, Select the object of the red bus, based on the image regions obtained in Step1.
Step3, Blur the background for given image, except for the image region of the red bus, where the image region of the red bus is obtained in Step2.
Step4, Visualize results.
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='red bus',category=None)
IMAGE0=BGBLUR(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Replace the red bus with a blue bus.
SubQuetion:
Step1, Perform segmentation for the given image, and obtain image regions of each object.
Step2, Select the object of the red bus, based on the image regions obtained in Step1.
Step3, Generate a blue bus to replace the red bus in the given image, where the image region of the red bus is obtained in Step2.
Step4, Visualize results.
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='red bus',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='blue bus')
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Replace the red bus with blue bus and the road with dirt road.
SubQuetion:
Step1, Perform segmentation for the given image, and obtain image regions of each object.
Step2, Select the object of the red bus, based on the image regions obtained in Step1.
Step3, Generate a blue bus to replace the red bus in the given image, where the image region of the red bus is obtained in Step2.
Step4, Perform segmentation for the image obtained in Step3, and obtain image regions of each object.
Step5, Select the object of the road, based on the image regions obtained in Step4.
Step6, Generate a dirt road to replace the road in the image obtained in Step3, where the image region of the road is obtained in Step5.
Step7, Visualize results.
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='red bus',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='blue bus')
OBJ2=SEG(image=IMAGE0)
OBJ3=SELECT(image=IMAGE0,object=OBJ2,query='road',category=None)
IMAGE1=REPLACE(image=IMAGE0,object=OBJ3,prompt='dirt road')
FINAL_RESULT=RESULT(var=IMAGE1)
""",
"""
Question: Replace the red bus with a truck.
SubQuetion:
Step1, Perform segmentation for the given image, and obtain image regions of each object.
Step2, Select the object of the red bus, based on the image regions obtained in Step1.
Step3, Generate a truck to replace the red bus in the given image, where the image region of the red bus is obtained in Step2.
Step4, Visualize results.
Program:
OBJ0=SEG(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='red bus',category='bus')
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='truck')
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Replace Barack Obama with Joe Biden.
SubQuetion:
Step1, Detect face regions from the given image, and obtain bounding boxes of face regions.
Step2, Select the face region of Barack Obama, based on the bounding boxes obtained in Step1.
Step3, Generate Joe Biden to replace Barack Obama in the given image, where the image region of Barack Obama is obtained in Step2.
Step4, Visualize results.
Program:
OBJ0=FACEDET(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='Barack Obama',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='Joe Biden')
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Replace Donald Trump with a panda.
SubQuetion:
Step1, Detect face regions from the given image, and obtain bounding boxes of face regions.
Step2, Select the face region of Donald Trump, based on the bounding boxes obtained in Step1.
Step3, Generate a panda to replace Donald Trump in the given image, where the image region of Donald Trump is obtained in Step2.
Step4, Visualize results.
Program:
OBJ0=FACEDET(image=IMAGE)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='Donald Trump',category=None)
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='panda')
FINAL_RESULT=RESULT(var=IMAGE0)
""",]


REFLECTION_STEP=[
"""
Question: Hide the face of Nicole Kidman with face_with_tongue.
The description of input image: a photography of a several person on the seat
Human Feedback: face_with_tongue hides the wrong face, it is on the face of another person, instead of Nicole Kidman.
SubQuetion:
Step1, Detect face regions from the given image, and obtain bounding boxes of face regions.
Step2, Select the face region of Nicole Kidman, based on the bounding boxes obtained in Step1.
Step3, Add the emoji face_with_tongue to the face region of Nicole Kidman in the given image, where the face region of Nicole Kidman is obtained in Step2.
Step4, Visualize results.
Program and obtained result in each step:
Step1
Program: OBJ0=FACEDET(image=IMAGE0)
Result of The coordinate of OBJ0: [[545, 241, 562, 266], [592, 245, 606, 279]]
Step2
Program: OBJ1=SELECT(image=IMAGE,object=OBJ0,query='Nicole Kidman',category=None)
Result of The coordinate of OBJ1: [[592, 245, 606, 279]]
Step3
Program: IMAGE0=EMOJI(image=IMAGE,object=OBJ1,emoji='face_with_tongue')
Description of IMAGE0: a photography of a several person on the seat, and one emoji is on one face
Step4
Program: FINAL_RESULT=RESULT(var=IMAGE0)
Description of FINAL_RESULT: a photography of a several person on the seat, and one emoji is on one face
Error Location: functions called by programs
Reason: In the Step2 of the program, the used function 'SELECT' failed to select the face of Nicole Kidman correctly.
""",
"""
Question: Create a color pop of Amy Daphne.
The description of input image: a photography of a several person on the seat
Feedback: make the color pop on a wrong object, rather than Amy Daphne.
SubQuetion:
Step1, Perform segmentation for the given image, and obtain image regions of each object.
Step2, Select the objects of Amy Daphne, based on the image regions obtained in Step1.
Step3, Perform the COLORPOP operation for image regions of Amy Daphne in the given image, where the image regions of Amy Daphne are obtained in Step2.
Step4, Visualize results.
Program and obtained result in each step:
Step1
Program: OBJ0=SEG(image=IMAGE)
Result of The coordinate of OBJ0: [[23, 56, 79, 56], [592, 245, 606, 765]]
Step2
Program: OBJ1=SELECT(image=IMAGE,object=OBJ0,query='Amy Daphne',category=None)
Result of The coordinate of OBJ1: [[592, 245, 606, 765]]
Step3
Program: IMAGE0=COLORPOP(image=IMAGE,object=OBJ1)
Description of IMAGE0: One man in the image is in color, and the rest region is in gray.
Step4
Program: FINAL_RESULT=RESULT(var=IMAGE0)
Description of FINAL_RESULT: One man in the image is in color, and the rest region is in gray.
Error Location: functions called by programs
Reason: In the Step2 of the program, the used function 'SELECT' failed to select the object of Amy Daphne.
""",
"""
Question: Replace the red bus with a truck.
The description of input image: a photography of a red bus
Human Feedback: the bus is not completely replaced with a truck.
SubQuetion:
Step1, Perform segmentation for the given image, and obtain image regions of each object.
Step2, Select the object of the red bus, based on the image regions obtained in Step1.
Step3, Generate a truck to replace the red bus in the given image, where the image region of the red bus is obtained in Step2.
Step4, Visualize results.
Program and obtained result in each step:
Step1
OBJ0=SEG(image=IMAGE)
Result of The coordinate of OBJ0: [[350, 756, 352, 789], [592, 245, 606, 765]]
Step2
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='red bus',category='bus')
Result of The coordinate of OBJ1: [[350, 756, 352, 789]]
Step3
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='truck')
Description of IMAGE0: One vehicle composed of a bus and a truck.
Step4
FINAL_RESULT=RESULT(var=IMAGE0)
Description of FINAL_RESULT: One vehicle composed of a bus and a truck.
Error Location: functions called by programs
Reason: In the Step1 of the program, the used function 'SEG' failed to select the object of the red bus.
""",
"""
Question: Replace the red bus with a truck.
The description of input image: a photography of a red bus
Human Feedback: the bus is replaced with a wrong object, not a truck.
SubQuetion:
Step1, Perform segmentation for the given image, and obtain image regions of each object.
Step2, Select the object of the red bus, based on the image regions obtained in Step1.
Step3, Generate a truck to replace the red bus in the given image, where the image region of the red bus is obtained in Step2.
Step4, Visualize results.
Program and obtained result in each step:
Step1
OBJ0=SEG(image=IMAGE)
Result of The coordinate of OBJ0: [[350, 756, 352, 789], [592, 245, 606, 765]]
Step2
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='red bus',category='bus')
Result of The coordinate of OBJ1: [[350, 756, 352, 789]]
Step3
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='truck')
Description of IMAGE0: a photography of a train in the wild
Step4
FINAL_RESULT=RESULT(var=IMAGE0)
Description of FINAL_RESULT: a photography of a train in the wild.
Error Location: functions called by programs
Reason: In the Step3 of the program, the used function 'REPLACE' failed to generate a truck to replace the bus.
""",
"""
Question: Replace the red bus with a truck.
The description of input image: a photography of a red bus
Human Feedback: The image is completely new generated, and the background is changed
SubQuetion:
Step1, Perform segmentation for the given image, and obtain image regions of each object.
Step2, Select the object of the red bus, based on the image regions obtained in Step1.
Step3, Generate a truck to replace the red bus in the given image, where the image region of the red bus is obtained in Step2.
Step4, Visualize results.
Program and obtained result in each step:
Step1
OBJ0=SEG(image=IMAGE)
Result of The coordinate of OBJ0: [[0, 0, 800, 800]]
Step2
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='red bus',category='bus')
Result of The coordinate of OBJ1: [[0, 0, 800, 800]]
Step3
IMAGE0=REPLACE(image=IMAGE,object=OBJ1,prompt='truck')
Description of IMAGE0: One truck in the wild.
Step4
FINAL_RESULT=RESULT(var=IMAGE0)
Description of FINAL_RESULT: One truck in the wild.
Error Location: functions called by programs
Reason: In the Step1 of the program, the used function 'SEG' failed to segment red bus.
""",
]


REFLECTION_INTERRUPT=[
    """
Question: Hide the face of Nicole Kidman with face_with_tongue.
SubQuetion:
Step1, Detect face regions from the given image, and obtain bounding boxes of face regions.
Step2, Select the face region of Nicole Kidman, based on the bounding boxes obtained in Step1.
Step3, Add the emoji face_with_tongue to the face region of Nicole Kidman in the given image, where the face region of Nicole Kidman is obtained in Step2.
Step4, Visualize results.
Program:
OBJ0=FACEDET(image=IMAGE0)
OBJ1=SELECT(image=IMAGE,object=OBJ0,query='Nicole Kidman',category=None)
IMAGE0=EMOJI(image=IMAGE,object=OBJ1,emoji='face_with_tongue')
FINAL_RESULT=RESULT(var=IMAGE0)
Reason: 
The bug in the program is in the first line: OBJ0=FACEDET(image=IMAGE0), where 'IMAGE0' is not defined. It should be 'IMAGE'.
""",]


INFERENCE=[""" """]