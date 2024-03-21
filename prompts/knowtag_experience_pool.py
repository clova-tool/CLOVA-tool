FAILED_SUBQUESTION=[
"""
Question: Tag the flags of two smallest countries in Europe by area.
Step1, Locate flags from the given image, and obtain bounding boxes of flags.
Step2, List 'smallest countries in Europe by area' by asking GPT.
Step3: Classify flags by their countries, based on the bounding boxes of flags obtained in Step1 and the country list obtain in Step2.
Step4: Sort the classified flags by area in descending order.
Step5: Take the two largest flags and tag the bounding boxes and labels to these flags.
Step6: Visualize results.
Reason:
In Step4 and Step5, the Sort and Take functions are not supported.
"""
"""
Question: Tag the book written by Newton, inspired by apples
Subquestion:
Step1, Locate books from the given image, and obtain bounding boxes of books.
Step2, List one 'the book written by Newton' by asking GPT.
Step3, List one 'inspired by apples' by asking GPT.
Step4: Fliter the book obtain in Step3 by the writter is Newton.
Step5: Classify the book written by Newton, based on the bounding boxes of book obtained in Step1 and the book list obtain in Step2.
Step6: Tag the bounding boxes and labels to the classified book.
Step7: Visualize results.
Reason:
In Step4, the Fliter functions are not supported.""",    
"""
Question: Tag the book written by Newton, inspired by apples
Subquestion:
Step1, Locate books from the given image, and obtain bounding boxes of books.
Step2, List one 'the book written by Newton' by asking GPT.
Step3, List one 'inspired by apples' by asking GPT.
Step4: Classify the book written by Newton, based on the bounding boxes of book obtained in Step1 and the book list obtain in Step2.
Step5: Tag the bounding boxes and labels to the classified book.
Step6: Visualize results.
Reason:
In Step2 and Step3, it is incorrect to list 'the book written by Newton' and 'inspired by apples' separately. It should list 'the book written by Newton inspired by apples' together.
""",
"""
Question: Tag the book written by Newton, inspired by apples
Subquestion:
Step1, Locate books from the given image, and obtain bounding boxes of books.
Step2, List one 'the book written by Newton' by asking GPT.
Step3: Classify the book written by Newton, based on the bounding boxes of book obtained in Step1 and the book list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified book.
Step5: Visualize results.
Reason:
In Step2, it is incorrect to only list 'the book written by Newton'. It should list 'the book written by Newton inspired by apples'.
""",
"""
Question: Tag the book written by Newton, inspired by apples
Subquestion:
Step1, Locate books from the given image, and obtain bounding boxes of books.
Step2, List one 'the book written by Newton, inspired by apples' by asking GPT.
Step3: Classify the book written by Newton, based on the bounding boxes of book obtained in Step1 and the book list obtain in Step2.
Step4, Locate apples from the given image, and obtain bounding boxes of apples.
Step5: Tag the bounding boxes and labels to the apples.
Step6: Visualize results.
Reason:
In Step4, it is incorrect to locate apples. It should directly tag classified books.
""",
"""
Question: Tag the book written by Newton, inspired by apples
Subquestion:
Step1, Locate books written by Newton from the given image, and obtain bounding boxes of books.
Step2, List one 'the book inspired by the apples' by asking GPT.
Step3: Classify the book inspired by the apples, based on the bounding boxes of book obtained in Step1 and the book list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified book.
Step5: Visualize results.
Reason:
In Step1, it is incorrect to locate books written by Newton. It should only locate books. 
""",
"""
Question: Tag two utensils used for drinking, which are usually in the kitchen
Subquestion:
Step1, Locate utensil from the given image, and obtain bounding boxes of utensil.
Step2, List two 'utensils used for drinking' by asking GPT.
Step3: Classify utensil, based on the bounding boxes of utensil obtained in Step1 and the utensil list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified utensil.
Step5: Visualize results.
Reason:
In Step2, it should list 'utensils used for drinking, which are usually in the kitchen', instead of only 'utensils used for drinking'.
""",
"""
Question: Tag two utensils used for drinking, which are usually in the kitchen
Subquestion:
Step1, Locate utensils used for drinking from the given image, and obtain bounding boxes of utensil.
Step2, List two 'utensils are usually in the kitchen' by asking GPT.
Step3: Classify utensil, based on the bounding boxes of utensil obtained in Step1 and the utensil list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified utensil.
Step5: Visualize results.
Reason:
In Step1, it should only locate utensils in the image. In Step2, it should list 'utensils used for drinking, which are usually in the kitchen', instead of only 'utensils are usually in the kitchen'.
""",
]


FAILED_PROGRAM=[""" """]


CURATED_SUBQUESTION=[
"""
Question: Tag the face of Barack Obama
Subquestion:
Step1, Detect face regions from the given image, and obtain bounding boxes of face regions.
Step2: Classify faces regions of Barack Obama, based on the face region obtained in Step1.
Step3: Tag the bounding boxes and labels to the classified face regions.
Step4: Visualize results.
""",
"""
Question: Tag the logo of NBA team Boston Celtics.
Subquestion:
Step1, Detect logo from the given image, and obtain bounding boxes of logo.
Step2: Classify logo of NBA team Boston Celtics, based on the logo region obtained in Step1.
Step3: Tag the bounding boxes and labels to the classified logo.
Step4: Visualize results.
""",
"""
Question: Tag the flag of the country Norway.
Subquestion:
Step1, Detect flag from the given image, and obtain bounding boxes of flag.
Step2: Classify flag of NBA team Boston Celtics, based on the flag region obtained in Step1.
Step3: Tag the bounding boxes and labels to the classified flag.
Step4: Visualize results.
""",
"""
Question: Tag the presidents of US
Subquestion:
Step1, Detect face regions from the given image, and obtain bounding boxes of face regions.
Step2, List 'the presidents of US' by asking GPT.
Step3: Classify faces regions of the presidents of US, based on the face region obtained in Step1 and the president list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified face regions.
Step5: Visualize results.
""",
"""
Question: Tag the wild animals that lives on the land
Subquestion:
Step1, Locate wild animals from the given image, and obtain bounding boxes of wild animals.
Step2, List 'wild animals that lives on the land' by asking GPT.
Step3: Classify wild animals that lives on the land, based on the bounding boxes of wild animals obtained in Step1 and the wild animal list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified wild animals.
Step5: Visualize results.
""",
"""
Question: Tag the shoes with their colors
Subquestion:
Step1, Locate shoes from the given image, and obtain bounding boxes of shoes.
Step2, List 'colors' by asking GPT.
Step3: Classify shoes by their colors, based on the bounding boxes of shoes obtained in Step1 and the color list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified shoes.
Step5: Visualize results.
""",
"""
Question: Tag the shoes by their type
Subquestion:
Step1, Locate shoes from the given image, and obtain bounding boxes of shoes.
Step2, List 'type of shoes' by asking GPT.
Step3: Classify shoes by their types, based on the bounding boxes of shoes obtained in Step1 and the shoe type list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified shoes.
Step5: Visualize results.
""",
"""
Question: Tag the shoes (4) by their type
Subquestion:
Step1, Locate shoes from the given image, and obtain bounding boxes of shoes.
Step2, List 4 'type of shoes' by asking GPT.
Step3: Classify shoes by their types, based on the bounding boxes of shoes obtained in Step1 and the shoe type list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified shoes.
Step5: Visualize results.
""",
"""
Question: Tag oscar winning hollywood actors
Subquestion:
Step1, Detect face regions from the given image, and obtain bounding boxes of face regions.
Step2, List 'oscar winning hollywood actors' by asking GPT.
Step3: Classify face regions of oscar winning hollywood actors, based on the bounding boxes of faces obtained in Step1 and the actor list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified face regions.
Step5: Visualize results.
""",
"""
Question: Tag these dishes with their cuisines
Subquestion:
Step1, Locate dishes from the given image, and obtain bounding boxes of dishes.
Step2, List 'cuisines' by asking GPT.
Step3: Classify dishes by their cuisines, based on the bounding boxes of dishes obtained in Step1 and the cuisine list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified dishes.
Step5: Visualize results.
""",
"""
Question: Tag the utensils used for drinking, which are usually in the kitchen
Subquestion:
Step1, Locate utensil from the given image, and obtain bounding boxes of utensil.
Step2, List 'utensils used for drinking, which are usually in the kitchen' by asking GPT.
Step3: Classify utensil, based on the bounding boxes of utensil obtained in Step1 and the utensil list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified utensil.
Step5: Visualize results.
""",
"""
Question: Tag the painting that have a shade of blue
Subquestion:
Step1, Locate painting from the given image, and obtain bounding boxes of painting.
Step2, List 'painting that have a shade of blue' by asking GPT.
Step3: Classify paintings, based on the bounding boxes of paintings obtained in Step1 and the painting list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified painting.
Step5: Visualize results.
""",
"""
Question: Tag signs (10) that have a shade of blue
Subquestion:
Step1, Locate signs from the given image, and obtain bounding boxes of signs.
Step2, List 10 'signs that have a shade of blue' by asking GPT.
Step3: Classify signs, based on the bounding boxes of signs obtained in Step1 and the signs list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified signs.
Step5: Visualize results.
""",
"""
Question: Tag leaders in Japan, Korse, and America
Subquestion:
Step1, Detect face regions from the given image, and obtain bounding boxes of faces.
Step2, List 3 'leaders in Japan, Korse, and America' by asking GPT.
Step3: Classify faces of leaders in Japan, Korse, and America, based on the bounding boxes of faces obtained in Step1 and the leader list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified faces.
Step5: Visualize results.
""",
"""
Question: Tag the actor who played Harry Potter
Subquestion:
Step1, Detect face regions from the given image, and obtain bounding boxes of faces.
Step2, List 'the actor who played Harry Potter' by asking GPT.
Step3: Classify face of the actor who played Harry Potter, based on the bounding boxes of faces obtained in Step1 and the actor list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified faces.
Step5: Visualize results.
""",
"""
Question: Tag the 7 dwarfs in Snow White
Subquestion:
Step1, Detect face regions from the given image, and obtain bounding boxes of faces.
Step2, List 7 'the 7 dwarfs in Snow White' by asking GPT.
Step3: Classify face of the 7 dwarfs in Snow White, based on the bounding boxes of faces obtained in Step1 and the dwarf list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified faces.
Step5: Visualize results.
""",
"""
Question: Tag the book written by Newton, inspired by apples
Subquestion:
Step1, Locate books from the given image, and obtain bounding boxes of books.
Step2, List 'the book written by Newton inspired by the apples' by asking GPT.
Step3: Classify the book written by Newton inspired by the apples, based on the bounding boxes of book obtained in Step1 and the book list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified book.
Step5: Visualize results.
""",
]


CURATED_PROGRAMS=[
"""
Question: Tag the face of Barack Obama
Subquestion:
Step1, Detect face regions from the given image, and obtain bounding boxes of face regions.
Step2: Classify faces regions of Barack Obama, based on the face region obtained in Step1.
Step3: Tag the bounding boxes and labels to the classified face regions.
Step4: Visualize results.
Program:
OBJ0=FACEDET(image=IMAGE)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories='Barack Obama')
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Tag the logo of NBA team Boston Celtics.
Subquestion:
Step1, Detect logo from the given image, and obtain bounding boxes of logo.
Step2: Classify logo of NBA team Boston Celtics, based on the logo region obtained in Step1.
Step3: Tag the bounding boxes and labels to the classified logo.
Step4: Visualize results.
Program:
OBJ0=LOC(image=IMAGE,object='logo')
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories='NBA team Boston Celtics')
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Tag the flag of the country Norway.
Subquestion:
Step1, Detect flag from the given image, and obtain bounding boxes of flag.
Step2: Classify flag of NBA team Boston Celtics, based on the flag region obtained in Step1.
Step3: Tag the bounding boxes and labels to the classified flag.
Step4: Visualize results.
Program:
OBJ0=LOC(image=IMAGE,object='flag')
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories='Norway')
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Tag the presidents of US
Subquestion:
Step1, Detect face regions from the given image, and obtain bounding boxes of face regions.
Step2, List 'the presidents of US' by asking GPT.
Step3: Classify faces regions of the presidents of US, based on the face region obtained in Step1 and the president list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified face regions.
Step5: Visualize results.
Program:
OBJ0=FACEDET(image=IMAGE)
LIST0=LIST(query='presidents of US',max=20)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Tag the wild animals that lives on the land
Subquestion:
Step1, Locate wild animals from the given image, and obtain bounding boxes of wild animals.
Step2, List 'wild animals that lives on the land' by asking GPT.
Step3: Classify wild animals that lives on the land, based on the bounding boxes of wild animal obtained in Step1 and the wild animals list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified wild animals.
Step5: Visualize results.
Program:
OBJ0=LOC(image=IMAGE,object='wild animal')
LIST0=LIST(query='wild animals that lives on the land',max=20)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Tag the shoes with their colors
Subquestion:
Step1, Locate shoes from the given image, and obtain bounding boxes of shoes.
Step2, List 'colors' by asking GPT.
Step3: Classify shoes by their colors, based on the bounding boxes of shoes obtained in Step1 and the color list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified shoes.
Step5: Visualize results.
Program:
OBJ0=LOC(image=IMAGE,object='shoe')
LIST0=LIST(query='colors',max=20)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Tag the shoes by their type
Subquestion:
Step1, Locate shoes from the given image, and obtain bounding boxes of shoes.
Step2, List 'type of shoes' by asking GPT.
Step3: Classify shoes by their types, based on the bounding boxes of shoes obtained in Step1 and the shoe type list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified shoes.
Step5: Visualize results.
Program:
OBJ0=LOC(image=IMAGE,object='shoe')
LIST0=LIST(query='type of shoes',max=20)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Tag the shoes (4) by their type
Subquestion:
Step1, Locate shoes from the given image, and obtain bounding boxes of shoes.
Step2, List 4 'type of shoes' by asking GPT.
Step3: Classify shoes by their types, based on the bounding boxes of shoes obtained in Step1 and the shoe type list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified shoes.
Step5: Visualize results.
Program:
OBJ0=LOC(image=IMAGE,object='shoe')
LIST0=LIST(query='type of shoes',max=4)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Tag oscar winning hollywood actors
Subquestion:
Step1, Detect face regions from the given image, and obtain bounding boxes of face regions.
Step2, List 'oscar winning hollywood actors' by asking GPT.
Step3: Classify face regions of oscar winning hollywood actors, based on the bounding boxes of faces obtained in Step1 and the shoe actor list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified face regions.
Step5: Visualize results.
Program:
OBJ0=FACEDET(image=IMAGE)
LIST0=LIST(query='oscar winning hollywood actors',max=20)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Tag these dishes with their cuisines
Subquestion:
Step1, Locate dishes from the given image, and obtain bounding boxes of dishes.
Step2, List 'cuisines' by asking GPT.
Step3: Classify dishes by their cuisines, based on the bounding boxes of dishes obtained in Step1 and the cuisine list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified dishes.
Step5: Visualize results.
Program:
OBJ0=LOC(image=IMAGE,object='dish')
LIST0=LIST(query='cuisines',max=20)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Tag the utensils used for drinking, which are usually in the kitchen
Subquestion:
Step1, Locate utensil from the given image, and obtain bounding boxes of utensil.
Step2, List 'utensils used for drinking, which are usually in the kitchen' by asking GPT.
Step3: Classify utensil, based on the bounding boxes of utensil obtained in Step1 and the utensil list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified utensil.
Step5: Visualize results.
Program:
OBJ0=LOC(image=IMAGE,object='utensil')
LIST0=LIST(query='utensils used for drinking, which are usually in the kitchen',max=20)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Tag the painting that have a shade of blue
Subquestion:
Step1, Locate paintings from the given image, and obtain bounding boxes of paintings.
Step2, List 'painting that have a shade of blue' by asking GPT.
Step3: Classify paintings, based on the bounding boxes of paintings obtained in Step1 and the painting list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified paintings.
Step5: Visualize results.
Program:
OBJ0=LOC(image=IMAGE,object='painting')
LIST0=LIST(query='painting that have a shade of blue',max=1)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Tag pictures (10) that have a shade of blue
Subquestion:
Step1, Locate pictures from the given image, and obtain bounding boxes of pictures.
Step2, List 10 'pictures that have a shade of blue' by asking GPT.
Step3: Classify pictures, based on the bounding boxes of pictures obtained in Step1 and the pictures list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified pictures.
Step5: Visualize results.
Program:
OBJ0=LOC(image=IMAGE,object='picture')
LIST0=LIST(query='picturess that have a shade of blue',max=10)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Tag leaders in Japan, Korse, and America
Subquestion:
Step1, Detect face regions from the given image, and obtain bounding boxes of faces.
Step2, List 'leaders in Japan, Korse, and America' by asking GPT.
Step3: Classify faces of leaders in Japan, Korse, and America, based on the bounding boxes of faces obtained in Step1 and the leader list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified faces.
Step5: Visualize results.
Program:
OBJ0=FACEDET(image=IMAGE)
LIST0=LIST(query='leaders in Japan, Korse, and America',max=3)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Tag the actor who played Harry Potter
Subquestion:
Step1, Detect face regions from the given image, and obtain bounding boxes of faces.
Step2, List 'the actor who played Harry Potter' by asking GPT.
Step3: Classify face of the actor who played Harry Potter, based on the bounding boxes of faces obtained in Step1 and the actor list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified faces.
Step5: Visualize results.
Program:
OBJ0=FACEDET(image=IMAGE)
LIST0=LIST(query='actor who played Harry Potter',max=1)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Tag the 7 dwarfs in Snow White
Subquestion:
Step1, Detect face regions from the given image, and obtain bounding boxes of faces.
Step2, List 'the 7 dwarfs in Snow White' by asking GPT.
Step3: Classify face of the 7 dwarfs in Snow White, based on the bounding boxes of faces obtained in Step1 and the dwarf list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified faces.
Step5: Visualize results.
Program:
OBJ0=FACEDET(image=IMAGE)
LIST0=LIST(query='dwarfs in snow white',max=7)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
"""
Question: Tag the book written by Newton, inspired by apples
Subquestion:
Step1, Locate books from the given image, and obtain bounding boxes of books.
Step2, List 'the book written by Newton inspired by the apples' by asking GPT.
Step3: Classify the book written by Newton inspired by the apples, based on the bounding boxes of book obtained in Step1 and the book list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified book.
Step5: Visualize results.
Program:
OBJ0=LOC(image=IMAGE,object='book')
LIST0=LIST(query='book written by Newton, inspired by the apples',max=1)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
IMAGE0=TAG(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
""",
]


REFLECTION_STEP=[
"""
Question: Tag the wild animals.
The description of input image: a photography of a several person on the seat
Human Feedback: There are errors in tagging this instance. For this instance, the precision, recall, and F1 score of our method is 0.0, 0.0, and 0, respectively. There are 1 objects should be tagged, while our method tags 2 objects. The details of desirable prediction is [([545, 241, 562, 266], 'rabbit'),([592, 245, 606, 279], 'fox')], while our prediction is [([545, 241, 562, 266], 'gorilla'),([592, 245, 606, 279], 'deer')].
Following are the decomposed subquestion, used program, and obtained result in each step. 
SubQuetion:
Program and obtained result in each step:
Step1
Program: OBJ0=LOC(image=IMAGE,object='wild animal')
Result of The coordinate of OBJ0: [[545, 241, 562, 266], [592, 245, 606, 279]]
Step2
Program: LIST0=LIST(query='wild animals',max=20)
Results of LIST0: ['rabbit', 'fox', 'gorilla', 'deer', 'frog']
Step3
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
Result of The coordinate of OBJ0: [[545, 241, 562, 266], [592, 245, 606, 279]]
Step4
Program: IMAGE0=TAG(image=IMAGE,object=OBJ1)
The description of IMAGE0: a photography of a several person on the seat, some animals are boxed
Step5
Program: FINAL_RESULT=RESULT(var=IMAGE0)
The description of FINAL_RESULT: a photography of a several person on the seat, some animals are boxed
Error Location: functions called by programs
Reason: In the Step3 of the program, the used function 'CLASSIFY' failed to select the correct wild animals.
""",
"""
Question: Tag one utensil used for drinking, which are usually in the kitchen
Description of the Input Image: a photography of delicious food on a table
Human Feedback: There are errors in tagging this instance. For this instance, the precision, recall, and F1 score of our method is 0.0, 0.0, and 0, respectively. There are 1 objects should be tagged, while our method tags 1 objects. The details of desirable prediction is [([105, 136, 250, 337], 'cup')], while our prediction is [([[45, 78, 245, 345]], 'cup')].
Following are the decomposed subquestion, used program, and obtained result in each step. 
Subquestion:
Step1, Locate utensil from the given image, and obtain bounding boxes of utensil.
Step2, List one 'utensil used for drinking' by asking GPT.
Step3: Classify utensil, based on the bounding boxes of utensil obtained in Step1 and the utensil list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified utensil.
Step5: Visualize results.
Program and obtained result in each step:
Step1
OBJ0=LOC(image=IMAGE,object='utensil')
Result of BOX0 is [[45, 78, 245, 345],[105, 136, 250, 337]]
Step2:
LIST0=LIST(query='utensil used for drinking, which are usually in the kitchen',max=1)
Result of LIST0: ['cup']
Step3:
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
Result OBJ1 is [[45, 78, 245, 345]]
Step4:
IMAGE0=TAG(image=IMAGE,object=OBJ1)
Description of IMAGE0: a photography of delicious food on a table
Step5:
FINAL_RESULT=RESULT(var=IMAGE0)
Description of FINAL_RESULT: a photography of delicious food on a table
Error Location: functions called by programs
Reason:
In the Step3 of the program, the used function 'CLASSIFY' failed to select the correct cup.
""",
"""
Question: Tag one book written by Newton, inspired by apples
Description of the Input Image: a photography of a book on a table
Human Feedback: There are errors in tagging this instance. For this instance, the precision, recall, and F1 score of our method is 0.0, 0.0, and 0, respectively. There are 1 objects should be tagged, while our method tags 2 objects. The details of desirable prediction is [([105, 136, 250, 337], 'Philosophiae Naturalis Principia Mathematica')], while our prediction is [([[45, 78, 245, 345]], 'Philosophiae Naturalis Principia Mathematica')].
Following are the decomposed subquestion, used program, and obtained result in each step. 
Subquestion:
Step1, Locate books from the given image, and obtain bounding boxes of books.
Step2, List one 'book written by Newton, inspired by apples' by asking GPT.
Step3: Classify the book, based on the bounding boxes of book obtained in Step1 and the book list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified book.
Step5: Visualize results.
Program and obtained result in each step:
Step1
OBJ0=LOC(image=IMAGE,object='book')
Result of The coordinate of BOX0: [[45, 78, 245, 345],[105, 136, 250, 337],[167, 198, 238, 252]]
Step2:
LIST0=LIST(query='book written by Newton, inspired by apples',max=1)
Result of LIST0: ['Philosophiae Naturalis Principia Mathematica']
Step3:
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
Result of The coordinate of OBJ1: [[45, 78, 245, 345]]
Step4:
IMAGE0=TAG(image=IMAGE,object=OBJ1)
Description of IMAGE0: a photography of a book on a table
Step5:
FINAL_RESULT=RESULT(var=IMAGE0)
Description of FINAL_RESULT: a photography of a book on a table
Error Location: functions called by programs
Reason:
In the Step3 of the program, the used function 'CLASSIFY' failed to classify the book Philosophiae Naturalis Principia Mathematica.
""",
"""
Question: Tag two utensils used for drinking, which are usually in the kitchen
Description of the Input Image: a photography of delicious food on a table
Human Feedback: There are errors in tagging this instance. For this instance, the precision, recall, and F1 score of our method is 0.0, 0.0, and 0, respectively. There are 1 objects should be tagged, while our method tags 1 objects. The details of desirable prediction is [([105, 136, 250, 337], 'cup'),([[45, 78, 245, 345]], 'glass')], while our prediction is [([105, 136, 250, 337], 'glass'),([[45, 78, 245, 345]], 'cup')].
Following are the decomposed subquestion, used program, and obtained result in each step. 
Subquestion:
Step1, Locate utensil from the given image, and obtain bounding boxes of utensil.
Step2, List two 'utensils used for drinking' by asking GPT.
Step3: Classify utensil, based on the bounding boxes of utensil obtained in Step1 and the utensil list obtain in Step2.
Step4: Tag the bounding boxes and labels to the classified utensil.
Step5: Visualize results.
Program and obtained result in each step:
Step1
OBJ0=LOC(image=IMAGE,object='utensil')
Result of BOX0 is [[45, 78, 245, 345],[105, 136, 250, 337]]
Step2:
LIST0=LIST(query='utensils used for drinking, which are usually in the kitchen',max=2)
Result of LIST0: ['cup','glass']
Step3:
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)
Result OBJ1 is [[45, 78, 245, 345],[105, 136, 250, 337]]
Step4:
IMAGE0=TAG(image=IMAGE,object=OBJ1)
Description of IMAGE0: a photography of delicious food on a table
Step5:
FINAL_RESULT=RESULT(var=IMAGE0)
Description of FINAL_RESULT: a photography of delicious food on a table
Error Location: functions called by programs
Reason:
In the Step3 of the program, the used function 'CLASSIFY' failed to select the correct cup and glass, it maks confusion with the two objects.
""",
]


REFLECTION_INTERRUPT=[
"""
Question: Tag the face of Barack Obama
Subquestion:
Step1, Detect face regions from the given image, and obtain bounding boxes of face regions.
Step2: Classify faces regions of Barack Obama, based on the face region obtained in Step1.
Step3: Tag the bounding boxes and labels to the classified face regions.
Step4: Visualize results.
Program:
OBJ0=FACEDET(image=IMAGE)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories='Barack Obama')
IMAGE0=TAG(image=IMAGE1,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
Reason: 
The bug in the program is in the third line: IMAGE0=TAG(image=IMAGE1,object=OBJ1), where 'IMAGE1' is not defined. It should be 'IMAGE'.
""",
"""
Question: Tag the face of Barack Obama
Subquestion:
Step1, Detect face regions from the given image, and obtain bounding boxes of face regions.
Step2: Classify faces regions of Barack Obama, based on the face region obtained in Step1.
Step3: Tag the bounding boxes and labels to the classified face regions.
Step4: Visualize results.
Program:
OBJ0=FACEDET(image=IMAGE)
OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories='Barack Obama')
IMAGE0=LABEL(image=IMAGE,object=OBJ1)
FINAL_RESULT=RESULT(var=IMAGE0)
Reason: 
The bug in the program is in the third line: IMAGE0=LABEL(image=IMAGE,object=OBJ1), where 'LABEL' is not defined. It should be 'IMAGE'.
""",
]


INFERENCE=[""" """]

