import json
import sys
import os
import urllib.request as ureq
from tqdm import tqdm

download=1 # 0 if images are already downloaded

###############################################################
######################### load dataset json file ###############
################################################################
with open('dataset.json', 'r') as fp:
        data = json.load(fp)

## dictionary data contains image URL, questions and answers ##




################################################################
############### Script for downloading images ##################
################################################################
## Make a directory images to store all images there ##########
if download == 1:
    os.mkdir('./images')
    for k in tqdm(data.keys()):
        ext=os.path.splitext(data[k]['imageURL'])[1]
        outputFile='images/%s%s'%(k,ext)
        ureq.urlretrieve(data[k]['imageURL'],outputFile)    



#################################################################
################### Example of data access #####################
################################################################
for k in data.keys():
    ext=os.path.splitext(data[k]['imageURL'])[1]
    imageFile='images/%s%s'%(k,ext)

    print('************************')
    print('Image file: %s'%(imageFile))
    print('List of questions:')
    print(data[k]['questions'])
    print('List of corresponding answers:')
    print(data[k]['answers'])
    print('Use this image as training (1), validation (2) or testing (3): %s'%(data[k]['split']))
    print('*************************')





######################################################################
########################### Get dataset stats ########################
######################################################################
genSet=set()
for k in data.keys():
    genSet.add(data[k]['genre'])



numImages=len(data.keys())
numQApairs=0
numWordsInQuestions=0
numWordsInAnswers=0
numQuestionsPerImage=0
ANS=set() # Set of unique answers
authorSet=set()
bookSet=set()


for imgId in data.keys():
    numQApairs = numQApairs+len(data[imgId]['questions'])
    numQuestionsPerImage = numQuestionsPerImage + len(data[imgId]['questions'])
    authorSet.add(data[imgId]['authorName'])
    bookSet.add(data[imgId]['title'])

    for qno in range(len(data[imgId]['questions'])):
        ques=data[imgId]['questions'][qno]
        numWordsInQuestions = numWordsInQuestions+len(ques.split())
    for ano in range(len(data[imgId]['answers'])):
        ans=data[imgId]['answers'][ano]
        ANS.add(ans)
        numWordsInAnswers = numWordsInAnswers+len(str(ans).split())



print("--------------------------------")
print("Number of Images: %d" %(numImages))
print("Number of QA pairs: %d" %(numQApairs))
print("Number of unique author: %d" %(len(authorSet)))
print("Number of unique title: %d" %(len(bookSet)))
print("Number of unique answers: %d" %(len(ANS)))
print("Number of unique genre: %d" %(len(genSet)))
print("Average question length (in words): %.2f" %(float(numWordsInQuestions)/float(numQApairs)))
print("Average answer length (in words): %.2f" %(float(numWordsInAnswers)/float(numQApairs)))
print("Average number of questions per image: %.2f" %(float(numQuestionsPerImage)/float(numImages)))
print("--------------------------------")

