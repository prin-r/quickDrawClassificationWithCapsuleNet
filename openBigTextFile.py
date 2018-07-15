import sys
import os
import numpy as np
import codecs
import json

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import random
from collections import defaultdict

pathData_ = "D:/src/quickDraw/data/"
pathImage_ = "D:/src/quickDraw/animalImg"
pathImageText_ = "D:/src/quickDraw/imageText"

listSizes = []
imageSize = 64*64

def prepareDir():
    for file in os.listdir(pathData_):
        if file.endswith(".ndjson"):
            fp = pathData_ + file
            with codecs.open(fp, 'r', 'utf-8') as f:
                class_name = ''
                for line in f:
                    sample = json.loads(line)
                    class_name = sample["word"]
                    break

                imgPath = pathImage_ + '/' + class_name + '/'
                directory = os.path.dirname(imgPath)
                print (imgPath)

                try:
                    os.stat(directory)
                except:
                    os.mkdir(directory)

def testDraw(data, ext):
    plt.figure(figsize=(1, 1))
    plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    for e in data:
        plt.plot(e[0], e[1], color='black',linewidth=np.random.uniform(1,2))

    plt.savefig(pathImage_ + ext + '.png', bbox_inches='tight', dpi=59, origin='upper',cmap=plt.cm.gray)
    plt.close()

def loadFiles():
    isStart = False
    for file in os.listdir(pathData_):
        if file.endswith(".ndjson"):
            fp = pathData_ + file
            fClass = file.replace(".ndjson","")
            if "tractor" == fClass:
                isStart = True

            if isStart:
                with codecs.open(fp, 'r', 'utf-8') as f:
                    i = 0
                    for line in f:
                        img , cName = parse_line(line)

                        #print (np.size(img, axis=0) , cName)
                        testDraw(img, '/' + cName + '/' + str(i))
                        i += 1
                        if i%200 == 0:
                            if i > 11000:
                                break
                            print (cName + ' ' + str(i))

def parse_line(ndjson_line):
  """Parse an ndjson line and return ink (as np array) and classname."""
  sample = json.loads(ndjson_line)
  class_name = sample["word"]
  inkarray = sample["drawing"]

  maxY = np.max(np.max([y[1] for y in inkarray]))
  for e in inkarray:
      e[1] = maxY - e[1]

  shouldFlipX = random.choice([True, False])

  if shouldFlipX:
      maxX = np.max(np.max([x[0] for x in inkarray]))
      for e in inkarray:
          e[0] = maxX - e[0]

  return inkarray, class_name

def imageToString():
    isStart = False

    classesDict = defaultdict(lambda : 0)

    maxFile = 10
    countFile = 0

    for file in os.listdir(pathImage_):
        if countFile < maxFile:
            classesDict[file] = 1
            countFile += 1
        else :
            break

    batchIndex = 0
    numBatchPerClass = 128 * 26

    for _ in range(13):
        allLine = []
        for a, b in classesDict.items():

            p2i = pathImage_ + "/" + a + "/"

            startIndex = batchIndex * numBatchPerClass
            iterIndex = startIndex
            endIndex = startIndex + numBatchPerClass

            for __ in range(numBatchPerClass):
                if iterIndex < endIndex:
                    pixes = mpimg.imread(p2i + str(iterIndex) + ".png")
                    pixes = pixes[:, :, 0]
                    pixes = np.reshape(pixes,[imageSize])
                    allLine += [a + " | " + str(pixes.tolist())[1:-1] + '\n']
                    iterIndex += 1
                else :
                    break

            print(a + " " + str(iterIndex))

        random.shuffle(allLine)

        f = open(pathImageText_ + "/" + str(batchIndex) + ".txt", "w")
        for line in allLine:
            f.write(line)
        f.close()

        print ("batch " + str(batchIndex))
        batchIndex += 1

    print (len(classesDict))

def fromNdjson2Img():

    classesDict = defaultdict(lambda: 0)
    for file in os.listdir(pathImage_):
        classesDict[file] = 1

    isStart = True
    startLine = 11000
    endLine = 22000

    for file in os.listdir(pathData_):
        if file.endswith(".ndjson"):
            fp = pathData_ + file
            fClass = file.replace(".ndjson","")
            if "tractor" == fClass:
                isStart = True

            if isStart and fClass in classesDict:
                with codecs.open(fp, 'r', 'utf-8') as f:
                    i = 0
                    for line in f:
                        img , cName = parse_line(line)
                        i += 1
                        if i > startLine:
                            testDraw(img, '/' + cName + '/' + str(i))
                        if i > endLine:
                            break
                        if i%200 == 0:
                            print (cName + ' ' + str(i))


def testReadData():
    trainingSetFileName = "TrainingSet.txt"
    with codecs.open(trainingSetFileName, 'r', 'utf-8') as f:
        for line in f:
            img, cName = parse_line(line)
            print (cName)
            plt.figure(figsize=(1, 1))
            plt.axis('off')
            plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off',
                            labelright='off', labelbottom='off')
            for e in img:
                plt.plot(e[0], e[1], color='black', linewidth=np.random.uniform(1, 2))

            plt.show()
            plt.close()


#prepareDir()
#loadFiles()
#testDraw()
#imageToString()

testReadData()



