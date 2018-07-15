import tensorflow as tf
import os
import codecs
import json
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from collections import defaultdict

pathData_ = "D:/src/quickDraw/data/"
path_ = "D:/src/quickDraw/test/"
pathImage_ = "D:/src/quickDraw/animalImg"
pathImageText_ = "D:/src/quickDraw/aimgText"

trainingSet = []
testSet = []

allClassNum = 6

allFilePath = []

image_size = 64 #4x4

def prepare_dirs(path):
    filenames = tf.gfile.ListDirectory(path)
    filenames = sorted(filenames)
    filenames = [os.path.join(path, f) for f in filenames]

    return filenames


def makeTrainingSet(files, sess, coord):

    dataDict = defaultdict(lambda : 0)
    for e in files:
        dataDict[e[1]] = []
    for e in files:
        dataDict[e[1]] += [e[0]]

    tr = []
    for a, b in dataDict.items():
        tr += makeNumpyArray(a , b, sess, coord)
        print ("loaded class " + str(a))
    random.shuffle(tr)

    return np.array(tr).T.tolist()


def makeNumpyArray(classNum, fileNames, sess, coord):

    _, images = loadImage(sess,fileNames, len(fileNames))

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    input = sess.run(images)

    output = np.zeros([allClassNum])
    output[classNum] = 1

    tmp = []
    for e in input:
        tmp += [[e, output]]

    return tmp

def loadImage(sess, filenames, imageNum):
    filename_queue = tf.train.string_input_producer(filenames)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    image = tf.image.decode_png(value, channels=1)
    image = tf.cast(image, tf.float32) / 255.0

    image.set_shape((image_size, image_size, 1))

    # Generate batch
    batch_size = imageNum
    num_preprocess_threads = 1
    min_queue_examples = imageNum
    image_batch = tf.train.batch(
        [image],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

    image_batch = tf.reshape(image_batch, [batch_size, image_size * image_size])

    return key, image_batch

def testing(bi):
    x_ = tf.placeholder(tf.float32, shape=[None, image_size * image_size])
    x_image = tf.reshape(x_, [-1, image_size, image_size, 1], name='image')

    print (tf.Session().run(tf.shape(x_image), {x_: bi}))


def prepareLabel():
    pivot = 10112
    train = []
    test = []
    for i in range(allClassNum):
        pp = prepare_dirs(path_ + str(i) + "/")
        tmp = []
        for e in pp:
            tmp += [[e, i]]
        train += tmp[0:pivot]
        test += tmp[pivot:pivot + 1024]
        print ("prepare " + str(i))

    random.shuffle(train)
    random.shuffle(test)
    nbt1 = len(train)//128
    nbt2 = len(test)//1024
    return [train[i::nbt1] for i in range(nbt1)], [test[i::nbt2] for i in range(nbt2)]


def initAllClass():
    classesDict = defaultdict(lambda: 0)
    firstClass = 0
    for file in os.listdir(pathImage_):
        if firstClass >= 9999:
            break
        if file not in classesDict:
            classesDict[file] = firstClass
            firstClass += 1

    return classesDict


def getDataForTesting(index, cd):
    imgList = []
    dictSize = len(cd)
    pIT = pathImageText_ + "/" + str(index) + ".txt"
    print (pIT)
    with open(pIT, "r") as f:
        for line in f:

            data = line.split(" | ")
            outLabel = np.zeros(dictSize)
            outLabel[cd[data[0]]] = 1

            arr = np.fromstring(data[1], sep=",")
            arr = np.reshape(arr, [64, 64]).tolist()

            if random.choice([True, False]):
                arr = np.flip(arr, 1)

            arr = np.reshape(arr , [4096])

            imgList += [[arr, outLabel]]

    random.shuffle(imgList)

    return imgList

def imageToArray():
    p = "D:/src/quickDraw/test/"
    l = []
    for i in range(1,10):
        pixes = mpimg.imread(p + str(i) + ".png")
        pixes = pixes[:, :, 0]
        print (np.shape(pixes))
        l.append(np.reshape(pixes, [4096]))

    return l

def getDataFromTxt(index, cd):
    imgList = []
    dictSize = len(cd)
    pIT = pathImageText_ + "/" + str(index) + ".txt"
    print (pIT)
    with open(pIT, "r") as f:
        for line in f:

            data = line.split(" | ")
            outLabel = np.zeros(dictSize)
            outLabel[cd[data[0]]] = 1

            arr = np.fromstring(data[1], sep=",")
            arr = np.reshape(arr, [64, 64]).tolist()

            if random.choice([True, False]):
                arr = np.flip(arr, 1)

            arr = np.reshape(arr , [4096])

            imgList += [[arr, outLabel]]

    #random.shuffle(imgList)

    trS = imgList[0:31104]
    teS = imgList[31104:]

    nbt1 = len(trS) // 64
    nbt2 = len(teS) // 64
    return [trS[i::nbt1] for i in range(nbt1)] , [teS[i::nbt2] for i in range(nbt2)]


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

def testDraw(data, ext):
    plt.figure(figsize=(1, 1))
    plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    for e in data:
        plt.plot(e[0], e[1], color='black',linewidth=np.random.uniform(1,2))

    plt.savefig(pathImage_ + ext + '.png', bbox_inches='tight', dpi=59, origin='upper',cmap=plt.cm.gray)
    plt.close()

def fromNdjson2Img():

    classesDict = defaultdict(lambda: 0)
    for file in os.listdir(pathImage_):
        classesDict[file] = 1

    isStart = False
    startLine = 22000
    endLine = 44000

    processing = True

    for file in os.listdir(pathData_):
        if file.endswith(".ndjson"):
            fp = pathData_ + file
            fClass = file.replace(".ndjson","")

            if "flamingo" == fClass:
                isStart = True
            if "mermaid" == fClass:
                processing = False

            if processing:
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


#fromNdjson2Img()

