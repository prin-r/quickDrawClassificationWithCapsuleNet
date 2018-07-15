import tensorflow as tf
import os
import numpy as np
import random

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
path_ = "D:/src/quickDraw/test/"

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


def makeTrainingSet():
    tr = []
    for i in range(0,allClassNum,1):
        tr += makeNumpyArray(path_ + str(i) + '/', i)
        print ("loaded class " + str(i))
    random.shuffle(tr)

    return np.array(tr).T.tolist()


def makeNumpyArray(path,classNum):
    sess = tf.InteractiveSession()
    file = prepare_dirs(path)
    _, images = loadImage(sess,file)
    threads = tf.train.start_queue_runners(sess=sess)

    input = sess.run(images)

    output = np.zeros([allClassNum])
    output[classNum] = 1

    tmp = []
    for e in input:
        tmp += [[e, output]]

    return tmp

def loadImage(sess, filenames):
    filename_queue = tf.train.string_input_producer(filenames)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    image = tf.image.decode_png(value, channels=1)
    image = tf.cast(image, tf.float32) / 255.0

    image.set_shape((image_size, image_size, 1))

    # Generate batch
    batch_size = 128
    num_preprocess_threads = 1
    min_queue_examples = 128
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
    pivot = 9600
    train = []
    test = []
    for i in range(allClassNum):
        pp = prepare_dirs(path_ + str(i) + "/")
        tmp = []
        for e in pp:
            tmp += [[e, i]]
        train += tmp[0:pivot]
        test += tmp[pivot:]
        print ("prepare " + str(i))

    random.shuffle(train)
    random.shuffle(test)
    return train, test


"""

trainingSet = makeTrainingSet()
trainingSet = np.transpose(trainingSet)
for e in trainingSet:
    print (e)

print (np.shape(trainingSet))

"""
#trainingSet = makeTrainingSet()
#testing(trainingSet[0])


trainingSet, testSet = prepareLabel()

for e in trainingSet:
    print (e)

