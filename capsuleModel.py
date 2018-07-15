from __future__ import division, print_function, unicode_literals

import matplotlib
import random
import matplotlib.pyplot as plt
import util
import numpy as np
import tensorflow as tf

tf.reset_default_graph()

np.random.seed(42)
tf.set_random_seed(42)

classDict = ct.initAllClass()
classNum = len(classDict)
inputSize = 64

savePath_ = "D:/src/quickDraw/saveModel/"

def get_cropReSizeParam(arr):
    boxes = []
    boxes_id = []
    rpow = random.uniform(0.5, 2)
    for i in range(len(arr)):
        boxes.append([random.uniform(0.0, 0.15625), random.uniform(0.0, 0.15625), random.uniform(0.84375, 1.0),
                      random.uniform(0.84375, 1.0)])
        boxes_id.append(i)
    return boxes, boxes_id, rpow

def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector


def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)

def model():

    # BEGIN_MODEL-----------------------------------------------------------------------------------------------------------------

    caps1_n_maps = 32
    caps1_n_caps = caps1_n_maps * 6 * 6  # 512 primary capsules
    caps1_n_dims = 8

    x_ = tf.placeholder(tf.float32, shape=[None, inputSize * inputSize], name="image")
    y_ = tf.placeholder(tf.float32, shape=[None, classNum], name="target")

    boxes_ = tf.placeholder(tf.float32, shape=[None, 4])
    boxes_id_ = tf.placeholder(tf.int32, shape=[None])
    rpow_ = tf.placeholder(tf.float32)


    X = tf.reshape(x_, [-1, inputSize, inputSize, 1], name='X')

    X_crop_and_resize = tf.image.crop_and_resize(X, boxes=boxes_, box_ind=boxes_id_, crop_size=[28,28])
    X_contrast = tf.pow(X_crop_and_resize, rpow_)

    W_conv1 = tf.Variable(tf.truncated_normal([9, 9, 1, 256], stddev=0.1) , name='w_cov1')
    conv1 = tf.nn.conv2d(X_contrast , W_conv1, strides=[1, 1, 1, 1], padding='VALID')
    relu1 = tf.maximum(conv1, 0.1 * conv1)

    W_conv2 = tf.Variable(tf.truncated_normal([9, 9, 256, 256], stddev=0.1) , name='w_cov2')
    conv2 = tf.nn.conv2d(relu1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') #[128   6 6 256]
    relu2 = tf.maximum(conv2, 0.1 * conv2) #can't find out any words from the paper whether the PrimaryCap convolution does a ReLU activation or not before squashing function, but experiment show that using ReLU get a higher test accuracy

    caps1_raw = tf.reshape(relu2, [-1, caps1_n_caps, caps1_n_dims], name="caps1_raw") #[ 128 1152 8]

    caps1_output = squash(caps1_raw, name="caps1_output") #[ 128 1152  8]

    caps2_n_caps = classNum#10
    caps2_n_dims = 16

    init_sigma = 0.01

    W_init = tf.random_normal( shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims), stddev=init_sigma, dtype=tf.float32, name="W_init")
    W = tf.Variable(W_init, name="W")

    batch_size = tf.shape(X_contrast )[0]
    W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

    caps1_output_expanded = tf.expand_dims(caps1_output, -1, name="caps1_output_expanded")

    caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2, name="caps1_output_tile")

    caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1], name="caps1_output_tiled") #[ 128 512  10    8    1]

    caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted")

    raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1], dtype=np.float32, name="raw_weights")

    # Round 1
    routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")
    weighted_predictions = tf.multiply(routing_weights, caps2_predicted, name="weighted_predictions")
    weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True, name="weighted_sum")
    caps2_output_round_1 = squash(weighted_sum, axis=-2, name="caps2_output_round_1")

    # Round 2
    caps2_output_round_1_tiled = tf.tile(caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],name="caps2_output_round_1_tiled")
    agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,transpose_a=True, name="agreement")
    raw_weights_round_2 = tf.add(raw_weights, agreement,name="raw_weights_round_2")

    routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,dim=2,name="routing_weights_round_2")
    weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,caps2_predicted,name="weighted_predictions_round_2")
    weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,axis=1, keep_dims=True,name="weighted_sum_round_2")
    caps2_output_round_2 = squash(weighted_sum_round_2,axis=-2,name="caps2_output_round_2")

    # Round 3
    caps2_output_round_2_tiled = tf.tile(caps2_output_round_2, [1, caps1_n_caps, 1, 1, 1],name="caps2_output_round_2_tiled")
    agreement = tf.matmul(caps2_predicted, caps2_output_round_2_tiled,transpose_a=True, name="agreement")
    raw_weights_round_3 = tf.add(raw_weights, agreement,name="raw_weights_round_3")

    routing_weights_round_3 = tf.nn.softmax(raw_weights_round_3,dim=2,name="routing_weights_round_3")
    weighted_predictions_round_3 = tf.multiply(routing_weights_round_3,caps2_predicted,name="weighted_predictions_round_3")
    weighted_sum_round_3 = tf.reduce_sum(weighted_predictions_round_3,axis=1, keep_dims=True,name="weighted_sum_round_3")
    caps2_output_round_3 = squash(weighted_sum_round_3,axis=-2,name="caps2_output_round_3") #[128   1  10  16   1]

    caps2_output = caps2_output_round_3


    #ROUND_END

    y_proba = safe_norm(caps2_output, axis=-2, name="y_proba") #[128   1  10   1]
    y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba") #[128   1   1]
    y_pred = tf.squeeze(y_proba_argmax, axis=[1, 2], name="y_pred") #[128]

    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5

    caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True, name="caps2_output_norm") #[128   1  10   1   1]

    present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),name="present_error_raw")
    present_error = tf.reshape(present_error_raw, shape=(-1, classNum),name="present_error")

    absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),name="absent_error_raw")
    absent_error = tf.reshape(absent_error_raw, shape=(-1, classNum),name="absent_error")

    L = tf.add(y_ * present_error, lambda_ * (1.0 - y_) * absent_error,name="L")

    margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

    mask_with_labels = tf.placeholder_with_default(False, shape=(),name="mask_with_labels")
    reconstruction_targets = tf.cond(mask_with_labels,  # condition
                                     lambda: tf.argmax(y_, axis=1),  # if True
                                     lambda: y_pred,  # if False
                                     name="reconstruction_targets")

    reconstruction_mask = tf.one_hot(reconstruction_targets,
                                     depth=caps2_n_caps,
                                     name="reconstruction_mask")

    reconstruction_mask_reshaped = tf.reshape(
        reconstruction_mask, [-1, 1, caps2_n_caps, 1, 1],
        name="reconstruction_mask_reshaped")

    caps2_output_masked = tf.multiply(
        caps2_output, reconstruction_mask_reshaped,
        name="caps2_output_masked")

    decoder_input = tf.reshape(caps2_output_masked,
                               [-1, caps2_n_caps * caps2_n_dims],
                               name="decoder_input")

    n_hidden1 = 512
    n_hidden2 = 1024
    n_output = 28 * 28

    with tf.name_scope("decoder"):
        hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                                  activation=tf.nn.relu,
                                  name="hidden1")
        hidden2 = tf.layers.dense(hidden1, n_hidden2,
                                  activation=tf.nn.relu,
                                  name="hidden2")
        decoder_output = tf.layers.dense(hidden2, n_output,
                                         activation=tf.nn.sigmoid,
                                         name="decoder_output")

    X_flat = tf.reshape(X_crop_and_resize, [-1, n_output], name="X_flat")
    squared_difference = tf.square(X_flat - decoder_output,
                                   name="squared_difference")
    reconstruction_loss = tf.reduce_mean(squared_difference,
                                         name="reconstruction_loss")

    alpha = 0.0005

    loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")

    correct = tf.equal(tf.argmax(y_, axis=1), y_pred, name="correct")
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss, name="training_op")

    # END_MODEL-----------------------------------------------------------------------------------------------------------------

    trainAcc = []
    testAcc = []
    saveIndex = 0
    j = 0

    restoreNum = 224
    shouldRestore = True

    isTesting = True

    for a, b in classDict.items():
        print (a,b)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        if shouldRestore:
            saver.restore(sess, savePath_ + 'my-model-' + str(restoreNum))

        if isTesting:
            fileIndex = 0
            tr = ct.imageToArray()
            print("Testing")
            score = 0
            totalIter = 0
            for e in tr:
                ypred = y_pred.eval(feed_dict={x_: [e], boxes_: [[0, 0, 1, 1]], boxes_id_: [0], rpow_: 1})
                ypred = ypred[0]
                img = np.reshape(e,[64,64])

                predText = ""
                tarText = ""

                for a, b in classDict.items():
                    if b == ypred:
                        predText = a

                plt.imshow(img)
                plt.show()
                print ("pred = " + predText)
                print("tar = " + tarText)
                print ("score = " + str(score) + "/" + str(totalIter))
                print ("____________________________________________________")
        else :
            for i in range(21,500000,1):
                tr, te = ct.getDataFromTxt(i % 68, classDict)
                print ("load epoch " + str(i))
                ttest = np.array(te[0]).T.tolist()
                for e in tr:
                    tt = np.array(e).T.tolist()
                    if j % 25 == 0:

                        boxes, boxes_id, rpow = get_cropReSizeParam(tt[0])

                        train_accuracy = accuracy.eval(feed_dict={x_: tt[0], y_: tt[1], boxes_ : boxes, boxes_id_ : boxes_id, rpow_ : rpow})

                        boxes, boxes_id, rpow = get_cropReSizeParam(ttest[0])

                        numTestSet = 0
                        test_accuracy = 0
                        for eTes in te:
                            etb = np.array(eTes).T.tolist()
                            test_accuracy += accuracy.eval(feed_dict={x_: etb[0], y_: etb[1], boxes_: boxes, boxes_id_: boxes_id, rpow_: rpow})
                            print ("testing : " + str(numTestSet) )
                            numTestSet += 1

                        test_accuracy /= numTestSet

                        trainAcc.append([train_accuracy, j / 25])
                        testAcc.append([test_accuracy, j / 25])

                        print('step %d, training accuracy %g' % (j, train_accuracy))
                        print('step %d, testing accuracy %g' % (j, test_accuracy))

                    if j % 344 == 0 and j > 0:
                        trainAccT = np.array(trainAcc).T.tolist()
                        testAccT = np.array(testAcc).T.tolist()
                        plt.plot(trainAccT[1], trainAccT[0], label='train')
                        plt.plot(testAccT[1], testAccT[0], label='test')

                        ma25 = []
                        ma50 = []
                        tmpExp25 = 0
                        tmpExp50 = 0
                        for e in testAccT[0]:
                            tmpExp25 = 0.9 * tmpExp25 + 0.1 * e
                            ma25.append(tmpExp25)

                            tmpExp50 = 0.975 * tmpExp50 + 0.025 * e
                            ma50.append(tmpExp50)

                        plt.plot(testAccT[1], ma25, label='ma25')
                        plt.plot(testAccT[1], ma50, label='ma50')

                        plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
                        saveIndex %= 10
                        saveIndex += 1
                        plt.savefig("D:/src/quickDraw/" + str(saveIndex) + ".png")
                        plt.close()

                        saver.save(sess, savePath_ + 'my-model-' + str(i))

                    boxes, boxes_id, rpow = get_cropReSizeParam(tt[0])
                    training_op.run(feed_dict={x_: tt[0], y_: tt[1] , boxes_ : boxes, boxes_id_ : boxes_id, rpow_ : rpow})
                    j += 1

                print ("trained epoch " + str(i))


model()
