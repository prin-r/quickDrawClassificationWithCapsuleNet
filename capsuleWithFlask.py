from __future__ import division, print_function, unicode_literals

import random
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask,redirect
from flask import request, jsonify, json, make_response, current_app
from flask_cors import CORS, cross_origin

from PIL import Image
from io import BytesIO
import base64
import math

from collections import defaultdict

app = Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'

myIp = "192.168.1.204"
cors = CORS(app, resources={r"/test": {"origins": myIp + ":5000"}}) #13.228.19.240
myWebUrl_ = "http://" + myIp + ":5000/"


# -----------------------------------------------------------------------------------------------------------------------

tf.reset_default_graph()

np.random.seed(42)
tf.set_random_seed(42)

classDict = {0 : 'ant', 1 : 'bat', 2 : 'bear', 3 : 'bee', 4 : 'bird', 5 : 'butterfly', 6 : 'camel', 7 : 'cat', 8 : 'cow', 9 : 'crab', 10 : 'crocodile', 11 : 'dog', 12 : 'dolphin', 13 : 'dragon', 14 : 'duck', 15 : 'elephant', 16 : 'fish', 17 : 'flamingo', 18 : 'frog', 19 : 'giraffe', 20 : 'hedgehog', 21 : 'horse', 22 : 'kangaroo', 23 : 'lion', 24 : 'lobster', 25 : 'mermaid', 26 : 'monkey', 27 : 'mosquito', 28 : 'mouse', 29 : 'octopus', 30 : 'owl', 31 : 'panda', 32 : 'penguin', 33 : 'pig', 34 : 'rabbit', 35 : 'raccoon', 36 : 'rhinoceros', 37 : 'scorpion', 38 : 'sea turtle', 39 : 'shark', 40 : 'sheep', 41 : 'snail', 42 : 'snake', 43 : 'spider', 44 : 'squirrel', 45 : 'swan', 46 : 'tiger', 47 : 'whale', 48 : 'zebra'}
classNum = len(classDict)
inputSize = 64

savePath_ = "./saveModel/"

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

        # ----------------------------------------------------------------------------------------------------------------------------------------START WEB

        @app.route('/test', methods=['GET', 'POST'])
        @cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
        def testWeb():
            if request.method == 'GET':
                return '''
                                <!DOCTYPE html>
                                <html>
                                    <head>
                                        <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
                                        <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.4/jquery.min.js"></script>
                                    </head>
                                    <body onload="init()">

                                        <center>
                                            <div>
                                                <h1>
                                                    Please insert 1D array of pixels (example "1, 0.5, 0.1, 1" for 2x2 image)
                                                    <br>
                                                    <input type="text" id="image" value="" size="60">
                                                    <button onclick="sendSentence();">send pixels</button>
                                                </h1>
                                            </div>
                                            <br>
                                            <div id="classOfImage">
                                                None
                                            </div>
                                        </center>
                                        <br>
                                        <div style="width: 80%; margin: auto;">
                                            <div>
                                                <b id="numPolite" style="float: left"></b>
                                                <b id="numObsence" style="float: right"></b>
                                            </div>
                                            <br>
                                            <div id="politenessList" style="float:left; width: 50%; background-color: lightblue;">
                                            </div>

                                            <div id="obscenityList" style="float:right; width: 50%; background-color: pink;">
                                            </div>
                                        </div>
										<br>
										<center>
										<canvas id="can" width="400" height="400" style="border:2px solid;"></canvas>
										<img id="canvasimg" style="display:none;">
										<br>
										<div style="border:2px solid;">
											<div>Pen</div>
											<div style="width:50px;height:50px;background:black;border:2px solid;" id="black" onclick="color(this)"></div>
											<div>Eraser</div>
											<div style="width:50px;height:50px;background:white;border:2px solid;" id="white" onclick="color(this)"></div>
										</div>
										
										<br>
										
										<div style="border:2px solid;">
										<input type="button" value="send image" id="btn" size="30" onclick="sendImage()" style="margin : 10px;">
										<input type="button" value="clear" id="clr" size="23" onclick="erase()" style="margin : 10px;">
										</div>
										</center>
										
                                        <script type="text/javascript">
                                            var lastestSentence = ""

                                            function sendSentence() {
                                                lastestSentence = document.getElementById("image").value
                                                $.ajax({
                                                    url: ''' + "'" + myWebUrl_ + 'test' + "'" + ''' ,
                                                    contentType: 'application/json;charset=UTF-8',
                                                    method: 'POST',
                                                    dataType: "html",
                                                    data: JSON.stringify({'image': lastestSentence}),
                                                    success: function (response) {
                                                        console.log(response);
                                                        document.getElementById("classOfImage").innerHTML = response
                                                    }
                                                })
                                            }
										</script>
									
									    <script type="text/javascript">
										
											var brushSize = 7;
										
											var canvas, ctx, flag = false,
												prevX = 0,
												currX = 0,
												prevY = 0,
												currY = 0,
												dot_flag = false;

											var x = "black",
												y = brushSize;
											
											function init() {
												canvas = document.getElementById('can');
												ctx = canvas.getContext("2d");
												w = canvas.width;
												h = canvas.height;
											
												canvas.addEventListener("mousemove", function (e) {
													findxy('move', e)
												}, false);
												canvas.addEventListener("mousedown", function (e) {
													findxy('down', e)
												}, false);
												canvas.addEventListener("mouseup", function (e) {
													findxy('up', e)
												}, false);
												canvas.addEventListener("mouseout", function (e) {
													findxy('out', e)
												}, false);

												canvas.addEventListener("touchstart", function (e) {
													findxy('down', e.touches[0])
												}, false);		

												canvas.addEventListener("touchend", function (e) {
													findxy('up', e.touches[0])
												}, false);

												canvas.addEventListener("touchmove", function (e) {
													findxy('move', e.touches[0])
												}, false);
											
												
												ctx.fillStyle = "white";
												ctx.fillRect(0, 0, w, h);
												ctx.fillStyle = "black";
											}
											
											function color(obj) {
												x = obj.id;
												if (x == "white") y = brushSize * 2;
												else y = brushSize;
											
											}
											
											function draw() {
												ctx.beginPath();
												ctx.moveTo(prevX, prevY);
												ctx.lineTo(currX, currY);
												ctx.strokeStyle = x;
												ctx.lineWidth = y;
												ctx.stroke();
												ctx.closePath();
											}
											
											function erase() {
												ctx.clearRect(0, 0, w, h);
												ctx.fillStyle = "white";
												ctx.fillRect(0, 0, w, h);
												ctx.fillStyle = "black";
												document.getElementById("canvasimg").style.display = "none";
											}
											
											function lerp(s, e, t){return s+(e-s)*t;}
											
											function blerp(c00,c10,c01,c11,tx,ty){
												return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
											}
											
											function sendImage() {
												
												var subImg = ctx.getImageData(0,0,400,400);

												var arraySize = 400*400*4;
												var onlyR = new Array(400*400);
												for (var i = 0; i < arraySize; i += 4) {
													onlyR[i/4] = subImg.data[i]/255.0;
												}
												
												var smallImg = new Array(64*64);
												
												var scale = 400.0/64.0;
												
												for (var i = 0; i < 64; i++) {
													for (var j = 0; j < 64; j++) {
														
														il = Math.floor(i * scale);
														jl = Math.floor(j * scale);
														
														sumVal = 0;
														
														for (var ii = 0; ii < 7; ii++) {
															for (var jj = 0; jj < 7; jj++) {
																dis = (ii - 3.0)*(ii - 3.0) + (jj - 3.0)*(jj - 3.0);
																posL = (il + ii) + (jl + jj) * 400;
															
																sumVal += onlyR[posL] * dis;
															}
														}
														
														sumVal /= 392.0;
														smallImg[i + j * 64] = sumVal;
													}
												}
												
												var tmpS = "";
												
												for (var i = 0; i < 4096; i++) {
													tmpS += smallImg[i].toString() + ",";
												}
												
												document.getElementById("image").value = tmpS;
												sendSentence();
											}
											
											function findxy(res, e) {
												if (res == 'down') {
													prevX = currX;
													prevY = currY;
													currX = e.clientX - canvas.offsetLeft;
													currY = e.clientY - canvas.offsetTop;
											
													flag = true;
													dot_flag = true;
													if (dot_flag) {
														ctx.beginPath();
														ctx.fillStyle = x;
														ctx.fillRect(currX, currY, brushSize, brushSize);
														ctx.closePath();
														dot_flag = false;
													}
												}
												if (res == 'up' || res == "out") {
													flag = false;
												}
												if (res == 'move') {
													if (flag) {
														prevX = currX;
														prevY = currY;
														currX = e.clientX - canvas.offsetLeft;
														currY = e.clientY - canvas.offsetTop;
														draw();
													}
												}
											}
										</script>
									</body>
								</html>                                                                                                       
                            '''
            elif request.method == 'POST':
                data = request.get_json()
                word = "Nothing"
                arr = []
                image = []
                if data == None:
                    return "receive None"
                if 'image' in data:
                    word = data['image']
                    try:
                        image = Image.open(BytesIO(base64.b64decode(str(word))))
                    except :
                        return "not base64"

                else :
                    return "data does not contain 'image' field"

                inputShape = ""
                try:
                    inputShape = str(np.shape(image))
                except:
                    return "can't get image shape"

                if inputShape != "(320, 320)":
                    return "image should be (320,320) , but got " + inputShape

                img = (np.array(image)) / 255.0
                img = np.reshape(img, [320 * 320])

                smallImg = np.zeros([64 * 64])

                scale = 320.0 / 64.0

                for i in range(0, 64, 1):
                    for j in range(0, 64, 1):
                        il = math.floor(i * scale)
                        jl = math.floor(j * scale)

                        sumVal = 0

                        for ii in range(0, 5, 1):
                            for jj in range(0, 5, 1):
                                dis = (ii - 2.0) * (ii - 2.0) + (jj - 2.0) * (jj - 2.0)
                                posL = (il + ii) + (jl + jj) * 320

                                sumVal += img[posL] * dis

                        sumVal /= 100.0
                        sumVal = 1.0 - sumVal
                        smallImg[i + j * 64] = sumVal * sumVal

                arr = smallImg

                arrShape = str(np.shape(arr))
                if arrShape == "(4096,)":
                    yProb = y_proba.eval(feed_dict={x_: [arr], boxes_: [[0, 0, 1, 1]], boxes_id_: [0], rpow_: 1})
                    yProb = np.array(np.reshape(yProb[0][0],[49]))
                    argMax5 = yProb.argsort()[-5:][::-1]

                    returnData = []
                    for e in argMax5:
                        returnData.append(classDict[e])


                    #print (yProb)
                    #plt.imshow(np.reshape(arr,[64,64]))
                    #plt.show()
                    #rint (yProb.argsort()[-5:][::-1])

                    return jsonify(results=returnData)
                else:
                    return "wrong format : should be (4096,) but got " + arrShape

        # ----------------------------------------------------------------------------------------------------------------------------------------END WEB
        app.run(host=myIp)

model()
