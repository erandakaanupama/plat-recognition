## Python program to classify an image
##  based on the model of exercise "Convolution Model Application"
##  Deep Learning specialization "Convolutional Neural Networks" - Week 1
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

from scipy import ndimage
import tensorflow as tf

# make predictions for video input
def vid_frameconvert(vid_path, images_path):  # video path and the destination for save frames
    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        cv2.imwrite(images_path + "frame%d.jpg" % count, image)  # save frame as JPEG file
        count += 1
    rem_img=images_path + "frame%d.jpg" % (count-1)
    os.remove(rem_img)

## Path of the checkpoint files
checkpoint_path = '/home/erandaka/PycharmProjects/disease_yieldrec/plant_rec/modelparameters/hydroponic_garden/'

## Path and name of the image to be classified
image_path = "/home/erandaka/PycharmProjects/disease_yieldrec/datasets/"
# image_name = "thumbs_up.jpg"

# image_name = "one.jpg"
# image_name = "two.jpg"
# image_name = "three.jpg"
# image_name = "number04.jpg"
# image_name = "five.jpg"



## Read and pre-process the image
def predict(image_path):
    imges_list=os.listdir(image_path)
    y=imges_list[0]
    for image_name in imges_list[0:len(imges_list)-2]:
        image = np.array(ndimage.imread(image_path + image_name, flatten=False))
        my_image = scipy.misc.imresize(image, size=(64, 64), mode='RGB')
        my_image = my_image / 255.
        my_image_work = np.expand_dims(my_image, 0)




        ## Predict the classification of the loaded image

        ## I am using the default data flow graph to run predictions
        tf.reset_default_graph()

        with tf.Session() as sess:
            # Show image (optional visual check)
            # print("Using a picture of shape", my_image_work.shape, "for the prediction")
            plt.imshow(my_image_work[0])
            plt.show(block=False)
            plt.pause(1)

            ## Load the entire model previuosly saved in a checkpoint
            print("Load the model from path", checkpoint_path)
            the_Saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
            the_Saver.restore(sess, checkpoint_path)

            ## Identify the predictor of the Tensorflow graph
            predict_op = tf.get_collection('predict_op')[0]

            ## Identify the restored Tensorflow graph
            dataFlowGraph = tf.get_default_graph()

            ## Identify the input placeholder to feed the images into as defined in the model
            x = dataFlowGraph.get_tensor_by_name("X:0")

            ## Predict the image category
            prediction = sess.run(predict_op, feed_dict={x: my_image_work})

            print("\nThe predicted image class is:", str(np.squeeze(prediction)))



# vid to frame conversion
vid_frameconvert("/home/erandaka/Music/english/Alan Walker/Alan Walker - Alone - YouTube.MP4","/home/erandaka/PycharmProjects/disease_yieldrec/plant_rec/image_frames/frames_aw/")
# predictions
# predict("/home/erandaka/PycharmProjects/disease_yieldrec/plant_rec/image_frames/")
