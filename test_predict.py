## Python program to classify an image
##  based on the model of exercise "Convolution Model Application"
##  Deep Learning specialization "Convolutional Neural Networks" - Week 1

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import tensorflow as tf

## Path of the checkpoint files
checkpoint_path = '/home/erandaka/PycharmProjects/disease_yieldrec/plant_rec/modelparameters/hydroponic_garden'

## Path and name of the image to be classified
image_path = "/home/erandaka/PycharmProjects/disease_yieldrec/datasets/"
# image_name = "thumbs_up.jpg"

# image_name = "one.jpg"
# image_name = "two.jpg"
# image_name = "three.jpg"
image_name = "6.jpg"
# image_name = "five.jpg"

## Read and pre-process the image
image = np.array(ndimage.imread(image_path + image_name, flatten=False))
my_image = scipy.misc.imresize(image, size=(64, 64), mode='RGB')
my_image = my_image / 255.
my_image_work = np.expand_dims(my_image, 0)

# ## Show image (optional visual check)
# print("Using a picture of shape", my_image_work.shape, "for the prediction")
# plt.imshow(my_image_work[0])
# plt.show()

## Predict the classification of the loaded image

## I am using the default data flow graph to run predictions
tf.reset_default_graph()

with tf.Session() as sess:
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


