import numpy as np
import cv2
import glob
import h5py
import matplotlib.pyplot as plt
import random
import os
import math
import re
from lxml import etree
import uuid
from collections import defaultdict
import json
import PIL
from PIL import Image


def dot_h5_make_hydrophonic_img_set(dot_h_path, img_path):
    # dot_h_path: save location for the .h file
    # img_dir_path: images path for build the dataset

    # read addresses and labels from the 'train' folder
    addrs = glob.glob(img_path)
    labels = [
        0 if 'bellpepper' in addr else 1 if 'bokchoy' in addr else 2 if 'gherkins' in addr else 3 if 'greenbeans' in addr else 4 if 'lettuce' in addr else 5
        for addr in
        addrs]  # 0 = bellpepper, 1 = bokchoy, 2 = gherkins, 3 = greenbeans, 4 = lettuce, 5 = tomato

    train_addrs = addrs[0:int(1 * len(addrs))]
    train_y = labels[0:int(1 * len(labels))]

    # crate h5 file with tables
    # check the order of data and chose proper data shape to save images
    train_shape = (len(train_addrs), 227, 227, 3)
    # open a hdf5 file and create earrays
    hdf5_file = h5py.File(dot_h_path, mode='w')
    hdf5_file.create_dataset("train_x", train_shape, np.uint8)
    hdf5_file.create_dataset("train_mean", train_shape[1:], np.uint8)
    hdf5_file.create_dataset("train_y", (len(train_addrs),), np.uint8)
    hdf5_file["train_y"][...] = train_y

    # Load images and save them
    # a numpy array to save the mean of the images
    mean = np.zeros(train_shape[1:], np.float32)
    # loop over train addresses
    for i in range(len(train_addrs)):
        # print how many images are saved every 1000 images
        if i % 1000 == 0 and i > 1:
            print
            'Train data: {}/{}'.format(i, len(train_addrs))
        # read an image and resize to (227, 227)
        # cv2 load images as BGR, convert it to RGB
        addr = train_addrs[i]
        img = cv2.imread(addr)
        img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # add any image pre-processing here
        # if the data order is Theano, axis orders should change
        """if data_order == 'th':
            img = np.rollaxis(img, 2)"""
        # save the image and calculate the mean so far
        hdf5_file["train_x"][i, ...] = img[None]
        mean += img / float(len(train_y))

    # save the mean and close the hdf5 file
    hdf5_file["train_mean"][...] = mean
    hdf5_file.close()
    print("dataset created successfully!")


def load_dataset():
    # dataset = h5py.File(dot_h_path, 'r')


    # from utils
    train_dataset = h5py.File('/home/erandaka/My Studies/Sem8/Prj/data_set/train_images/train_dataset.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_x"][
                                :])  # your test set features                                                                                                                                            train_set_x_orig = np.array(train_dataset["train_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_y"][:])  # your train set labels

    test_dataset = h5py.File('/home/erandaka/My Studies/Sem8/Prj/data_set/test_images/test_dataset.h5', "r")
    test_set_x_orig = np.array(test_dataset["train_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["train_y"][:])  # your test set label   s

    # classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig
    # end
    # train_x = dataset["train_x"]
    # shape_x=train_x.shape
    # train_y = dataset["train_y"]
    # guss=random.randrange(0,1400)
    # train_lbl = train_y[guss]
    # shape_y=train_lbl.shape
    # img = train_x[guss]
    # if train_lbl == int(0):
    #     print("bellpepper")
    # elif train_lbl == int(1):
    #     print("bokchoy")
    # elif train_lbl == int(2):
    #     print("gherkins")
    # elif train_lbl == int(3):
    #     print("greenbeans")
    # elif train_lbl == int(4):
    #     print("lettuce")
    # elif train_lbl == int(5):
    #     print("tomato")
    #
    # plt.imshow(img)
    # plt.show()

    # return dataset


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def random_mini_batches(X, Y, mini_batch_size=64):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.rand(len(mini_batches))

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# create frames from a video file

def frame_split(vid_path, save_path):  # video path and the destination for save frames
    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        cv2.imwrite(save_path + "frame%d.jpg" % count, image)  # save frame as JPEG file
        count += 1


# online plant dataset preparation
def images_separate_to_id_xml(dir_path):
    xml_files = []
    imag_dict = defaultdict(list)
    file_id_list = []
    for file in os.listdir(dir_path):
        if re.search(".+xml", file):
            xml_files.append(file)
        else:
            continue
    for xml_file in xml_files:
        a_file = etree.parse(str(dir_path) + str(xml_file))
        id = a_file.find("ClassId")
        file_id_list.append({"id": id.text, "file": xml_file})
    for item in file_id_list:
        imag_dict[item["id"]].append(item["file"])

    save_dict = {"save_dict": imag_dict}
    with open('/home/erandaka/My Studies/Sem8/Prj/data_set/file.txt', 'w') as file:
        file.write(json.dumps(save_dict))
    return imag_dict


def online_plant_dataset_train_renaming(src_dir_path, des_dir_path):
    xml_files = []
    for file in os.listdir(src_dir_path):
        if re.search(".+xml", file):
            xml_files.append(file)
        else:
            continue
    for xml_file in xml_files:
        a_file = etree.parse(str(src_dir_path) + str(xml_file))
        id = a_file.find("ClassId")

        jpg_file_prv = str(xml_file).split(".xml")[0] + ".jpg"

        if (id.text=="3465" or
            id.text=="1325" or
            id.text=="30131" or
            id.text=="3124" or
            id.text=="3750" or
            id.text=="30249" or
            id.text=="1973" or
            id.text=="4838" or
            id.text=="1842" or
            id.text=="3956" or
            id.text=="4721" or
            id.text=="1329" or
            id.text=="30122" or
            id.text=="4109" or
            id.text=="326" or
            id.text=="329" or
            id.text=="13623" or
            id.text=="14872" or
            id.text=="7305" or
            id.text=="3958" or
            id.text=="2648" or
            id.text=="30040" or
            id.text=="2689" or
            id.text=="2918" or
            id.text=="29924" or
            id.text=="5128" or
            id.text=="3279" or
            id.text=="6367" or
            id.text=="5537" or
            id.text=="55" or
            id.text=="8523" or
            id.text=="30056" or
            id.text=="4379" or
            id.text=="5634" or
            id.text=="2631" or
            id.text=="3962" or
            id.text=="1321" or
            id.text=="8522" or
            id.text=="4074" or
            id.text=="30126" or
            id.text=="3849" or
            id.text=="3942" or
            id.text=="54" or
            id.text=="1328" or
            id.text=="3846"):
            jpg_file_aft = id.text + "-" + uuid.uuid4().hex[:6].upper() + ".jpg"
            os.rename(os.path.join(src_dir_path, jpg_file_prv), os.path.join(des_dir_path, jpg_file_aft))

326
def dot_h5_make_onlineplant_img_set(dot_h_path, img_path):
    # dot_h_path: save location for the .h file
    # img_dir_path: images path for build the dataset

    # read addresses and labels from the 'train' folder
    addrs = glob.glob(img_path)
    labels = [
        0 if '3465' in addr
        else 1 if '1325' in addr
        else 2 if '30131' in addr
        else 3 if '3124' in addr
        else 4 if '3750' in addr
        else 5 if '30249' in addr
        else 6 if '1973' in addr
        else 7 if '4838' in addr
        else 8 if '1842' in addr
        else 9 if '3956' in addr
        else 10 if '4721' in addr
        else 11 if '1329' in addr
        else 12 if '30122' in addr
        else 13 if '4109' in addr
        else 14 if '326' in addr
        else 15 if '329' in addr
        else 16 if '13623' in addr
        else 17 if '14872' in addr
        else 18 if '7305' in addr
        else 19 if '3958' in addr
        else 20 if '2648' in addr
        else 21 if '30040' in addr
        else 22 if '2689' in addr
        else 23 if '2918' in addr
        else 24 if '29924' in addr
        else 25 if '5128' in addr
        else 26 if '3279' in addr
        else 27 if '6367' in addr
        else 28 if '5537' in addr
        else 29 if '55' in addr
        else 30 if '8523' in addr
        else 31 if '30056' in addr
        else 32 if '4379' in addr
        else 33 if '5634' in addr
        else 34 if '2631' in addr
        else 35 if '3962' in addr
        else 36 if '1321' in addr
        else 37 if '8522' in addr
        else 38 if '4074' in addr
        else 39 if '30126' in addr
        else 40 if '3849' in addr
        else 41 if '3942' in addr
        else 42 if '54' in addr
        else 43 if '1328' in addr
        else 44
        for addr in
        addrs]

    train_addrs = addrs[0:int(1 * len(addrs))]
    train_y = labels[0:int(1 * len(labels))]

    # crate h5 file with tables
    # check the order of data and chose proper data shape to save images
    train_shape = (len(train_addrs), 227, 227, 3)
    # open a hdf5 file and create earrays
    hdf5_file = h5py.File(dot_h_path, mode='w')
    hdf5_file.create_dataset("train_x", train_shape, np.uint8)
    hdf5_file.create_dataset("train_mean", train_shape[1:], np.uint8)
    hdf5_file.create_dataset("train_y", (len(train_addrs),), np.uint8)
    hdf5_file["train_y"][...] = train_y

    # Load images and save them
    # a numpy array to save the mean of the images
    mean = np.zeros(train_shape[1:], np.float32)
    # loop over train addresses
    for i in range(len(train_addrs)):
        # print how many images are saved every 1000 images
        if i % 1000 == 0 and i > 1:
            print
            'Train data: {}/{}'.format(i, len(train_addrs))
        # read an image and resize to (227, 227)
        # cv2 load images as BGR, convert it to RGB
        addr = train_addrs[i]
        img = cv2.imread(addr)

        img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # add any image pre-processing here
        # if the data order is Theano, axis orders should change
        """if data_order == 'th':
            img = np.rollaxis(img, 2)"""
        # save the image and calculate the mean so far
        hdf5_file["train_x"][i, ...] = img[None]
        mean += img / float(len(train_y))

    # save the mean and close the hdf5 file
    hdf5_file["train_mean"][...] = mean
    hdf5_file.close()
    print("dataset created successfully!")


def print_file_list(file_path):
    img_dict = images_separate_to_id_xml(file_path)
    for key, val in img_dict.items():
        print(key + ": " + str(len(val)))

# resize garden image set
def resize_image(images_path,save_path):
    file_list = os.listdir(images_path)
    base_width= 912
    for i in range(0,len(file_list)):
        img = Image.open(images_path+file_list[i])
        wpercent = (base_width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((base_width, hsize), PIL.Image.ANTIALIAS)
        img.save(save_path+file_list[i])
    print("images resize completed succeffully!")



# print_file_list("/home/erandaka/My Studies/Sem8/Prj/data_set/train_online_plantdset_original/")

# online_plant_dataset_train_renaming("/home/erandaka/My Studies/Sem8/Prj/data_set/train_online_plantdset_original/")

# dot_h5_make_online_plant_train_img_set("/home/erandaka/My Studies/Sem8/Prj/data_set/train_sample_online_dset/train.h5","/home/erandaka/My Studies/Sem8/Prj/data_set/train_sample_online_dset/")
# online_plant_dataset_train("/home/erandaka/My Studies/Sem8/Prj/data_set/train_sample_online_dset/")

# load_dataset('/home/erandaka/My Studies/Sem8/Prj/Data_Set/dataset_hdf5/plant/dataset.h5')

#dot_h5_make_onlineplant_img_set("/home/erandaka/My Studies/Sem8/Prj/data_set/dataset_hdf5/test_online_plantdataset.h5","/home/erandaka/My Studies/Sem8/Prj/data_set/test_online_plantdataset_images_only/*.jpg")

#online_plant_dataset_train_renaming("/home/erandaka/My Studies/Sem8/Prj/data_set/train_online_plantdataset_original/","/home/erandaka/My Studies/Sem8/Prj/data_set/train_online_plantdataset_images_only/")

resize_image("/home/erandaka/My Studies/Sem8/Prj/data_set/Plants/Lettuce/","/home/erandaka/My Studies/Sem8/Prj/data_set/resized_images/lettuce/")