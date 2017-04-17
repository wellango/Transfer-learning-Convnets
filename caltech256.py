import os
import numpy as np
# For GPU
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
from keras import backend as K
K.set_session(session)

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img
import keras
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Dense
import matplotlib.image as mtplt
# import matplotlib.pyplot as plt

# @param output_dim - the number of classes
# return - compiled model
def getModel( output_dim ):
    vgg_model = VGG16( weights='imagenet', include_top=True )
    vgg_out = vgg_model.layers[-2].output # Last FC layer's output
    softmax_layer = Dense(output_dim, activation='softmax')(vgg_out) #Create softmax layer taking input as vgg_out
    #Create new transfer learning model
    tl_model = Model( input=vgg_model.input, output=softmax_layer )
    for i in range(len(tl_model.layers)):
        tl_model.layers[i].trainable = False
    tl_model.layers[i].trainable = True
    tl_model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['acc'])
    return tl_model

# @param img_path - path to the image
# return - preprocessed image (vgg16's pre-processeing)
def preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    if len(img.shape) != 3:     # If greyscale, convert to color
        x_new = np.zeros((img.shape[0], img.shape[1], 3))
        x_new[:, :, 0] = img
        x_new[:, :, 1] = img
        x_new[:, :, 2] = img
        img = x_new
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    return x

# @param samples - list of lists containining image file names
# @param n - number of images to be read
# return - tensor pair X, y (shuffled)
def make_tensor(samples,n):
    X_list = []
    y_list = []
    for i in range(num_classes):
        samples_lst = samples[i]
        OHclassi = to_categorical(np.array([i]), num_classes)
        if n!=-1:
            idx = np.random.permutation(len(samples_lst))
            samples_lst = [samples_lst[ix] for ix in idx[:n]]

        for j in range(len(samples_lst)):
            if samples_lst[j].lower().endswith(('.jpg', 'jpeg', 'png')):
                x = preprocess(folders_lst[i]+'/'+samples_lst[j])
            else:
                continue
            x = x.astype(float)

            X_list.append(x[0])
            y_list.append(OHclassi)

    X = np.array(X_list)
    y = np.array(y_list)[:,0,:]
    idx = list(range(X.shape[0]))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    return X, y

np.random.seed(42)

# Path to dataset
pathToDataset = '256_ObjectCategories/'
folders_lst = [x[0] for x in os.walk(pathToDataset)]
folders_lst.sort()
folders_lst = folders_lst[1:]

# Appropriately sized network output
num_classes = len(folders_lst)
output_dim = num_classes

# Number of validation and test samples
numVal = 4
numTest = 25

# Use the remaining images for training
test_samples =[]
train_samples =[]
validation_samples =[]
for i in range(num_classes):
    samples_lst = [e[2] for e in os.walk(folders_lst[i])][0]
    np.random.shuffle(samples_lst)
    validation_samples.append(samples_lst[:numVal])
    test_samples.append(samples_lst[numVal:numVal+numTest])
    train_samples.append(samples_lst[numVal+numTest:])

Xtest, ytest = make_tensor(test_samples,-1)
Xval, yval = make_tensor(validation_samples,-1)

# File for logging results
test_fname = open('test_log.txt', 'w')
training_size = [2,4,8,16,32] # Number of training examples used per class
for t in training_size:
    tl_model = getModel( output_dim )
    Xtrain, ytrain = make_tensor(train_samples,t)

    log_name = 'training_' + str(t) + '.log'
    csv_logger = keras.callbacks.CSVLogger(log_name, separator=',', append=False)
    historyEpoch = tl_model.fit(Xtrain, ytrain, nb_epoch=15, batch_size=128, callbacks=[csv_logger], validation_data=(Xval,yval))
    l = tl_model.evaluate(Xtest,ytest)
    print l
    test_fname.write("%s"%t)
    for item in l:
        test_fname.write(";%s" % item)
    test_fname.write("\n")
    print 'FINISHED t=', t
    # pandas.DataFrame(history).to_csv('training_batch_' + t +'.log')

test_fname.close()