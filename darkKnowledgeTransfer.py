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
# @temperature - temperature parameter for the final softmax
# return - compiled model
def getModel( output_dim, temperature ):
    vgg_model = VGG16( weights='imagenet', include_top=True )
    vgg_out = vgg_model.layers[-2].output # Last FC layer's output
    vgg_model.layers[-1].set_weights([vgg_model.layers[-1].get_weights()[0], vgg_model.layers[-1].get_weights()[1]/temperature])
    vgg_softmax = vgg_model.layers[-1]
    dark_layer = Lambda(lambda x: x/temperature)(vgg_out)
    softmax1_layer = vgg_softmax(dark_layer)
    tl_model = Model( input=vgg_model.input, output=softmax1_layer)
    for i in range(len(tl_model.layers)):
        tl_model.layers[i].trainable = False
    tl_model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['acc'])
    tl_model.summary()
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

# File for logging results
test_fname = open('test_logDark_optimised.txt', 'w')

output_dim = 256
training_size = [2] # Number of training examples used per class

temperatureT = 2
vgg_model = getModel( output_dim, temperatureT)
Xtest , ytest = make_tensor(test_samples,-1)
Xval , yval = make_tensor(validation_samples,-1)

Xpval = vgg_model.predict(Xval)
Xptest = vgg_model.predict(Xtest)

for t in training_size:
    Xtrain ,ytrain = make_tensor(train_samples,t)
    Xptrain = vgg_model.predict(Xtrain)
    tl_model = Sequential()
    tl_model.add(Dense(num_classes,input_dim=1000, activation='softmax'))
    for i in range(len(tl_model.layers)):
        tl_model.layers[i].trainable = True

    tl_model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['acc'])
    tl_model.summary()

    log_name = 'training_' + str(temperatureT) + 'Dark.log'
    csv_logger = keras.callbacks.CSVLogger(log_name, separator=',', append=False)
    historyEpoch = tl_model.fit(Xptrain, ytrain, nb_epoch=500, batch_size=128, callbacks=[csv_logger], validation_data=(Xpval,yval))
    l = tl_model.evaluate(Xptest,ytest)

    test_fname.write("%s"%temperatureT)
    for item in l:
        test_fname.write(";%s" % item)
    test_fname.write("\n")
    print 'FINISHED t =', t
test_fname.close()
