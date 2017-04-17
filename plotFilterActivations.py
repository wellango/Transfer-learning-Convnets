import numpy as np
import os
import keras
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
from skimage import exposure
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import matplotlib.image as mtplt

# @param output_dim - the number of classes
# return - compiled model
def getModel( output_dim ):
    vgg_model = VGG16( weights='imagenet', include_top=True )
    vgg_out = vgg_model.layers[1].output # Last FC layer's output
    tl_model = Model( input=vgg_model.input, output=vgg_out )
    for i in range(len(tl_model.layers)):
        tl_model.layers[i].trainable = False
    tl_model.layers[i].trainable = True
    tl_model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['acc'])
    return tl_model

# @param img_path - path to the image
# return - preprocessed image (vgg16's pre-processeing)
def preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    plt.imshow(img)
    plt.show()
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

# Load the network with appropriately sized output
num_classes = len(folders_lst)
output_dim = num_classes
tl_model = getModel( output_dim )

# Load a random image to visualize filter activations
x = preprocess('256_ObjectCategories/236.unicorn/236_0026.jpg')
x_pred = tl_model.predict(x)

plt.figure()
plt.imshow(x[0,:,:,0])
plt.show()

# Create a 2 by 2 plot where each plot is a 4 by 4 subplot
n = 4
k = 16 # Change for every plot
plt.figure(figsize=(100, 100))
for i in range(n**2):
    plt.subplot(n, n, i+k)
    plt.imshow(x_pred[0,:,:,i+0])
plt.show()