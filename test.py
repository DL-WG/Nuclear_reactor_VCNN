
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
#from sympy import Symbol, nsolve, solve
#from sympy.solvers import solve
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import animation as animation

from PIL import Image
import matplotlib as mpl
import time

from tensorflow.python.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D,Cropping2D, AveragePooling2D,Dense,Flatten,Reshape,Dropout
from tensorflow.python.keras.models import Model,Sequential
import tensorflow as tf

import tensorflow as tf

from tensorflow.python.keras.layers import LSTM,LeakyReLU,RepeatVector
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint

from tensorflow import keras
import pickle
import keras
from keras.layers import LeakyReLU
from keras import backend as K
from keras import optimizers



datasize = 9261
repeat_sample = 2

test_index = list(range(1,datasize*repeat_sample,10))
train_index = list(set(range(1,datasize*repeat_sample)) - set(range(1,datasize*repeat_sample,10)))

vor_data = np.load('drive/MyDrive/VCNN_test/data/helin/vor_field_data_apower1_2.npy')
true_data = np.load('drive/MyDrive/VCNN_test/data/helin/true_field_data_apower1_2.npy')

vor_train = vor_data[train_index,:,:]
vor_test = vor_data[test_index,:,:]

del vor_data

true_train = true_data[train_index,:,:]
true_test = true_data[test_index,:,:]

del true_data


model =  keras.models.load_model('drive/MyDrive/VCNN_test/model/nuclear_VCNN_9261_fix_apower1_large.h5')

predict_test = model.predict(vor_test)

np.save('drive/MyDrive/VCNN_test/data/helin/test_results.npy',predict_test)

plt.imshow(predict_test[1000,:,:])

