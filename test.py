
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

relative_L2_test = []

for i in range(true_test.shape[0]):
  relative_L2_test.append(np.linalg.norm(true_test[i,:,:]-predict_test[i,:,:,0])/np.linalg.norm(true_test[i,:,:]))

relative_Linfini_test = []

for i in range(true_test.shape[0]):
  relative_Linfini_test.append(np.max(np.abs((true_test[i,:,:]-predict_test[i,:,:,0])))/np.max(np.abs((true_test[i,:,:]))))

print('average L2 error',np.mean(relative_L2_test))
print('std L2 error',np.std(relative_L2_test))

print('average Linfini error',np.mean(relative_Linfini_test))
print('std Linfini error',np.std(relative_Linfini_test))

plt.plot(relative_L2_test)
plt.show()
plt.close()
plt.plot(relative_Linfini_test)
plt.show()
plt.close()
plt.imshow(predict_test[1000,:,:])
plt.show()
plt.close()


ssim_list = []

from skimage.metrics import structural_similarity

for i in range(true_test.shape[0]):
  ssim_score = structural_similarity(true_test[i,:,:], predict_test[i,:,:].reshape(171,171))
  ssim_list.append(ssim_score)
  
plt.plot(ssim_list)

print('average SSIM',np.mean(ssim_list))
print('std SSIM',np.std(ssim_list))


norm_list = []

for i in range(true_test.shape[0]):
  norm_list.append(np.linalg.norm(true_test[i,:,:]))

np.mean(norm_list)
index = [i for i,v in enumerate(norm_list) if v > 0.9*np.max(norm_list)]


relative_error_large09 = []

for i in index:
  relative_error_large09.append(np.linalg.norm(true_test[i,:,:]-predict_test[i,:,:,0])/np.linalg.norm(true_test[i,:,:]))

relative_Linfini_large09 = []

for i in index:
  relative_Linfini_large09.append(np.max(np.abs((true_test[i,:,:]-predict_test[i,:,:,0])))/np.max(np.abs((true_test[i,:,:]))))
  
plt.plot(relative_error_large09)
plt.plot(relative_Linfini_large09)
