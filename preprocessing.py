

import numpy as np
from google.colab import drive
drive.mount('/content/drive')
import matplotlib.pyplot as plt
import random
from scipy.interpolate import griddata

field_data = np.loadtxt('drive/MyDrive/VCNN_test/data/helin/apower1_large.txt')

field_data = field_data.reshape(9261,171,171)


random_range = 3

def stucture_obs(x_list,y_list):
  x_coord = []
  y_coord = []
  for i in range(len(x_list)-1):
    for j in range(len(y_list)-1):
      #x_coord.append(random.randint(x_list[i],x_list[i+1]))
      #y_coord.append(random.randint(y_list[j],y_list[j+1]))
      try:
        x_coord.append(random.randint(x_list[i],x_list[i]+random_range))
        y_coord.append(random.randint(y_list[j],y_list[j]+random_range))
      except:
        pass
  return x_coord,y_coord


x_list = [0,10,30,50,70,90,110,130,150,170]
y_list = [0,10,30,50,70,90,110,130,150,170]


x_coord,y_coord = stucture_obs(x_list,y_list)


Ni = 100
N = 171
count = 0

x_coord,y_coord = stucture_obs(x_list,y_list)

datasize = 9261
repeat_sample = 2

vor_field_data = np.zeros((datasize*repeat_sample, 171, 171))
true_field_data = np.zeros((datasize*repeat_sample, 171, 171))

#for index in range(9261):

for index in range(datasize):
  for repeat in range(repeat_sample):

    Xi, Yi = np.array(x_coord), np.array(y_coord)
    Pi = np.zeros((len(x_coord),2))
    Pi[:,0] = Xi
    Pi[:,1] = Yi
    u_t = field_data[index,:,:,]
    Zi = u_t[Xi.astype(int),N-1-Yi.astype(int)]

    x = np.linspace(0., N, N)
    y = np.linspace(0., N, N)[::-1]
    X, Y = np.meshgrid(x, y)
    P = np.array([X.flatten(), Y.flatten() ]).transpose()

    Z_nearest = griddata(Pi, Zi, P, method = "nearest").reshape([N, N])

    vor_field_data[count,:,:] = Z_nearest
    true_field_data[count,:,:] = u_t
    count += 1


np.save('drive/MyDrive/VCNN_test/data/helin/vor_field_data_apower1_2.npy',vor_field_data)
np.save('drive/MyDrive/VCNN_test/data/helin/true_field_data_apower1_2.npy',true_field_data)

