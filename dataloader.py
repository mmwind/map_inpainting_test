import h5py
import cv2
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

datafile = 'dtec_2_10_2015_173_90_-90_N_-180_180_E_a160.h5'
#datafile = 'dtec_2_10_2015_173_90_-90_N_-180_180_E_5700.h5'

f = h5py.File(datafile, 'r')

query_id = list(f['metadata'])[0]
dates = list(f['data'].keys())

dt = np.dtype('float,float,float')
rawdata = list(f['data'][dates[0]][query_id])
target = np.zeros((len(rawdata), 3))

for i in range(len(rawdata)):
    target[i] = np.asarray(list(rawdata[i]))

target = pd.DataFrame(target, columns=['lat', 'lon', 'tec'])
target['lon'] = (target['lon'] - target['lon'].min()) / (target['lon'].max() - target['lon'].min())
target['lat'] = (target['lat'] - target['lat'].min()) / (target['lat'].max() - target['lat'].min())

cvmap = np.zeros((200, 400, 2), dtype="float")

for i in range(len(target)-1):
    lon = int(np.round(target['lon'][i] * cvmap.shape[1])) - 1
    lat = int(np.round(target['lat'][i] * cvmap.shape[0])) - 1
    cvmap[lat, lon, 0] += target['tec'][i] # Add value to accumulator
    cvmap[lat, lon, 1] += 1 # Increment value counter

cvout = np.divide(cvmap[:, :, 0], cvmap[:, :, 1])
not_nan_values = cvout[~np.isnan(cvout)].flatten()
cvout = (cvout - not_nan_values.min()) / (not_nan_values.max() - not_nan_values.min())

vis_map = 255 - np.uint8(255*cvout)
vis_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
plt.imshow(vis_map, origin="lower")

# Show values distribution
plt.hist(not_nan_values, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()
