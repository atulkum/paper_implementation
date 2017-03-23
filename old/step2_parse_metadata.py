import numpy as np
import os
import cPickle as pickle
import scipy.io as sio
import h5py
import sys

prefix = sys.argv[1]
filename = '../data/SVHN/%s/digitStruct.mat'%(prefix)

def extractValue(obj):
	obj_len = len(obj)
	if obj_len == 1:
    		return [obj.value[0][0]]
	else:
    		return [f[obj.value[j].item()].value[0][0] for j in range(obj_len)]


f = h5py.File(filename)
digitStructName = f['digitStruct']['name']
digitStructBbox = f['digitStruct']['bbox']

metadata= {}
for i in range(len(digitStructName)):
        bbox = {}
        bb = digitStructBbox[i].item()
        bbox['height'] = extractValue(f[bb]["height"])
        bbox['label'] = extractValue(f[bb]["label"])
        bbox['left'] = extractValue(f[bb]["left"])
        bbox['top'] = extractValue(f[bb]["top"])
        bbox['width'] = extractValue(f[bb]["width"])
	name=''.join([chr(c[0]) for c in f[digitStructName[i][0]].value])
 	metadata[name] = bbox


 
pickle_file = '../data/SVHN/%s_metadata.pickle'%(prefix)

try:
  pickleData = open(pickle_file, 'wb')
  pickle.dump(metadata, pickleData, pickle.HIGHEST_PROTOCOL)
  pickleData.close()
except Exception as e:
  print 'Unable to save data to', pickle_file, ':', e
  raise



