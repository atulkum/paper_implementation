import numpy as np
import os
import cPickle as pickle
import scipy.io as sio
import h5py
import sys

prefix = sys.argv[1]
pickle_file = '../data/SVHN/%s_metadata.pickle'%(prefix)

with open(pickle_file, 'rb') as pickleData:
	metadata = pickle.load(pickleData)


def get_full_bbox(md):
	mx = {}
	mi = {}

	for i, l in enumerate(md['label']):
		if 'x' in mi:
			mi['x'] = min(mi['x'], md['left'][i])
			mi['y'] = min(mi['y'], md['top'][i])

			mx['x'] = max(mx['x'], md['left'][i] + md['width'][i])
			mx['y'] = max(mx['y'], md['top'][i] + md['height'][i])
        else:
		mi['x'] = md['left'][i]
		mi['y'] = md['top'][i]

		mx['x'] = md['left'][i] + md['width'][i]
		mx['y'] = md['top'][i] + md['height'][i]


	bbox = {'height': (mx['y'] - mi['y']), 'width' : (mx['x'] - mi ['x']) , 'top': mi['y'], 'left':mi['x']}

	return bbox


md =  metadata['13.png']
print get_full_bbox(md)
