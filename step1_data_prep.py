
import matplotlib.pyplot as plt
import numpy as np
import os
import tarfile
import urllib
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
import cPickle as pickle
import gzip
import pylab

url = 'http://yann.lecun.com/exdb/mnist/'

def maybe_download(filename):
  if not os.path.exists(filename):
    filename, _ = urllib.urlretrieve(url + filename, filename)
 
  return filename

train_data_filename = maybe_download('../data/MNIST_DATA/train-images-idx3-ubyte.gz')
train_labels_filename = maybe_download('../data/MNIST_DATA/train-labels-idx1-ubyte.gz')
test_data_filename = maybe_download('../data/MNIST_DATA/t10k-images-idx3-ubyte.gz')
test_labels_filename = maybe_download('../data/MNIST_DATA/t10k-labels-idx1-ubyte.gz')



def get_one_hot(raw_labels, num_col):
    labels = tf.expand_dims(raw_labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, num_col]), 1.0, 0.0)
    return onehot_labels

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
NUM_CHAR = 5

def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].
  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels


train_data = extract_data(train_data_filename, 60000)
train_labels = extract_labels(train_labels_filename, 60000)
test_data = extract_data(test_data_filename, 10000)
test_labels = extract_labels(test_labels_filename, 10000)


blank = np.random.rand(28,28) - 0.5

"""
nums = np.random.choice(train_data.shape[0], size=5, replace=True)
pylab.axis('off'); 
cont_img = list()
for i in range(5):
    cont_img.append(train_data[nums[i]].reshape(28,28))
    
pylab.imshow(np.hstack(cont_img))

pylab.show()
"""

np.random.seed(42)
cont_train_data = np.ndarray(shape=(60000, IMAGE_SIZE, IMAGE_SIZE*NUM_CHAR), dtype=np.float32)
cont_train_labels = np.ndarray(shape=(60000, NUM_CHAR), dtype=np.int32)
    
for i in range(60000):
    num_blanck = np.random.randint(0, 3)
    nums = np.random.choice(60000, size=(5-num_blanck))
    cont_img = list()
    cont_label = list()
    
    val = 0
    for j in range(5-num_blanck):
        cont_img.append(train_data[nums[j]].reshape(28,28))
        cont_label.append(train_labels[nums[j]])
        #val = val*10 + int(train_labels[nums[j]])
    
    #cont_label = [5-num_blanck, val]
    
    for k in range(num_blanck):
        #if np.random.rand() < 0.5:
        #    cont_img = cont_img + [blank]
        #    cont_label = cont_label + [-1]
        #else:
        #    cont_img = [blank] + cont_img
        #    cont_label = [-1] + cont_label 
        
        cont_img = cont_img + [blank]
        cont_label = cont_label + [-1]
        
    cont_train_labels[i] = cont_label
    cont_train_data[i, :, :]  = np.hstack(cont_img)
    #print cont_train_labels[i]
    #pylab.imshow(cont_train_data[i])
    #pylab.show()
    
pickle_file = '../data/MNIST_DATA/cont_train.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'X': cont_train_data[10000:],
    'y': cont_train_labels[10000:]
  }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print 'Unable to save data to', pickle_file, ':', e
  raise
    
pickle_file = '../data/MNIST_DATA/cont_valid.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'X': cont_train_data[:10000],
    'y': cont_train_labels[:10000]
  }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print 'Unable to save data to', pickle_file, ':', e
  raise

del train_data
del train_labels
import gc
gc.collect()

del cont_train_data
del cont_train_labels
import gc
gc.collect()


np.random.seed(42)
cont_test_data = np.ndarray(shape=(10000, IMAGE_SIZE, IMAGE_SIZE*NUM_CHAR), dtype=np.float32)
cont_test_labels = np.ndarray(shape=(10000, NUM_CHAR), dtype=np.int32)
    
for i in range(10000):
    num_blanck = np.random.randint(0, 3)
    nums = np.random.choice(10000, size=(5-num_blanck))
    cont_img = list()
    cont_label = list()
    val = 0
    for j in range(5-num_blanck):
        cont_img.append(test_data[nums[j]].reshape(28,28))
        cont_label.append(test_labels[nums[j]])
        #val = val*10 + int(test_labels[nums[j]])
        
    #cont_label = [5-num_blanck, val]
    
    for k in range(num_blanck):
        #if np.random.rand() < 0.5:
        #    cont_img = cont_img + [blank]
        #    cont_label = cont_label + [-1]
        #else:
        #    cont_img = [blank] + cont_img
        #    cont_label = [-1] + cont_label 
        
        cont_img = cont_img + [blank]
        cont_label = cont_label + [-1]

    cont_test_labels[i] = cont_label
    cont_test_data[i, :, :]  = np.hstack(cont_img)
    #print cont_train_labels[i]
    #pylab.imshow(cont_train_data[i])
    #pylab.show()
    
pickle_file = '../data/MNIST_DATA/cont_test.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'X': cont_test_data,
    'y': cont_test_labels
  }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print 'Unable to save data to', pickle_file, ':', e
  raise




