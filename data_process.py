import numpy as np
import bcolz

path = "data/"

class DataIterator():
    def __init__(self, X, y):
        # in_feature_shape = (512, 14, 14)
        self.X = X
        self.y = y
        self.size = len(self.X)
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        idx = np.random.shuffle(np.arange(self.size))
        self.X = np.squeeze(self.X[idx])
        self.y = np.squeeze(self.y[idx])
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor+n-1 > self.size:
            self.epochs += 1
            self.shuffle()
        batch_X = self.X[self.cursor:self.cursor+n] # (512, 14, 14)
        batch_y = self.y[self.cursor:self.cursor+n] 
        self.cursor += n
        return (batch_X, batch_y)

def get_train_data():
    conv_feat = bcolz.open(path+'results/conv_feat.dat')[:]
    conv_feat = np.swapaxes(conv_feat,1,3)
    feat_shape = conv_feat.shape
    conv_feat = conv_feat.reshape(feat_shape[0], feat_shape[1], -1) 
    trn_labels = np.load(path+'results/trn_labels.npy')
    print('X => ', conv_feat.shape, 'y => ', trn_labels.shape)
    return (conv_feat, trn_labels)

def get_val_data():
    conv_val_feat = bcolz.open(path+'results/conv_val_feat.dat')[:]
    conv_val_feat = np.swapaxes(conv_val_feat,1,3)
    feat_shape = conv_val_feat.shape
    conv_val_feat=conv_val_feat.reshape(feat_shape[0], feat_shape[1], -1) 
    val_labels = np.load(path+'results/val_labels.npy')
    print('X => ', conv_val_feat.shape, 'y => ', val_labels.shape)
    return (conv_val_feat, val_labels)

def get_test_data():
    conv_test_feat = bcolz.open(path+'results/conv_test_feat.dat')[:]
    conv_test_feat = np.swapaxes(conv_test_feat,1,3)
    feat_shape = conv_test_feat.shape
    conv_test_feat=conv_test_feat.reshape(feat_shape[0], feat_shape[1], -1) 
    test_filenames = np.load(path+'results/test_filenames.npy')
    raw_test_filenames = [f.split('/')[-1] for f in test_filenames]
    
    print('X => ', conv_test_feat.shape)
    return (conv_test_feat, raw_test_filenames)

