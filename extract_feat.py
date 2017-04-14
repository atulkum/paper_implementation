'''
keras.json

{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
'''


from keras.preprocessing import image, sequence
from keras.utils.np_utils import to_categorical
import bcolz
import sys
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

def get_data(filepath, is_test=False, target_size=(224,224)):
    gen=image.ImageDataGenerator()
    batches = gen.flow_from_directory(filepath, target_size=target_size,
            class_mode=None, shuffle=False, batch_size=1)
    data = np.concatenate([batches.next() for i in range(batches.nb_sample)])
    if is_test:
        return (data, batches.filenames)
    else:
        return (data, to_categorical(batches.classes))

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()
    
def load_array(fname):
    return bcolz.open(fname)[:]

def save_data(path):
    trn, trn_labels = get_data(path+'/train')
    save_array(path+'/results/trn.dat', trn)
    np.save(path+'/results/trn_labels.npy', trn_labels)

    val, val_labels = get_data(path+'/valid')
    save_array(path+'/results/val.dat', val)
    np.save(path+'/results/val_labels.npy', val_labels)

    #test, test_filenames = get_data(path+'/test', True)
    #save_array(path+'/results/test.dat', test)
    #np.save(path+'/results/test_filenames.npy', test_filenames)


def ext_feat(path):
    trn = load_array(path+'/results/trn.dat')
    val = load_array(path+'/results/val.dat')
    #test = load_array(path+'/results/test.dat')
    trn = preprocess_input(trn)
    val = preprocess_input(val)
    #test = preprocess_input(test)

    base_model = VGG16(weights='imagenet', include_top=False)
    conv_model = Model(input=base_model.input, output=base_model.get_layer('block5_conv3').output)

    conv_feat = conv_model.predict(trn)
    save_array(path+'/results/conv_feat.dat', conv_feat)
    conv_val_feat = conv_model.predict(val)
    save_array(path+'/results/conv_val_feat.dat', conv_val_feat)
    #conv_test_feat = conv_model.predict(test)
    #save_array(path+'/results/conv_test_feat.dat', conv_test_feat)

if __name__ == "__main__":
    path = sys.argv[1]
    save_data(path)
    ext_feat(path)

