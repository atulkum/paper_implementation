import os
import sys
import numpy
import scipy.io

import gzip
import tarfile
import h5py
from PIL import Image
import six.moves.cPickle as pickle
from six.moves import urllib
import shutil

def load_data(data_dir):
    def check_dataset(dataset):
        data_path = os.path.join(data_dir, dataset)
        if (not os.path.isfile(data_path)):
            origin = (
                'http://ufldl.stanford.edu/housenumbers/' + dataset
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, data_path)
        return data_path

    train_dataset = check_dataset('train.tar.gz')
    test_dataset = check_dataset('test.tar.gz')

    def format_data(dataset):
        data_path = os.path.join(data_dir, dataset)
        tar = tarfile.open(data_path, 'r:gz')

        data_file_split = os.path.splitext(dataset)[0]
        data_type = os.path.splitext(data_file_split)[0]

        def check_file(folder_name):
            new_path = os.path.join(data_dir, folder_name)
            if (not os.path.exists(new_path)):
                tar.extractall(data_dir)
                process_data()

        def process_data():
            print '... processing data (should only occur when downloading for the first time)'
            # Access label information in digitStruct.mat
            new_path = os.path.join(data_dir, data_type, 'digitStruct.mat')
            f = h5py.File(new_path, 'r')

            digitStructName = f['digitStruct']['name']
            digitStructBbox = f['digitStruct']['bbox']

            def getName(n):
                return ''.join([chr(c[0]) for c in f[digitStructName[n][0]].value])

            def bboxHelper(attr):
                if (len(attr) > 1):
                    attr = [f[attr.value[j].item()].value[0][0] for j in range(len(attr))]
                else:
                    attr = [attr.value[0][0]]
                return attr

            def getBbox(n):
                bbox = {}
                bb = digitStructBbox[n].item()
                # bbox = bboxHelper(f[bb]["label"])
                bbox['height'] = bboxHelper(f[bb]["height"])
                bbox['label'] = bboxHelper(f[bb]["label"])
                bbox['left'] = bboxHelper(f[bb]["left"])
                bbox['top'] = bboxHelper(f[bb]["top"])
                bbox['width'] = bboxHelper(f[bb]["width"])
                return bbox

            def getDigitStructure(n):
                s = getBbox(n)
                s['name'] = getName(n)
                return s

            # Process labels
            print '... creating image box bound dict for %s data' % data_type
            image_dict = {}
            for i in range(len(digitStructName)):
                image_dict[getName(i)] = getBbox(i)
                if (i%1000 == 0):
                    print '     image dict processing: %i/%i complete' %(i,len(digitStructName))
            print '... dict processing complete'

            # Process the data
            print('... processing image data and labels')

            names = []
            for item in os.listdir(os.path.join(data_dir, data_type)):
                if item.endswith('.png'):
                    names.append(item)

            y = []
            x = []
            for i in range(len(names)):
                path = os.path.join(data_dir, data_type)
                y.append(image_dict[names[i]]['label'])
                image = Image.open(path + '/' + names[i]).convert('L')
                left = int(min(image_dict[names[i]]['left']))
                upper = int(min(image_dict[names[i]]['top']))
                right = int(max(image_dict[names[i]]['left'])) + int(max(image_dict[names[i]]['width']))
                lower = int(max(image_dict[names[i]]['top'])) + int(max(image_dict[names[i]]['height']))
                image = image.crop(box = (left, upper, right, lower))
                image = image.resize([32,32])
                image_array = numpy.array(image)
                x.append(image_array)
                if (i%1000 == 0):
                    print '     image processing: %i/%i complete' %(i,len(names))
            print '... image processing complete'

            # Save data
            print '... pickling data'
            out = {}
            out['names'] = names
            out['labels'] = y
            out['images'] = x
            output_file = data_type + 'pkl.gz'
            out_path = os.path.join(data_dir, output_file)
            p = gzip.open(out_path, 'wb')
            pickle.dump(out, p)
            p.close()

            tar.close()
            # clean up (delete test/train folders that were used to create the pickled data)
            shutil.rmtree(os.path.join(data_dir, data_type))

        check_file(data_type)

    # This check will run everytime load_data() is called

    if (not os.path.isfile(os.path.join(data_dir, 'trainpkl.gz'))):
        format_data('train.tar.gz')

    f_train = gzip.open(os.path.join(data_dir, 'trainpkl.gz'), 'rb')
    train_set = pickle.load(f_train)
    f_train.close()

    if (not os.path.isfile(os.path.join(data_dir, 'testpkl.gz'))):
        format_data('test.tar.gz')

    f_test = gzip.open(os.path.join(data_dir, 'testpkl.gz'), 'rb')
    test_set = pickle.load(f_test)
    f_test.close()

    # Convert data format
    def convert_data_format(data):
        data['X'] = data.pop('images')
        data['X'] = numpy.array(data['X'])
        data['X'] = numpy.rollaxis(data['X'],0, data['X'].ndim)
        data['y'] = data.pop('labels')

        X = numpy.reshape(data['X'],
                          (numpy.prod(data['X'].shape[:-1]), data['X'].shape[-1]),
                          order='C').T / 255.

        def process_sequence(labels):
            for i in range(len(labels)):
                l = len(labels[i])-1
                labels[i].insert(0,l)
                zeros = numpy.zeros(6-l-1).tolist()
                labels[i].extend(zeros)
            return numpy.array(labels)

        y = process_sequence(data['y'])
        return (X,y)

    train_set = convert_data_format(train_set)
    test_set = convert_data_format(test_set)

    train_set_len = len(train_set[1])

    # Extract validation dataset from train dataset (10% of the train_set)
    valid_set = [x[-(train_set_len//10):] for x in train_set]
    train_set = [x[:-(train_set_len//10)] for x in train_set]

    # train_set, valid_set, test_set each contain a list [flattened image, sequence].
    # The 'flattened image' part of the list is a 2D numpy array where each row
    # corresponds to a 32x32x3 image. The sequence is a 2D numpy array of the
    # number represented in the image. The first element in the sequence is the
    # length of the number (where 0 = a 1 digit number), the second element in
    # the sequence is the first digit of the number (where 0 means no digit
    # present and 10 = 0), and so on.

    return [train_set, valid_set, test_set]

if __name__ == '__main__':
    load_data(sys.argv[1])
