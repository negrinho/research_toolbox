import numpy as np
import os
import cPickle
import research_toolbox.tb_augmentation as tb_au
from tensorflow.examples.tutorials.mnist import input_data

def load_mnist(data_folderpath, flatten=False, one_hot=True, normalize_range=False,
        border_pad_size=0):
    mnist = input_data.read_data_sets(data_folderpath, one_hot=one_hot, reshape=flatten)

    def _extract_fn(x):
        X = x.images
        y = x.labels
        if not normalize_range:
            X *= 255.0
        return (X, y)

    Xtrain, ytrain = _extract_fn(mnist.train)
    Xval, yval = _extract_fn(mnist.validation)
    Xtest, ytest = _extract_fn(mnist.test)

    if border_pad_size > 0:
        Xtrain = tb_au.zero_pad_border(Xtrain, border_pad_size)
        Xval = tb_au.zero_pad_border(Xval, border_pad_size)
        Xtest = tb_au.zero_pad_border(Xtest, border_pad_size)

    return (Xtrain, ytrain, Xval, yval, Xtest, ytest)

def _load_cifar_datafile(filepath, labels_key, num_classes, normalize_range, flatten, one_hot):
    with open(filepath, 'rb') as f:
        d = cPickle.load(f)

        # for the data
        X = d['data'].astype('float32')

        # reshape the data to the format (num_images, height, width, depth)
        num_images = X.shape[0]
        X = X.reshape((num_images, 3, 32, 32))
        X = X.transpose((0,2,3,1))
        X = X.astype('float32')

        # transformations based on the argument options.
        if normalize_range:
            X = X / 255.0
        if flatten:
            X = X.reshape((num_images, -1))

        # for the labels
        y = np.array(d[labels_key])

        if one_hot:
            y_one_hot = np.zeros((num_images, num_classes), dtype='float32')
            y_one_hot[np.arange(num_images),  y] = 1.0
            y = y_one_hot

        return (X, y)

def load_cifar10(data_folderpath, num_val, flatten=False, one_hot=True,
        normalize_range=False, whiten_pixels=True, border_pad_size=0):
    """Loads all of CIFAR-10 in a numpy array.

    Provides a few options for the output formats. For example,
    normalize_range returns the output images with pixel values in [0.0, 1.0].
    The other options are self explanatory. Border padding corresponds to
    upsampling the image by zero padding the border of the image.

    """
    train_filenames = ['data_batch_1', 'data_batch_2', 'data_batch_3',
        'data_batch_4', 'data_batch_5']
    test_filenames = ['test_batch']

    # NOTE: uses some arguments from the outer scope.
    def _load_data_multiple_files(fname_list):

        X_parts = []
        y_parts = []
        for fname in fname_list:
            filepath = os.path.join(data_folderpath, fname)
            X, y = _load_cifar_datafile(filepath, 'labels', 10, normalize_range, flatten, one_hot)
            X_parts.append(X)
            y_parts.append(y)

        X_full = np.concatenate(X_parts, axis=0)
        y_full = np.concatenate(y_parts, axis=0)

        return (X_full, y_full)

    Xtrain, ytrain = _load_data_multiple_files(train_filenames)
    Xtest, ytest = _load_data_multiple_files(test_filenames)
    Xval, yval = Xtrain[- num_val:], ytrain[- num_val:]

    if whiten_pixels:
        mean = Xtrain.mean(axis=0)[None, :]
        std = Xtrain.std(axis=0)[None, :]
        Xtrain = (Xtrain - mean) / std
        Xval = (Xval - mean) / std
        Xtest = (Xtest - mean) / std

    # NOTE: the zero padding is done after the potential whitening
    if border_pad_size > 0:
        Xtrain = tb_au.zero_pad_border(Xtrain, border_pad_size)
        Xval = tb_au.zero_pad_border(Xval, border_pad_size)
        Xtest = tb_au.zero_pad_border(Xtest, border_pad_size)

    return (Xtrain, ytrain, Xval, yval, Xtest, ytest)

def load_cifar100(data_folderpath, num_val, flatten=False, one_hot=True, normalize_range=False):
    Xtrain, ytrain = _load_cifar_datafile(os.path.join(data_folderpath, 'train'),
        'fine_labels', 100, normalize_range, flatten, one_hot)
    Xtest, ytest = _load_cifar_datafile(os.path.join(data_folderpath, 'test'),
        'fine_labels', 100, normalize_range, flatten, one_hot)

    Xval, yval = Xtrain[-num_val:], ytrain[-num_val:]
    return (Xtrain, ytrain, Xval, yval, Xtest, ytest)

def fake_labels(num_elems, num_classes):
    return np.random.randint(num_classes, size=(num_elems,))

def fake_sequences(num_sequences, vocab_size, sequence_length):
    return np.random.randint(vocab_size, size=(num_sequences, sequence_length))

def fake_vectors(num_vecs, dim):
    return np.random.normal(size=(num_vecs, dim))

def fake_images(num_images, height, width, num_channels):
    return np.random.normal(size=(num_images, height, width, num_channels))

def fake_videos(num_videos, num_frames, height, width, num_channels):
    return np.random.normal(size=(num_videos, num_frames, height, width, num_channels))

def fake_tensors(num_tensors, shape):
    return np.random.normal(size=(num_tensors, ) + shape)