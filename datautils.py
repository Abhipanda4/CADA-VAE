import torch
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable

import numpy as np
from sklearn.preprocessing import MinMaxScaler

class ZSLDataset(Dataset):
    def __init__(self, dset, n_train, n_test, train=True, gzsl=False, synthetic=False, syn_dataset=None):
        '''TODO: Docstring here

        Args:
            dset (TODO): TODO
            n_train (TODO): TODO
            n_test (TODO): TODO

        Kwargs:
            train (TODO): TODO
            gzsl (TODO): TODO
            synthetic (TODO): TODO
            syn_dataset (TODO): TODO

        '''
        super(ZSLDataset, self).__init__()
        self.dset = dset
        self.n_train = n_train
        self.n_test = n_test
        self.train = train
        self.gzsl = gzsl
        self.synthetic = synthetic

        self.split = 0.9

        # a np array of size (n_samples, 2048)
        features = np.load('./datasets/%s/features.npy' % dset)
        self.features = self.normalize(features)

        # a np array of size (n_samples,)
        self.labels = np.load('./datasets/%s/labels.npy' % dset)

        # a np array of size (n_classes, n_attributes)
        attributes = np.load('./datasets/%s/attributes.npy' % dset)
        self.attributes = self.normalize(attributes)

        # file with all class names for deciding train/test split
        self.class_names = './datasets/%s/classes.txt' % dset

        if dset == 'awa2':
            self.test_classes = [
                'sheep','dolphin','bat','seal','blue+whale',
                'rat','horse','walrus','giraffe','bobcat'
            ]
        else:
            raise NotImplementedError

        if self.synthetic:
            assert syn_dataset is not None
            self.syn_dataset = syn_dataset
        else:
            self.train_dataset, self.test_dataset, self.gzsl_dataset = self.create_orig_dataset()
            if train:
                self.dataset = self.train_dataset
            else:
                self.dataset = self.test_dataset

    def normalize(self, matrix):
        scaler = MinMaxScaler()
        return scaler.fit_transform(matrix)

    def get_label_maps(self):
        '''
        Returns the labels of all classes to be used as test set
        as described in proposed split
        '''
        with open(self.class_names) as fp:
            all_classes = fp.readlines()

        test_count = 0
        train_count = 0

        train_labels = dict()
        test_labels = dict()
        for line in all_classes:
            idx, name = [i.strip() for i in line.split(' ')]
            if name in self.test_classes:
                if self.gzsl:
                    # train classes are also included in test time
                    test_labels[int(idx)] = self.n_train + test_count
                else:
                    test_labels[int(idx)] = test_count
                test_count += 1
            else:
                train_labels[int(idx)] = train_count
                train_count += 1

        return train_labels, test_labels

    def create_orig_dataset(self, n_samples=200):
        '''
        Partitions all 37322 image features into train/test based on proposed split
        Returns 2 lists, train_set & test_set: each entry of list is a 3-tuple
        (feature, label_in_dataset, label_for_classification)
        '''
        self.train_labels, self.test_labels = self.get_label_maps()
        train_map, test_map = {}, {}
        train_set, test_set  = [], []
        partial_train_set = []

        for feat, label in zip(self.features, self.labels):
            if label in self.test_labels.keys():
                try:
                    test_map[label].append((feat, label, self.test_labels[label]))
                except:
                    test_map[label] = [(feat, label, self.test_labels[label])]
            else:
                try:
                    train_map[label].append((feat, label, self.train_labels[label]))
                except:
                    train_map[label] = [(feat, label, self.train_labels[label])]

        for label in train_map.keys():
            if self.gzsl:
                # In case of GZSL, some data from seen classes is also tested
                cutoff = int(self.split * len(train_map[label]))
                train_set.extend(train_map[label][:cutoff])
                test_set.extend(train_map[label][cutoff:])
            else:
                train_set.extend(train_map[label])
            partial_train_set.extend(train_map[label][:n_samples])

        for label in test_map.keys():
            test_set.extend(test_map[label])

        return train_set, test_set, partial_train_set

    def __getitem__(self, index):
        if self.synthetic:
            # choose an example from synthetic dataset
            img_feature, orig_label, label_idx = self.syn_dataset[index]
        else:
            # choose an example from original dataset
            img_feature, orig_label, label_idx = self.dataset[index]

        label_attr = self.attributes[orig_label - 1]
        return img_feature, label_attr, label_idx

    def __len__(self):
        if self.synthetic:
            return len(self.syn_dataset)
        else:
            return len(self.dataset)
