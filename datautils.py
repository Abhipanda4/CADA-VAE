import torch
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable

import numpy as np
import random
import scipy.io as scio
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

        res101_data = scio.loadmat('./datasets/%s/res101.mat' % dset)
        self.features = self.normalize(res101_data['features'].T)
        self.labels = res101_data['labels'].reshape(-1)

        self.attribute_dict = scio.loadmat('./datasets/%s/att_splits.mat' % dset)
        self.attributes = self.attribute_dict['att'].T

        # file with all class names for deciding train/test split
        self.class_names_file = './datasets/%s/classes.txt' % dset

        # test class names
        with open('./datasets/%s/testclasses.txt' % dset) as fp:
            self.test_class_names = [i.strip() for i in fp.readlines()]

        if self.synthetic:
            assert syn_dataset is not None
            self.syn_dataset = syn_dataset
        else:
            self.dataset = self.create_orig_dataset()
            if self.train:
                self.gzsl_dataset = self.create_gzsl_dataset()

    def normalize(self, matrix):
        scaler = MinMaxScaler()
        return scaler.fit_transform(matrix)

    def get_classmap(self):
        '''
        Returns the classmap of all classes to be used as test set
        as described in proposed split
        '''
        with open(self.class_names_file) as fp:
            all_classes = fp.readlines()

        test_count = 0
        train_count = 0

        train_classmap = dict()
        test_classmap = dict()
        for line in all_classes:
            idx, name = [i.strip() for i in line.split(' ')]
            if name in self.test_class_names:
                if self.gzsl:
                    # train classes are also included in test time
                    test_classmap[int(idx)] = self.n_train + test_count
                else:
                    test_classmap[int(idx)] = test_count
                test_count += 1
            else:
                train_classmap[int(idx)] = train_count
                train_count += 1

        return train_classmap, test_classmap

    def create_gzsl_dataset(self, n_samples=200):
        dataset = []
        for key, features in self.gzsl_map.items():
            n_points = min(n_samples, len(features))
            for f in random.sample(features, n_points):
                dataset.append((f, -1, key))
        return dataset

    def create_orig_dataset(self):
        '''
        Partitions all image features into train/test based on proposed split
        Returns 2 lists, train_set & test_set: each entry of list is a 3-tuple
        (feature, label_in_dataset, label_for_classification)
        '''
        self.train_classmap, self.test_classmap = self.get_classmap()

        if self.train:
            labels = self.attribute_dict['trainval_loc'].reshape(-1)
            classmap = self.train_classmap
            self.gzsl_map = dict()
        else:
            labels = self.attribute_dict['test_unseen_loc'].reshape(-1)
            if self.gzsl:
                labels = np.concatenate((labels, self.attribute_dict['test_seen_loc'].reshape(-1)))
                classmap = {**self.train_classmap, **self.test_classmap}
            else:
                classmap = self.test_classmap


        dataset = []
        for l in labels:
            idx = self.labels[l - 1]
            dataset.append((self.features[l - 1], idx, classmap[idx]))

            if self.train:
                # create a map bw class label and features
                try:
                    self.gzsl_map[classmap[idx]].append(self.features[l - 1])
                except Exception as e:
                    self.gzsl_map[classmap[idx]] = [self.features[l - 1]]

        return dataset

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
