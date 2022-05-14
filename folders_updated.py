# function: Dataset of DIBR image databases
# usage: pls refer to "__main__" below
# author: Jiebin Yan
# email: jiebinyan@foxmail.com
# v1.0.0

import torch.utils.data as data
import torch
import os
import random
import os.path
import scipy.io
import numpy as np
import pandas as pd
from torchvision import transforms
from tools import pilLoader
import warnings



def generateBTFVVTrainTest(iter_num=20, ratio=0.8):
    """
        we follow the traditional test methodology: 80% (default ratio) for training, and 20% for testing.
        we repeat this process 20 times, and the medium value of the results of 20 experiments is considered as the final result.
    """

    for idx in range(iter_num):

        train_data_saving_name = "train_iter_" + str(idx) + ".npy"
        test_data_saving_name = "test_iter_" + str(idx) + ".npy"

        data_mos_path = r"./Overall_Mos"
        url_mos = np.load(os.path.join(data_mos_path, "moku_fvv_oss_url_overall.npy"))

        data_num = len(url_mos)
        train_num = int(data_num * ratio)
        test_num = data_num - train_num
        print("totall samples:{}, train samples:{}, test samples:{}".format(data_num, train_num, test_num))

        num_list = [idx for idx in range(data_num)]  # already shuffled
        random.shuffle(num_list)
        train_list = num_list[0:train_num]
        test_list = num_list[train_num:data_num]

        train_url = url_mos[:, 0][train_list]
        train_mos = url_mos[:, 1][train_list]
        test_url = url_mos[:, 0][test_list]
        test_mos = url_mos[:, 1][test_list]

        train_data = np.array([train_url, train_mos])
        test_data = np.array([test_url, test_mos])

        np.save(os.path.join(data_mos_path, train_data_saving_name), train_data)
        np.save(os.path.join(data_mos_path, test_data_saving_name), test_data)


class BTFVVFolder(data.Dataset):
    """
    Paper:
        FVV QoE
    Note:
        totally, 1944 FVVs and the associated quality scores
        we split the database into training set (1555) and testing test (389)
    """
    def __init__(self, url_mos_dir, vector_dir, iter_num, frame_num=16, train_test=True, transform=None):
        """

        Args:
            url_mos_dir: dir where the FVV names and the associated mos are saved
            vector_dir: dir where the feature vectors are saved
            iter_num: 0~20, denotes the iteration number
            frame_num: The number of consecutive frames
            train_test: True -> train, False -> test
            transform: -> tensor
        """

        self._url_mos_dir = url_mos_dir  # "./First_Mos"
        self._vector_dir = vector_dir   # "./VSFA_Features"
        self._frame_num = frame_num  # extract extract number of frames
        self._train_test = train_test

        im_name = []
        im_score = []

        if self._train_test:
            name = "train_iter_"
        else:
            name = "test_iter_"

        file_name = name + str(iter_num) + ".npy"
        data = np.load(os.path.join(url_mos_dir, file_name))
        data_num = data.shape[1]

        self._length = np.zeros((data_num, 1))

        for idx in range(data_num):
            im_name.append(data[0][idx])
            im_score.append(float(data[1][idx]))
            self._length[idx] = self._frame_num

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

        self._samples = im_name
        self._scores = im_score

        print('im_name length: {}'.format(len(self._samples)))
        print('im_score length: {}'.format(len(self._scores)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """


        if self._train_test:  ## train

            length = self._length[index]

            name, score = self._samples[index], self._scores[index]
            name = name.split(".")[0] + ".npy"
            path = os.path.join(self._vector_dir, name)
            sample = np.load(path)
            sample = np.squeeze(sample)

            # # a sampling strategy
            frame_number = sample.shape[0]
            start_index = random.randint(0, frame_number - self._frame_num)
            end_index = start_index + self._frame_num
            sample = sample[start_index:end_index, :]

            sample = self.transform(sample)
            sample = torch.squeeze(sample)

        else:   ## test: we need rewrite

            name, score = self._samples[index], self._scores[index]
            name = name.split(".")[0] + ".npy"
            path = os.path.join(self._vector_dir, name)
            sample = np.load(path)
            sample = np.squeeze(sample)
            frame_number = sample.shape[0]
            vector_dim = sample.shape[1]

            pseudo_batch_size = int(frame_number/self._frame_num)
            more_frame = frame_number % self._frame_num
            if more_frame != 0:
                pseudo_batch_size = pseudo_batch_size + 1
            test_data = np.zeros((pseudo_batch_size, self._frame_num, vector_dim))

            for idx in range(pseudo_batch_size-1):
                start_index = idx * self._frame_num
                end_index = (idx+1) * self._frame_num
                test_data[idx, :, :] = sample[start_index:end_index, :]

            test_data[pseudo_batch_size-1, :, :] = sample[frame_number-self._frame_num:frame_number, :]
            test_data = test_data.transpose((1, 2, 0))

            sample = self.transform(test_data)
            sample = torch.squeeze(sample)

            length = np.zeros((pseudo_batch_size, 1))
            for idx in range(pseudo_batch_size):
                length[idx] = self._frame_num

        return sample, score, length

    def sample_strategy(self, frame_list):
        pass

    def __len__(self):
        length = len(self._samples)
        return length


class BTFVVFolderInterval(data.Dataset):
    """
    Paper:
        FVV QoE
    Note:
        totally, 1944 FVVs and the associated quality scores
        we split the database into training set (1555) and testing test (389)
        we test sparsely sampling
    """
    def __init__(self, url_mos_dir, vector_dir, iter_num, frame_num=16, interval_num=1, train_test=True, transform=None):
        """

        Args:
            url_mos_dir: dir where the FVV names and the associated mos are saved
            vector_dir: dir where the feature vectors are saved
            iter_num: 0~20, denotes the iteration number
            frame_num: The number of consecutive frames
            train_test: True -> train, False -> test
            transform: -> tensor
        """

        self._url_mos_dir = url_mos_dir  # "./First_Mos"
        self._vector_dir = vector_dir   # "./VSFA_Features"
        self._frame_num = frame_num  # extract extract number of frames
        self._train_test = train_test
        self._interval_num = interval_num  # newly added parameter

        im_name = []
        im_score = []

        if self._train_test:
            name = "train_iter_"
        else:
            name = "test_iter_"

        file_name = name + str(iter_num) + ".npy"
        data = np.load(os.path.join(url_mos_dir, file_name))
        data_num = data.shape[1]

        self._length = np.zeros((data_num, 1))

        for idx in range(data_num):
            im_name.append(data[0][idx])
            im_score.append(float(data[1][idx]))
            self._length[idx] = self._frame_num

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

        self._samples = im_name
        self._scores = im_score

        print('im_name length: {}'.format(len(self._samples)))
        print('im_score length: {}'.format(len(self._scores)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        if self._train_test:  ## train

            length = self._length[index]

            name, score = self._samples[index], self._scores[index]
            name = name.split(".")[0] + ".npy"
            path = os.path.join(self._vector_dir, name)
            sample = np.load(path)
            sample = np.squeeze(sample)

            # # sparsely sampling strategy
            frame_number = sample.shape[0]  # video frame number
            frame_cross = self._frame_num + (self._frame_num - 1) * (self._interval_num - 1)  # plus all frames

            if frame_cross >= frame_number:
                raise AttributeError('--the sparsely sampling strategy parameters should be double-checked!!!')
            else:

                start_index = random.randint(0, frame_number - frame_cross)
                end_index = start_index + frame_cross
                sample = sample[start_index:end_index:self._interval_num, :]
                sample = self.transform(sample)
                sample = torch.squeeze(sample)
                print('good luck')

        else:   ## test

            name, score = self._samples[index], self._scores[index]
            name = name.split(".")[0] + ".npy"
            path = os.path.join(self._vector_dir, name)
            sample = np.load(path)
            sample = np.squeeze(sample)
            frame_number = sample.shape[0]
            vector_dim = sample.shape[1]

            frame_cross = self._frame_num + (self._frame_num - 1) * (self._interval_num - 1)  # plus all frames
            if frame_cross >= frame_number:
                raise AttributeError('--the sparsely sampling strategy parameters should be double-checke!!!')
            else:
                pass

            pseudo_batch_size = int(frame_number/frame_cross)
            more_frame = frame_number % frame_cross

            if more_frame != 0:
                pseudo_batch_size = pseudo_batch_size + 1
            test_data = np.zeros((pseudo_batch_size, self._frame_num, vector_dim))


            for idx in range(pseudo_batch_size-1):
                start_index = idx * frame_cross
                end_index = (idx+1) * frame_cross
                test_data[idx, :, :] = sample[start_index:end_index:self._interval_num, :]

            test_data[pseudo_batch_size-1, :, :] = sample[frame_number-frame_cross:frame_number:self._interval_num, :]
            test_data = test_data.transpose((1, 2, 0))

            sample = self.transform(test_data)
            sample = torch.squeeze(sample)

            length = np.zeros((pseudo_batch_size, 1))
            for idx in range(pseudo_batch_size):
                length[idx] = self._frame_num

        return sample, score, length

    def sample_strategy(self, frame_list):
        pass

    def __len__(self):
        length = len(self._samples)
        return length


class BTFVVFolderIntervalTSN(data.Dataset):
    """
    Paper:
        FVV QoE
    Note:
        totally, 1944 FVVs and the associated quality scores
        we split the database into training set (1555) and testing test (389)
        we test the sampling strategy proposed in TSN
    """
    def __init__(self, url_mos_dir, vector_dir, iter_num, frame_num=16, interval_num=1, train_test=True, transform=None):
        """

        Args:
            url_mos_dir: dir where the FVV names and the associated mos are saved
            vector_dir: dir where the feature vectors are saved
            iter_num: 0~20, denotes the iteration number
            frame_num: The number of consecutive frames
            train_test: True -> train, False -> test
            transform: -> tensor
        """

        self._url_mos_dir = url_mos_dir  # "./First_Mos"
        self._vector_dir = vector_dir   # "./VSFA_Features"
        self._frame_num = frame_num  # extract number of frames
        self._train_test = train_test
        # self._interval_num = interval_num  # we do not use this parameter

        im_name = []
        im_score = []

        if self._train_test:
            name = "train_iter_"
        else:
            name = "test_iter_"

        file_name = name + str(iter_num) + ".npy"
        data = np.load(os.path.join(url_mos_dir, file_name))
        data_num = data.shape[1]  # # the number of videos

        self._length = np.zeros((data_num, 1))

        for idx in range(data_num):
            im_name.append(data[0][idx])
            im_score.append(float(data[1][idx]))
            self._length[idx] = self._frame_num

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

        self._samples = im_name
        self._scores = im_score

        print('im_name length: {}'.format(len(self._samples)))
        print('im_score length: {}'.format(len(self._scores)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        if self._train_test:  ## train

            length = self._length[index]

            name, score = self._samples[index], self._scores[index]
            name = name.split(".")[0] + ".npy"
            path = os.path.join(self._vector_dir, name)
            sample = np.load(path)
            sample = np.squeeze(sample)

            # # sparsely sampling strategy
            frame_number = sample.shape[0]  # video frame number
            segments = frame_number//self._frame_num  # the number of segments

            frame_index = []
            if segments == 0:
                raise AttributeError('--the sparsely sampling strategy parameters should be double-checke!!!')
            else:
                for i in range(self._frame_num):
                    low_inter = segments*i
                    high_inter = segments*(i + 1)
                    frame_index_i = np.random.randint(low_inter, high_inter)
                    frame_index.append(frame_index_i)
                frame_index = np.array(frame_index)
                sample = sample[frame_index, :]
                sample = self.transform(sample)
                sample = torch.squeeze(sample)

        else:   # # test
            name, score = self._samples[index], self._scores[index]
            name = name.split(".")[0] + ".npy"
            path = os.path.join(self._vector_dir, name)
            sample = np.load(path)
            sample = np.squeeze(sample)
            frame_number = sample.shape[0]  # video frame number
            vector_dim = sample.shape[1]
            segments = frame_number//self._frame_num  # the number of segments

            pseudo_batch_size = segments
            test_data = np.zeros((pseudo_batch_size, self._frame_num, vector_dim))

            for idx in range(pseudo_batch_size):

                frame_index = []
                for i in range(self._frame_num):
                    low_inter = segments * i
                    high_inter = segments * (i + 1)
                    frame_index_i = np.random.randint(low_inter, high_inter)
                    frame_index.append(frame_index_i)
                frame_index = np.array(frame_index)
                sample_i = sample[frame_index, :]
                test_data[idx, :, :] = sample_i[:, :]

            test_data = test_data.transpose((1, 2, 0))
            sample = self.transform(test_data)
            sample = torch.squeeze(sample)

            length = np.zeros((pseudo_batch_size, 1))
            for idx in range(pseudo_batch_size):
                length[idx] = self._frame_num

        return sample, score, length

    def sample_strategy(self, frame_list):
        pass

    def __len__(self):
        length = len(self._samples)
        return length


class BTFVVFolderVSFA(data.Dataset):
    """
    Paper:
        FVV QoE
    Note:
        totally, 1944 FVVs and the associated quality scores
        we split the database into training set (1555) and testing test (389)
    """
    def __init__(self, url_mos_dir, vector_dir, iter_num, max_len=300, train_test=True, feat_dim=4096, transform=None):
        """

        Args:
            url_mos_dir: dir where the FVV names and the associated mos are saved
            vector_dir: dir where the feature vectors are saved
            iter_num: 0~20, denotes the iteration number
            frame_num: The number of consecutive frames
            train_test: True -> train, False -> test
            transform: -> tensor
        """

        self._url_mos_dir = url_mos_dir  # "./First_Mos"
        self._vector_dir = vector_dir   # "./VSFA_Features"
        self._max_len = max_len  # extract extract number of frames
        self._train_test = train_test
        self._feat_dim = feat_dim

        im_name = []
        im_score = []

        if self._train_test:
            name = "train_iter_"
        else:
            name = "test_iter_"

        file_name = name + str(iter_num) + ".npy"
        data = np.load(os.path.join(url_mos_dir, file_name))
        data_num = data.shape[1]

        self._length = np.zeros((data_num, 1))

        for idx in range(data_num):
            im_name.append(data[0][idx])
            im_score.append(float(data[1][idx]))
            self._length[idx] = self._max_len

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

        self._samples = im_name
        self._scores = im_score

        print('im_name length: {}'.format(len(self._samples)))
        print('im_score length: {}'.format(len(self._scores)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        # # normalize the input frames to the same quantity
        frames = np.zeros((self._max_len, self._feat_dim))
        length = self._length[index]

        name, score = self._samples[index], self._scores[index]
        name = name.split(".")[0] + ".npy"
        path = os.path.join(self._vector_dir, name)
        sample = np.load(path)
        sample = np.squeeze(sample)

        frame_number = sample.shape[0]
        frames[0:frame_number, :] = sample[:, :]

        sample = self.transform(frames)
        sample = torch.squeeze(sample)

        return sample, score, length

    def __len__(self):
        length = len(self._samples)
        return length


if __name__ == "__main__":

    # # test FVV Folder
    print("good luck ...")
