import os
import glob
import h5py
import numpy as np
import random
import torch
import torch.nn as nn
import sys
import argparse
from torch.utils.data import Dataset
from datasets.pc_utlis import (farthest_point_sample_np, scale_to_unit_cube, rotate_one_axis_by_angle,
                               random_rotate_one_axis, drop_hole, jitter_pointcloud)
  
  

eps = 10e-4
NUM_POINTS = 1024
PATCH_NUM = 27
neighbor_num = 128
knn_mode = True
idx_to_label = {0: "bathtub", 1: "bed", 2: "bookshelf", 3: "cabinet",
                4: "chair", 5: "lamp", 6: "monitor",
                7: "plant", 8: "sofa", 9: "table"}
label_to_idx = {"bathtub": 0, "bed": 1, "bookshelf": 2, "cabinet": 3,
                "chair": 4, "lamp": 5, "monitor": 6,
                "plant": 7, "sofa": 8, "table": 9}


def load_data_h5py_scannet10(partition, dataroot):
    """
    Input:
        partition - train/test
    Return:
        data,label arrays
    """
    DATA_DIR = dataroot + '/PointDA_data/scannet'
    all_data = []
    all_label = []
    for h5_name in sorted(glob.glob(os.path.join(DATA_DIR, '%s_*.h5' % partition))):
        f = h5py.File(h5_name, 'r')
        data = f['data'][:]
        label = f['label'][:]
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return np.array(all_data).astype('float32'), np.array(all_label).astype('int64')


def knn(pc, pc_patch_center, neighbor_num):

    pc = pc.squeeze(0).transpose(1, 0) #[1024, 3]
    pc_patch_center = pc_patch_center.squeeze(0).transpose(1, 0) #[196, 3]
    inner = -2 * np.dot(pc_patch_center, pc.transpose(1, 0)) #[196, 1024]
    pc_2 = np.tile((pc ** 2).sum(axis = 1), (PATCH_NUM, 1)) #[196, 1024]
    pc_patch_center_2 = np.tile((pc_patch_center ** 2).sum(axis = 1), (NUM_POINTS, 1)).transpose(1, 0)
    pairwise_distance = -pc_2 - inner - pc_patch_center_2

    idx = pairwise_distance.argsort()
    return idx[:, -neighbor_num : ]



class ScanNet(Dataset):
    """
    scannet dataset for pytorch dataloader
    """
    def __init__(self, io, dataroot, partition='train', random_rotation=True):
        self.partition = partition
        self.random_rotation = random_rotation

        # read data
        self.data, self.label = load_data_h5py_scannet10(self.partition, dataroot)
        self.num_examples = self.data.shape[0]

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in scannet" + ": " + str(self.data.shape[0]))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in scannet " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.copy(self.data[item])[:, :3]
        label = np.copy(self.label[item])
        pointcloud, _ = scale_to_unit_cube(pointcloud)
        # Rotate ScanNet by -90 degrees
        pointcloud = self.rotate_pc(pointcloud)

        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud_raw = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            _, pointcloud = farthest_point_sample_np(pointcloud_raw, NUM_POINTS)
            idx_center, pc_patch_center = farthest_point_sample_np(pointcloud, PATCH_NUM)
            idx_patch = knn(pointcloud, pc_patch_center, neighbor_num)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        # apply data rotation and augmentation on train samples
        if self.random_rotation==True:
            pointcloud = random_rotate_one_axis(pointcloud, "z")
        if self.partition == 'train' and item not in self.val_ind:
            pointcloud = jitter_pointcloud(pointcloud)
        point_patch = pointcloud[idx_patch]#.reshape(PATCH_NUM, -1)


        return (pointcloud, point_patch, label)

    def __len__(self):
        return self.data.shape[0]

    # scannet is rotated such that the up direction is the y axis
    def rotate_pc(self, pointcloud):
        pointcloud = rotate_one_axis_by_angle(pointcloud, 'x', -np.pi / 2)
        return pointcloud


class ModelNet(Dataset):
    """
    modelnet dataset for pytorch dataloader
    """
    def __init__(self, io, dataroot, partition='train', random_rotation=True):
        self.partition = partition
        self.random_rotation = random_rotation
        self.pc_list = []
        self.lbl_list = []
        DATA_DIR = os.path.join(dataroot, "PointDA_data", "modelnet")

        npy_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', partition, '*.npy')))

        for _dir in npy_list:
            self.pc_list.append(_dir)
            self.lbl_list.append(label_to_idx[_dir.split('/')[-3]])

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in modelnet : " + str(len(self.pc_list)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in modelnet " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.load(self.pc_list[item])[:, :3].astype(np.float32)
        label = np.copy(self.label[item])
        pointcloud, _ = scale_to_unit_cube(pointcloud)
        if self.partition == 'train':
            p = random.uniform(0.2, 0.4)
            pointcloud = drop_hole(pointcloud, p)

        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud_raw = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            _, pointcloud = farthest_point_sample_np(pointcloud_raw, NUM_POINTS)
            idx_center, pc_patch_center = farthest_point_sample_np(pointcloud, PATCH_NUM)
            idx_patch = knn(pointcloud, pc_patch_center, neighbor_num)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        # apply data rotation and augmentation on train samples
        if self.random_rotation==True:
            pointcloud = random_rotate_one_axis(pointcloud, "z")
        if self.partition == 'train': # and item not in self.val_ind:
            pointcloud = jitter_pointcloud(pointcloud)

        point_patch = pointcloud[idx_patch]#.reshape(PATCH_NUM, -1)

        return (pointcloud, point_patch, label)

    def __len__(self):
        return len(self.pc_list)


class ShapeNet(Dataset):
    """
    Sahpenet dataset for pytorch dataloader
    """
    def __init__(self, io, dataroot, partition='train', random_rotation=True):
        self.partition = partition
        self.random_rotation = random_rotation
        self.pc_list = []
        self.lbl_list = []
        DATA_DIR = os.path.join(dataroot, "PointDA_data", "shapenet")

        npy_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', partition, '*.npy')))

        for _dir in npy_list:
            self.pc_list.append(_dir)
            self.lbl_list.append(label_to_idx[_dir.split('/')[-3]])

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in shapenet: " + str(len(self.pc_list)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in shapenet " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.load(self.pc_list[item])[:, :3].astype(np.float32)
        label = np.copy(self.label[item])
        pointcloud, _ = scale_to_unit_cube(pointcloud)
        if self.partition == 'train':
            pointcloud_jitter = jitter_pointcloud(pointcloud)
            pointcloud = np.concatenate((pointcloud, pointcloud_jitter), 0)
            p = random.uniform(0.2, 0.4)
            pointcloud = drop_hole(pointcloud, p)
        # Rotate ShapeNet by -90 degrees
        pointcloud = self.rotate_pc(pointcloud, label)
        # sample according to farthest point sampling
 
        if pointcloud.shape[0] >= NUM_POINTS:
            pointcloud_raw = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            _, pointcloud = farthest_point_sample_np(pointcloud_raw, NUM_POINTS)
            idx_center, pc_patch_center = farthest_point_sample_np(pointcloud, PATCH_NUM)
            idx_patch = knn(pointcloud, pc_patch_center, neighbor_num)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        # apply data rotation and augmentation on train samples
        if self.random_rotation==True:
            pointcloud = random_rotate_one_axis(pointcloud, "z")
        if self.partition == 'train': # and item not in self.val_ind:
            pointcloud = jitter_pointcloud(pointcloud)

        point_patch = pointcloud[idx_patch]#.reshape(PATCH_NUM, -1)

        return (pointcloud, point_patch, label)

    def __len__(self):
        return len(self.pc_list)

    # shpenet is rotated such that the up direction is the y axis in all shapes except plant
    def rotate_pc(self, pointcloud, label):
        if label.item(0) != label_to_idx["plant"]:
            pointcloud = rotate_one_axis_by_angle(pointcloud, 'x', -np.pi / 2)
        return pointcloud