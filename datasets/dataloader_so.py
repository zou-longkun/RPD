import os
import glob
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from datasets.pc_utlis import *

NUM_POINTS = 1024
PATCH_NUM = 28
neighbor_num = 128
label_to_idx1 = {"bag": 0, "bed": 1, "cabinet": 2, "chair": 3, "display": 4,
                 "pillow": 5, "shelf": 6, "sofa": 7, "table": 8}

label_to_idx2 = {"bed": 0, "cabinet": 1, "chair": 2, "desk": 3, "display": 4,
                 "door": 5, "shelf": 6, "sink": 7, "sofa": 8, "table": 9, "toilet": 10}


def knn(pc, pc_patch_center, neighbor_num):
    pc = pc.squeeze(0).transpose(1, 0)  # [1024, 3]
    pc_patch_center = pc_patch_center.squeeze(0).transpose(1, 0)  # [196, 3]
    inner = -2 * np.dot(pc_patch_center, pc.transpose(1, 0))  # [196, 1024]
    pc_2 = np.tile((pc ** 2).sum(axis=1), (PATCH_NUM, 1))  # [196, 1024]
    pc_patch_center_2 = np.tile((pc_patch_center ** 2).sum(axis=1), (NUM_POINTS, 1)).transpose(1, 0)
    pairwise_distance = -pc_2 - inner - pc_patch_center_2

    idx = pairwise_distance.argsort()
    return idx[:, -neighbor_num:]


# sim2real augmentation
def process_data(pc):
    rand_points = np.random.uniform(-1, 1, 40000)
    x1 = rand_points[:20000]
    x2 = rand_points[20000:]
    power_sum = x1 ** 2 + x2 ** 2
    p_filter = power_sum < 1
    power_sum = power_sum[p_filter]
    sqrt_sum = np.sqrt(1 - power_sum)
    x1 = x1[p_filter]
    x2 = x2[p_filter]
    x = (2 * x1 * sqrt_sum).reshape(-1, 1)
    y = (2 * x2 * sqrt_sum).reshape(-1, 1)
    z = (1 - 2 * power_sum).reshape(-1, 1)
    density_points = np.hstack([x, y, z])
    fn = [
        lambda pc: drop_hole(pc, p=0.24),
        lambda pc: drop_hole(pc, p=0.36),
        lambda pc: drop_hole(pc, p=0.45),
        lambda pc: p_scan(pc, pixel_size=0.017),
        lambda pc: p_scan(pc, pixel_size=0.022),
        lambda pc: p_scan(pc, pixel_size=0.035),
        lambda pc: density(pc, density_points[np.random.choice(density_points.shape[0])], 1.3),
        lambda pc: density(pc, density_points[np.random.choice(density_points.shape[0])], 1.4),
        lambda pc: density(pc, density_points[np.random.choice(density_points.shape[0])], 1.6),
        lambda pc: pc.copy(),
    ]
    fn_index = list(range(len(fn)))
    ind = np.random.choice(fn_index)
    pc = fn[ind](pc)

    return pc


class ModelNet11(Dataset):
    """
    modelnet dataset for pytorch dataloader
    """
    def __init__(self, io, dataroot, partition='train'):
        self.partition = partition
        self.pc_list = []
        self.lbl_list = []
        PC_DIR = os.path.join(dataroot, "Sim2Real_data", "modelnet_11")
        pc_dir_list = sorted(glob.glob(os.path.join(PC_DIR, '*', partition, '*.npy')))

        for pc_dir in pc_dir_list:
            self.pc_list.append(pc_dir)
            self.lbl_list.append(label_to_idx2[pc_dir.split('/')[-3]])

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)
        self.transforms = transforms.Compose([
            PointcloudToTensor(),
            PointcloudScale(),
            PointcloudRotatePerturbation(),
            PointcloudTranslate(),
            # PointcloudJitter(),
        ])

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype('int64')
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype('int64')
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in modelnet_11 : " + str(len(self.pc_list)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in modelnet_11 " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.load(self.pc_list[item])[:, :3].astype('float32')
        label = np.copy(self.label[item])
        # pointcloud, furthest_distance = scale_to_unit_cube(pointcloud)
        pointcloud = normalization(pointcloud)
        # pointcloud = jitter_pointcloud(pointcloud)
        pointcloud = process_data(pointcloud)
        pointcloud = self.transforms(pointcloud).numpy()

        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            _, pointcloud = farthest_point_sample_np(pointcloud, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        if pointcloud.shape[0] < NUM_POINTS:
            indices = np.random.randint(pointcloud.shape[0], size=(NUM_POINTS - pointcloud.shape[0]))
            tem_pointcloud = pointcloud[indices]
            tem_pointcloud = jitter_pointcloud(tem_pointcloud)
            pointcloud = np.concatenate((pointcloud, tem_pointcloud), axis=0)

        pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
        _, pc_patch_center = farthest_point_sample_np(pointcloud, PATCH_NUM)
        idx_patch = knn(pointcloud, pc_patch_center, neighbor_num)
        pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        pointcloud = random_rotate_one_axis(pointcloud, 'z')
        point_patch = pointcloud[idx_patch]

        return pointcloud, point_patch, label

    def __len__(self):
        return len(self.pc_list)


class ScanObjectNet11(Dataset):
    """
    scannet dataset for pytorch dataloader
    """

    def __init__(self, io, dataroot, partition='train'):
        self.partition = partition
        self.pc_list = []
        self.lbl_list = []
        PC_DIR = os.path.join(dataroot, "Sim2Real_data", "scanobjectnn_11")
        pc_dir_list = sorted(glob.glob(os.path.join(PC_DIR, '*', partition, '*.npy')))

        for pc_dir in pc_dir_list:
            self.pc_list.append(pc_dir)
            self.lbl_list.append(label_to_idx2[pc_dir.split('/')[-3]])

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)
        self.transforms = transforms.Compose([
            PointcloudToTensor(),
            PointcloudScale(),
            PointcloudRotatePerturbation(),
            PointcloudTranslate(),
            PointcloudJitter(),
        ])

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype('int64')
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype('int64')
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in scanobjectnn_11 : " + str(len(self.pc_list)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in scanobjectnn_11 " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.load(self.pc_list[item])[:, :3].astype('float32')
        label = np.copy(self.label[item])
        # pointcloud, furthest_distance = scale_to_unit_cube(pointcloud)
        pointcloud = normalization(pointcloud)
        # pointcloud = jitter_pointcloud(pointcloud)
        # pointcloud = self.transforms(pointcloud).numpy()

        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            _, pointcloud = farthest_point_sample_np(pointcloud, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        if pointcloud.shape[0] < NUM_POINTS:
            indices = np.random.randint(pointcloud.shape[0], size=(NUM_POINTS - pointcloud.shape[0]))
            tem_pointcloud = pointcloud[indices]
            tem_pointcloud = jitter_pointcloud(tem_pointcloud)
            pointcloud = np.concatenate((pointcloud, tem_pointcloud), axis=0)

        pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
        _, pc_patch_center = farthest_point_sample_np(pointcloud, PATCH_NUM)
        idx_patch = knn(pointcloud, pc_patch_center, neighbor_num)
        pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        # scannet is rotated such that the up direction is the z axis
        pointcloud = rotate_one_axis_by_angle(pointcloud, 'x', -np.pi / 2)
        pointcloud = random_rotate_one_axis(pointcloud, 'z')
        point_patch = pointcloud[idx_patch]

        return pointcloud, point_patch, label

    def __len__(self):
        return len(self.pc_list)


class ShapeNet9(Dataset):
    """
    shapenet dataset for pytorch dataloader
    """
    def __init__(self, io, dataroot, partition='train'):
        self.partition = partition
        self.pc_list = []
        self.lbl_list = []
        PC_DIR = os.path.join(dataroot, "Sim2Real_data", "shapenet_9")
        pc_dir_list = sorted(glob.glob(os.path.join(PC_DIR, '*', partition, '*.npy')))

        for pc_dir in pc_dir_list:
            self.pc_list.append(pc_dir)
            self.lbl_list.append(label_to_idx1[pc_dir.split('/')[-3]])

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)
        self.transforms = transforms.Compose([
            PointcloudToTensor(),
            PointcloudScale(),
            PointcloudRotatePerturbation(),
            PointcloudTranslate(),
            # PointcloudJitter(),
        ])

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype('int64')
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype('int64')
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in shapenet_9 : " + str(len(self.pc_list)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in shapenet_9 " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.load(self.pc_list[item])[:, :3].astype('float32')
        label = np.copy(self.label[item])
        # pointcloud, furthest_distance = scale_to_unit_cube(pointcloud)
        pointcloud = normalization(pointcloud)
        # pointcloud = jitter_pointcloud(pointcloud)
        pointcloud = process_data(pointcloud)
        pointcloud = self.transforms(pointcloud).numpy()

        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            _, pointcloud = farthest_point_sample_np(pointcloud, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        if pointcloud.shape[0] < NUM_POINTS:
            indices = np.random.randint(pointcloud.shape[0], size=(NUM_POINTS - pointcloud.shape[0]))
            tem_pointcloud = pointcloud[indices]
            tem_pointcloud = jitter_pointcloud(tem_pointcloud)
            pointcloud = np.concatenate((pointcloud, tem_pointcloud), axis=0)

        pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
        _, pc_patch_center = farthest_point_sample_np(pointcloud, PATCH_NUM)
        idx_patch = knn(pointcloud, pc_patch_center, neighbor_num)
        pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        # shapenet is rotated such that the up direction is the z axis
        pointcloud = rotate_one_axis_by_angle(pointcloud, 'x', -np.pi / 2)
        pointcloud = random_rotate_one_axis(pointcloud, 'z')
        point_patch = pointcloud[idx_patch]

        return pointcloud, point_patch, label

    def __len__(self):
        return len(self.pc_list)


class ScanObjectNet9(Dataset):
    """
    scannet dataset for pytorch dataloader
    """

    def __init__(self, io, dataroot, partition='train'):
        self.partition = partition
        self.pc_list = []
        self.lbl_list = []
        PC_DIR = os.path.join(dataroot, "Sim2Real_data", "scanobjectnn_9")
        pc_dir_list = sorted(glob.glob(os.path.join(PC_DIR, '*', partition, '*.npy')))

        for pc_dir in pc_dir_list:
            self.pc_list.append(pc_dir)
            self.lbl_list.append(label_to_idx1[pc_dir.split('/')[-3]])

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype('int64')
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype('int64')
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in scanobjectnn_9 : " + str(len(self.pc_list)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in scanobject_9 " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.load(self.pc_list[item])[:, :3].astype('float32')
        label = np.copy(self.label[item])
        # pointcloud, furthest_distance = scale_to_unit_cube(pointcloud)
        pointcloud = normalization(pointcloud)
        # pointcloud = jitter_pointcloud(pointcloud)

        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            _, pointcloud = farthest_point_sample_np(pointcloud, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        if pointcloud.shape[0] < NUM_POINTS:
            indices = np.random.randint(pointcloud.shape[0], size=(NUM_POINTS - pointcloud.shape[0]))
            tem_pointcloud = pointcloud[indices]
            tem_pointcloud = jitter_pointcloud(tem_pointcloud)
            pointcloud = np.concatenate((pointcloud, tem_pointcloud), axis=0)

        pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
        _, pc_patch_center = farthest_point_sample_np(pointcloud, PATCH_NUM)
        idx_patch = knn(pointcloud, pc_patch_center, neighbor_num)
        pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        # scannet is rotated such that the up direction is the z axis
        pointcloud = rotate_one_axis_by_angle(pointcloud, 'x', -np.pi / 2)
        pointcloud = random_rotate_one_axis(pointcloud, 'z')
        point_patch = pointcloud[idx_patch]

        return pointcloud, point_patch, label

    def __len__(self):
        return len(self.pc_list)

