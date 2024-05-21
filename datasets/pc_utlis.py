import torch
import random
from sklearn.neighbors import KDTree
import numpy as np


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device

    B, C, N = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # B x npoint
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    centroids_vals = torch.zeros(B, C, npoint).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, :, farthest].view(B, 3, 1)  # get the current chosen point value
        centroids_vals[:, :, i] = centroid[:, :, 0].clone()
        dist = torch.sum((xyz - centroid) ** 2, 1)  # euclidean distance of points from the current centroid
        mask = dist < distance  # save index of all point that are closer than the current max distance
        distance[mask] = dist[
            mask]  # save the minimal distance of each point from all points that were chosen until now
        farthest = torch.max(distance, -1)[1]  # get the index of the point farthest away
    return centroids, centroids_vals


def farthest_point_sample_np(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    B, C, N = xyz.shape
    centroids = np.zeros((B, npoint), dtype=np.int64)
    distance = np.ones((B, N)) * 1e10
    farthest = np.random.randint(0, N, (B,), dtype=np.int64)
    batch_indices = np.arange(B, dtype=np.int64)
    centroids_vals = np.zeros((B, C, npoint))
    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, :, farthest].reshape(B, 3, 1)  # get the current chosen point value
        centroids_vals[:, :, i] = centroid[:, :, 0].copy()
        dist = np.sum((xyz - centroid) ** 2, 1)  # euclidean distance of points from the current centroid
        mask = dist < distance  # save index of all point that are closer than the current max distance
        distance[mask] = dist[
            mask]  # save the minimal distance of each point from all points that were chosen until now
        farthest = np.argmax(distance, axis=1)  # get the index of the point farthest away
    return centroids, centroids_vals


def farthest_point_sample_one_np(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    N, C = xyz.shape
    centroids = np.zeros(npoint, dtype=np.int64)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N, dtype=np.int64)
    centroids_vals = np.zeros((npoint, C))
    for i in range(npoint):
        centroids[i] = farthest  # save current chosen point index
        centroid = xyz[farthest].reshape(1, C)  # get the current chosen point value
        centroids_vals[i] = centroid.copy()
        dist = np.sum((xyz - centroid) ** 2, 1)  # euclidean distance of points from the current centroid
        mask = dist < distance  # save index of all point that are closer than the current max distance
        distance[mask] = dist[mask]  # save the minimal distance of each point from all points that were chosen until now
        farthest = np.argmax(distance, axis=0)  # get the index of the point farthest away
    return centroids_vals


def random_rotate_data_SO3(x1):
    """ Randomly rotate the point cloud to augument the dataset
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, rotated point cloud
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    x1 = rotate_one_axis_by_angle(x1, 'x', rotation_angle)
    rotation_angle = np.random.uniform() * 2 * np.pi
    x1 = rotate_one_axis_by_angle(x1, 'y', rotation_angle)
    rotation_angle = np.random.uniform() * 2 * np.pi
    x1 = rotate_one_axis_by_angle(x1, 'z', rotation_angle)

    return x1


def rotate_one_axis_by_angle(x, axis, angle):
    """
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
        angle: rotation angle
    Return:
        A rotated shape
    """
    R_x = np.asarray([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    R_y = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    R_z = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    if axis == "x":
        return x.dot(R_x).astype('float32')
    elif axis == "y":
        return x.dot(R_y).astype('float32')
    else:
        return x.dot(R_z).astype('float32')


def random_rotate_one_axis(X, axis):
    """
    Apply random rotation about one axis
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
    Return:
        A rotated shape
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    if axis == 'x':
        R_x = [[1, 0, 0], [0, cosval, -sinval], [0, sinval, cosval]]
        X = np.matmul(X, R_x)
    elif axis == 'y':
        R_y = [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        X = np.matmul(X, R_y)
    else:
        R_z = [[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]]
        X = np.matmul(X, R_z)
    return X.astype('float32')


def translate_pointcloud(pointcloud):
    """
    Input:
        pointcloud: pointcloud data, [B, C, N]
    Return:
        A translated shape
    """
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    """
    Input:
        pointcloud: pointcloud data, [N, C]
        sigma:
        clip:
    Return:
        A jittered shape
    """
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud.astype('float32')


def scale_to_unit_cube(x):
    """
   Input:
       x: pointcloud data, [B, C, N]
   Return:
       A point cloud scaled to unit cube
   """
    if len(x) == 0:
        return x

    centroid = np.mean(x, axis=0)
    x -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(x) ** 2, axis=-1)))
    x /= furthest_distance
    return x, furthest_distance


def normalization(cloud):
    bbox = np.zeros((2, 3))
    bbox[0][0] = np.min(cloud[:, 0])
    bbox[0][1] = np.min(cloud[:, 1])
    bbox[0][2] = np.min(cloud[:, 2])
    bbox[1][0] = np.max(cloud[:, 0])
    bbox[1][1] = np.max(cloud[:, 1])
    bbox[1][2] = np.max(cloud[:, 2])
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max()
    scale1 = 1 / scale
    for i in range(cloud.shape[0]):
        cloud[i] = cloud[i] - loc
        cloud[i] = cloud[i] * scale1
    return cloud


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v):  # if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    else:
        return np.eye(3)  # cross of all zeros only occurs on identical directions


def region_dropout_pc(pc, N, M):
    data_out = pc.copy()
    indices = np.random.randint(pc.shape[0], size=N)

    region_centers = data_out[indices, :]
    tree = KDTree(pc)
    dist, idx = tree.query(region_centers, M)

    data_out = np.delete(pc, idx, 0)

    return data_out


def drop_hole(pc, p):
    random_point = np.random.randint(0, pc.shape[0])
    index = np.linalg.norm(pc - pc[random_point].reshape(1, 3), axis=1).argsort()
    return pc[index[int(pc.shape[0] * p):]]


def density(pc, v_point=np.array([1, 0, 0]), gate=1):
    dist = np.sqrt((v_point ** 2).sum())
    max_dist = dist + 1
    min_dist = dist - 1
    dist = np.linalg.norm(pc - v_point.reshape(1,3), axis=1)
    dist = (dist - min_dist) / (max_dist - min_dist)
    r_list = np.random.uniform(0, 1, pc.shape[0])
    tmp_pc = pc[dist * gate < r_list]
    return tmp_pc


def rotate_point_cloud_3d(pc):
    rotation_angle = np.random.rand(3) * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix_1 = np.array([[cosval[0], 0, sinval[0]],
                                 [0, 1, 0],
                                 [-sinval[0], 0, cosval[0]]])
    rotation_matrix_2 = np.array([[1, 0, 0],
                                 [0, cosval[1], -sinval[1]],
                                 [0, sinval[1], cosval[1]]])
    rotation_matrix_3 = np.array([[cosval[2], -sinval[2], 0],
                                 [sinval[2], cosval[2], 0],
                                 [0, 0, 1]])
    rotation_matrix = np.matmul(np.matmul(rotation_matrix_1, rotation_matrix_2), rotation_matrix_3)
    rotated_data = np.dot(pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def p_scan(pc, pixel_size=0.017):
    pixel = int(2 / pixel_size)
    rotated_pc = rotate_point_cloud_3d(pc)
    pc_compress = (rotated_pc[:, 2] + 1) / 2 * pixel * pixel + (rotated_pc[:, 1] + 1) / 2 * pixel
    points_list = [None for i in range((pixel + 5) * (pixel + 5))]
    pc_compress = pc_compress.astype(np.int)
    for index, point in enumerate(rotated_pc):
        compress_index = pc_compress[index]
        if compress_index > len(points_list):
            print('out of index:', compress_index, len(points_list), point, pc[index], (pc[index] ** 2).sum(), (point ** 2).sum())
        if points_list[compress_index] is None:
            points_list[compress_index] = index
        elif point[0] > rotated_pc[points_list[compress_index]][0]:
            points_list[compress_index] = index
    points_list = list(filter(lambda x: x is not None, points_list))
    points_list = pc[points_list]
    return points_list


def angle_axis(angle, axis):
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                              [u[2], 0.0, -u[0]],
                              [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    return R.float()


class PointcloudScale(object):
    def __init__(self, lo=0.8, hi=1.25):
        self.lo, self.hi = lo, hi

    def __call__(self, points):
        scaler = np.random.uniform(self.lo, self.hi)
        points[:, 0:3] *= scaler
        return points


class PointcloudRotate(object):
    def __init__(self, axis=np.array([0.0, 1.0, 0.0])):
        self.axis = axis

    def __call__(self, points):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = angle_axis(rotation_angle, self.axis)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

    def _get_angles(self):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )

        return angles

    def __call__(self, points):
        angles = self._get_angles()
        Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = (
            points.new(points.size(0), 3)
            .normal_(mean=0.0, std=self.std)
            .clamp_(-self.clip, self.clip)
        )
        points[:, 0:3] += jittered_data
        return points


class PointcloudTranslate(object):
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, points):
        translation = np.random.uniform(-self.translate_range, self.translate_range)
        points[:, 0:3] += translation
        return points


class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()
