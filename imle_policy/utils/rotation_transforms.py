# Functions to convert between 6D rotation representation and rotation matrix extracted from pytorch3D

from typing import Union
import torch
import torch.nn.functional as F
import pdb
import scipy

def quaternion_to_rotation_6D(quaternion: torch.Tensor) -> torch.Tensor:
    rotation_matrix = scipy.spatial.transform.Rotation.from_quat(quaternion).as_matrix()
    rotation_6D = matrix_to_rotation_6d(rotation_matrix)
    return rotation_6D

def rotation_6D_to_quaternion(rotation_6D: torch.Tensor) -> torch.Tensor:
    rotation_matrix = rotation_6d_to_matrix(rotation_6D)
    quaternion = scipy.spatial.transform.Rotation.from_matrix(rotation_matrix).as_quat()
    return quaternion

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix)

    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


# test
# d6 = torch.rand(2, 6)
# matrix = rotation_6d_to_matrix(d6)
# d6_ = matrix_to_rotation_6d(matrix)
# matrix_ = rotation_6d_to_matrix(d6_)
# print(matrix)
# print(matrix_)
# assert torch.allclose(matrix, matrix_)

# create a rotation matrix
# rot = scipy.spatial.transform.Rotation.from_euler('xyz', [0.1, 0.2, 0.3]).as_matrix()
# # convert to quaternion
# quat = scipy.spatial.transform.Rotation.from_matrix(rot).as_quat()
# print(quat)
# rot_6D = quaternion_to_rotation_6D(quat)
# print(rot_6D)
# quat_ = rotation_6D_to_quaternion(rot_6D)
# print(quat_)