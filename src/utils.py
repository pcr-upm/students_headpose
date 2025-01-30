import torch
import torch.nn.functional as F


def convert_6d_to_rotation_matrix(poses):
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    eps = torch.finfo(poses.dtype).eps
    x = F.normalize(x_raw, dim=1, eps=eps)  # batch*3
    z = torch.cross(x, y_raw, dim=1)  # batch*3
    z = F.normalize(z, dim=1, eps=eps)  # batch*3
    y = torch.cross(z, x, dim=1)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix
