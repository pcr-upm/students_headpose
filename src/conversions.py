import torch
from scipy.spatial.transform import Rotation
import torch.nn.functional as F

# Y. Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"
# https://github.com/papagina/RotationContinuity
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

def convert_euler_tensor_to_rotation_matrix(tensor, device, order):
    rotations = Rotation.from_euler(order, tensor.cpu(), degrees=True).as_matrix()
    return torch.tensor(rotations, dtype=torch.float32).to(device=device)


def convert_rotation_matrix_to_euler(matrix, order, device, degrees=True):
    # Convert the rotation matrix to a numpy array
    matrix_np = matrix.cpu().numpy()
    # Create a Rotation object from the rotation matrix
    rotation = Rotation.from_matrix(matrix_np)
    # Convert the Rotation object to Euler angles
    euler_angles = rotation.as_euler(order, degrees=degrees)
    # Convert the Euler angles back to a tensor
    euler_tensor = torch.tensor(euler_angles, dtype=torch.float32).to(device)
    return euler_tensor