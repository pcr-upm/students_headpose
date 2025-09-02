import torch
from scipy.spatial.transform import Rotation
import torch.nn.functional as F


# Y. Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"
# https://github.com/papagina/RotationContinuity
def convert_6d_to_rotation_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def convert_euler_tensor_to_rotation_matrix(tensor, device, order):
    rotations = Rotation.from_euler(order, tensor.detach().cpu().numpy(), degrees=True)
    rotation_matrices = rotations.as_matrix()
    return torch.tensor(rotation_matrices, dtype=torch.float32, requires_grad=tensor.requires_grad).to(device)

def convert_rotation_matrix_to_euler(matrix, order, device, degrees=True):
    # Detach the tensor from the computation graph and convert to numpy array
    rotation = Rotation.from_matrix(matrix.detach().cpu().numpy())
    # Convert the Rotation object to Euler angles
    euler_angles = rotation.as_euler(order, degrees=degrees)
    # Convert the Euler angles back to a tensor
    euler_tensor = torch.tensor(euler_angles, dtype=torch.float32, requires_grad=matrix.requires_grad).to(device)
    return euler_tensor
