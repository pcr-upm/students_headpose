import torch


def _geodesic_distance(rot_mat_1: torch.Tensor, rot_mat_2: torch.Tensor) -> torch.Tensor:
    """Computes the geodesic distance between two batches of 3x3 rotation matrices.

    Args:
        rot_mat_1 (torch.tensor): rotation matrix batch of shape (..., 3, 3).
        rot_mat_2 (torch.tensor): rotation matrix batch of shape (..., 3, 3).

    Returns:
        torch.tensor: geodesic distance in radians of each pair of rotation matrices
        in the batch.
    """
    m = torch.bmm(rot_mat_1, rot_mat_2.transpose(1, 2))
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2

    # Avoid nan in backprop
    eps = torch.finfo(cos.dtype).eps
    cos = torch.clamp(cos, -1 + eps, 1 - eps)

    # Geodesic angle in radians
    return torch.acos(cos)


class GeodesicLoss(torch.nn.Module):

    def __init__(
            self,
            reduction: str = 'mean'
    ):
        super().__init__()

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f'Invalid reduction type {reduction}')

        self.reduction = reduction

    def forward(self, inputTensor: torch.Tensor, targetTensor: torch.Tensor) -> torch.Tensor:
        geo_dist = _geodesic_distance(inputTensor, targetTensor)
        loss = self._reduce_loss(geo_dist)
        return loss

    def _reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
