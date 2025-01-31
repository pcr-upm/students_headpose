from abc import ABC, ABCMeta, abstractmethod
import images_framework.alignment.students_headpose.src.conversions as conversions
from images_framework.alignment.students_headpose.src.losses import GeodesicLoss
import torch.nn as nn

class ConversionParams:
    def __init__(self, device, order):
        self.device = device
        self.order = order

class SingletonMeta(ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class PoseRepresentation(ABC, metaclass=SingletonMeta):

    name = None

    def __init__(self, params, num_classes=None):
        self._num_classes = num_classes
        self._params = params

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def params(self):
        return self._params

    @abstractmethod
    def convert_to_rotation_matrix(self, pose):
        pass

    @abstractmethod
    def convert_to_rotation_euler(self, pose):
        pass


class EulerPose(PoseRepresentation):

    name = 'euler'

    def __init__(self, params):
        super().__init__(num_classes=3, params=params)

    def convert_to_rotation_matrix(self, output):
        return conversions.convert_euler_tensor_to_rotation_matrix(output, self.params.device, self.params.order)

    def convert_to_rotation_euler(self, pose):
        return pose



class SixDPose(PoseRepresentation):

    name = '6d'

    def __init__(self, params):
        super().__init__(num_classes=6, params=params)


    def convert_to_rotation_matrix(self, pose):
        return conversions.convert_6d_to_rotation_matrix(pose)

    def convert_to_rotation_euler(self, pose):
        rot_matrix = self.convert_to_rotation_matrix(pose)
        return conversions.convert_rotation_matrix_to_euler(rot_matrix, self.params.order, self.params.device)


class PoseRepresentationFactory:
    @staticmethod
    def create_pose_representation(name, params):
        for pose_representation in PoseRepresentation.__subclasses__():
            if pose_representation.name == name:
                return pose_representation(params)
        raise ValueError(f"Unknown pose representation: {name}")


class PoseLossCalculator(ABC, metaclass=SingletonMeta):
    name = None

    def __init__(self, pose_representation_output, pose_representation_target, loss=None):
        self.pose_representation_output = pose_representation_output
        self.pose_representation_target = pose_representation_target
        self.loss = loss

    @abstractmethod
    def calculate_loss(self, outputs, targets):
        pass

class GeodesicLossCalculator(PoseLossCalculator):
    name = 'Geodesic'

    def __init__(self, pose_representation_output, pose_representation_target):
        super().__init__(pose_representation_output, pose_representation_target, GeodesicLoss())

    def calculate_loss(self, outputs, targets):
        outputs = self.pose_representation_output.convert_to_rotation_matrix(outputs)
        targets = self.pose_representation_target.convert_to_rotation_matrix(targets)
        return self.loss(outputs, targets)

class L1LossCalculator(PoseLossCalculator):
    name = 'L1'

    def __init__(self, pose_representation_output, pose_representation_target):
        super().__init__(pose_representation_output, pose_representation_target, nn.L1Loss())

    def calculate_loss(self, outputs, targets):
        outputs = self.pose_representation_output.convert_to_rotation_euler(outputs)
        targets = self.pose_representation_target.convert_to_rotation_euler(targets)
        return self.loss(outputs, targets)

class PoseLossCalculatorFactory:
    @staticmethod
    def create_loss_calculator(name, pose_representation_output, pose_representation_target):
        for subclass in PoseLossCalculator.__subclasses__():
            if subclass.name == name:
                return subclass(pose_representation_output, pose_representation_target)
        raise ValueError(f"Unknown loss calculator: {name}")