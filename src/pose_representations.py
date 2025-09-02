from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, Type

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

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Abstract property for representation name"""
        pass

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
    """Factory with registry pattern and better error reporting."""

    _registry: Dict[str, Type[PoseRepresentation]] = {
        cls.name: cls for cls in PoseRepresentation.__subclasses__()
    }

    @classmethod
    def create_pose_representation(cls, name: str, params: ConversionParams) -> PoseRepresentation:
        """Create pose representation instance with validation."""
        try:
            representation_class = cls._registry[name.lower()]
        except KeyError:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown pose representation: {name}. Available: {available}")

        return representation_class(params)


class PoseLossCalculator(ABC, metaclass=SingletonMeta):

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Abstract property for representation name"""
        pass

    def __init__(self, pose_representation_output, pose_representation_target, loss=None):
        self.pose_representation_output = pose_representation_output
        self.pose_representation_target = pose_representation_target
        self.loss = loss

    @abstractmethod
    def calculate_loss(self, outputs, targets):
        pass

class GeodesicLossCalculator(PoseLossCalculator):
    name = 'geodesic'

    def __init__(self, pose_representation_output, pose_representation_target):
        super().__init__(pose_representation_output, pose_representation_target, GeodesicLoss())

    def calculate_loss(self, outputs, targets):
        outputs = self.pose_representation_output.convert_to_rotation_matrix(outputs)
        targets = self.pose_representation_target.convert_to_rotation_matrix(targets)
        return self.loss(outputs, targets)

class L1LossCalculator(PoseLossCalculator):
    name = 'l1'

    def __init__(self, pose_representation_output, pose_representation_target):
        super().__init__(pose_representation_output, pose_representation_target, nn.L1Loss())

    def calculate_loss(self, outputs, targets):
        outputs = self.pose_representation_output.convert_to_rotation_euler(outputs)
        targets = self.pose_representation_target.convert_to_rotation_euler(targets)
        return self.loss(outputs, targets)


class PoseLossCalculatorFactory:
    """Factory with registry pattern and better error handling."""

    _registry: Dict[str, Type[PoseLossCalculator]] = {
        cls.name: cls for cls in PoseLossCalculator.__subclasses__()
    }

    @classmethod
    def create_loss_calculator(
            cls,
            name: str,
            output_rep: PoseRepresentation,
            target_rep: PoseRepresentation
    ) -> PoseLossCalculator:
        """Create loss calculator instance with validation."""
        try:
            calculator_class = cls._registry[name.lower()]
        except KeyError:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown loss calculator: {name}. Available: {available}")

        return calculator_class(output_rep, target_rep)
