"""
Stub for pytorch3d to allow GR00T to load without full pytorch3d installation.
This is a temporary workaround for testing.
"""

import torch
import numpy as np


class MockTransforms:
    """Mock pytorch3d.transforms module."""
    
    @staticmethod
    def matrix_to_quaternion(matrix):
        """Convert rotation matrix to quaternion (simplified)."""
        # Simple stub - returns identity quaternion
        batch_size = matrix.shape[0] if len(matrix.shape) > 2 else 1
        return torch.tensor([[1.0, 0.0, 0.0, 0.0]] * batch_size)
    
    @staticmethod
    def quaternion_to_matrix(quaternion):
        """Convert quaternion to rotation matrix (simplified)."""
        # Simple stub - returns identity matrix
        batch_size = quaternion.shape[0] if len(quaternion.shape) > 1 else 1
        return torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
    
    @staticmethod
    def axis_angle_to_matrix(axis_angle):
        """Convert axis-angle to rotation matrix (simplified)."""
        batch_size = axis_angle.shape[0] if len(axis_angle.shape) > 1 else 1
        return torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
    
    @staticmethod
    def matrix_to_axis_angle(matrix):
        """Convert rotation matrix to axis-angle (simplified)."""
        batch_size = matrix.shape[0] if len(matrix.shape) > 2 else 1
        return torch.zeros(batch_size, 3)
    
    @staticmethod
    def euler_angles_to_matrix(euler_angles, convention="XYZ"):
        """Convert Euler angles to rotation matrix (simplified)."""
        batch_size = euler_angles.shape[0] if len(euler_angles.shape) > 1 else 1
        return torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
    
    @staticmethod
    def matrix_to_euler_angles(matrix, convention="XYZ"):
        """Convert rotation matrix to Euler angles (simplified)."""
        batch_size = matrix.shape[0] if len(matrix.shape) > 2 else 1
        return torch.zeros(batch_size, 3)


# Create module structure
class pytorch3d:
    transforms = MockTransforms()


# Monkey-patch into sys.modules
import sys
sys.modules['pytorch3d'] = pytorch3d
sys.modules['pytorch3d.transforms'] = MockTransforms