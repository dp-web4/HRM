"""
Patch for transformers to add missing VideoInput type for GR00T compatibility.
"""

import sys
from typing import Union, List
import numpy as np

# Define VideoInput if it doesn't exist
VideoInput = Union[
    List["PIL.Image.Image"],
    "np.ndarray",
    "torch.Tensor", 
    List["np.ndarray"],
    List["torch.Tensor"],
    List[List["PIL.Image.Image"]],
    List[List["np.ndarray"]],
    List[List["torch.Tensor"]],
]

# Monkey patch transformers.image_utils
if 'transformers.image_utils' in sys.modules:
    sys.modules['transformers.image_utils'].VideoInput = VideoInput
else:
    # Create a mock module
    class MockImageUtils:
        VideoInput = VideoInput
    
    sys.modules['transformers.image_utils'] = MockImageUtils()