
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.7")

setup(
    name='mailbox_ext',
    ext_modules=[
        CUDAExtension(
            name='mailbox_ext',
            sources=[
                'src/mailbox_ext.cpp',
                'src/mailbox_peripheral.cu',
                'src/mailbox_focus.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
