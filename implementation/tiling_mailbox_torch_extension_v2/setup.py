
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mailbox_ext',
    ext_modules=[
        CUDAExtension(
            name='mailbox_ext',
            sources=[
                'src/mailbox_ext.cpp',
                'src/mailbox_kernels.cu',
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
