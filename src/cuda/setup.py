from pybind11 import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='neural_kernels',
    ext_modules=[
        CUDAExtension('neural_kernels', [
            'neural_kernels.cpp',
            'neural_kernels.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)
