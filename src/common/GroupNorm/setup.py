from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
setup(
    name='groupnorm_ext',
    ext_modules=[
        CUDAExtension(
            'groupnorm_ext',
            [
                'src/test.cu',
            ],
            include_dirs=[],
            extra_compile_args={
                'cxx': ['-std=c++17'],
                'nvcc': [
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '--extended-lambda',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_90,code=sm_90'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)