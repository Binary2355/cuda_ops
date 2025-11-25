from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

import os
os.environ['PIP_NO_INDEX'] = '1'

setup(
    name='focalloss_ext',
    ext_modules=[
        CUDAExtension(
            'focalloss_ext',
            [
                'src/focalloss.cu',
            ],
            include_dirs=[],
            extra_compile_args={
                'cxx': ['-std=c++17'],
                'nvcc': [
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '--extended-lambda',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_90,code=sm_90'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)