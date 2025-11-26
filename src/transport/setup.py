import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='', sources=None):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.cmake_sources = sources or []

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        nvshmem_lib = os.environ.get('NVSHMEM_LIB')
        nvshmem_include = os.environ.get('NVSHMEM_INCLUDE')
        if nvshmem_lib:
            print(f"Found NVSHMEM_LIB: {nvshmem_lib}")
        else:
            print("WARNING: NVSHMEM_LIB environment variable not set. NVSHMEM support will be disabled.")
        if nvshmem_include:
            print(f"Found NVSHMEM_INCLUDE: {nvshmem_include}")
        else:
            print("WARNING: NVSHMEM_INCLUDE environment variable not set. NVSHMEM support will be disabled.")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        os.makedirs(extdir, exist_ok=True)

        current_file = Path(__file__).resolve()
        project_root = current_file.parent
        
        project_root = Path(__file__).parent
        ops_sources = [str(project_root / 'src' / 'rdma_tensor_move.cu')]
        sources_str = ';'.join(ops_sources)
        name = 'rdma_ext'

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE=Release',
            f'-DOPS_SOURCES={sources_str}',
            f'-DPROJECT_NAME={name}',
            f'-DCUDA_RESOLVE_DEVICE_SYMBOLS=ON',
        ]

        nvshmem_lib = os.environ.get('NVSHMEM_LIB')
        nvshmem_include = os.environ.get('NVSHMEM_INCLUDE')
        if nvshmem_lib:
            cmake_args.append(f'-DNVSHMEM_LIB={nvshmem_lib}')
        if nvshmem_include:
            cmake_args.append(f'-DNVSHMEM_INCLUDE={nvshmem_include}')

        build_args = ['--', '-j1']

        env = os.environ.copy()
        env['CUDACXX'] = 'nvcc'
        
        build_temp = Path(self.build_temp)
        if not build_temp.exists():
            build_temp.mkdir(parents=True)
            
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=str(build_temp), env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=str(build_temp))

setup(
    name='rdma_ext',
    ext_modules=[CMakeExtension('rdma_ext')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)