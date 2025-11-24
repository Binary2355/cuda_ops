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

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        os.makedirs(extdir, exist_ok=True)

        current_file = Path(__file__).resolve()
        project_root = current_file.parent
        
        cutlass_path_abs = Path('./third_party/cutlass/').resolve()
        cutlass_tool_path_abs = Path('./third_party/cutlass/tools/util/').resolve()

        project_root = Path(__file__).parent
        ops_sources = [str(project_root / 'src' / 'swiglu.cu')]
        sources_str = ';'.join(ops_sources)
        name = 'swiglu_ext'

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE=Release',
            f'-DCUTLASS_PATH={cutlass_path_abs}',
            f'-DCUTLASS_TOOL_PATH={cutlass_tool_path_abs}',
            f'-DOPS_SOURCES={sources_str}',
            f'-DPROJECT_NAME={name}',
            f'-DCUTLASS_NVCC_ARCHS="80;86;89;90"',
            f'-DCUTLASS_ENABLE_PROFILER=ON',
            f'-DCUTLASS_ENABLE_TENSOR_CORE_MMA=ON',
        ]

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
    name='swiglu_ext',
    ext_modules=[CMakeExtension('swiglu_ext')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)