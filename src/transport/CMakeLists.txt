cmake_minimum_required(VERSION 3.18)
project(${PROJECT_NAME})

set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

# 新增：查找 NVSHMEM
if(NOT DEFINED ENV{NVSHMEM_LIB})
    message(ERROR "NVSHMEM_LIB environment variable not set, NVSHMEM support may not be available")
else()
    set(NVSHMEM_LIB $ENV{NVSHMEM_LIB})
    message(STATUS "Found NVSHMEM at: ${NVSHMEM_LIB}")
endif()
if(NOT DEFINED ENV{NVSHMEM_INCLUDE})
    message(ERROR "NVSHMEM_INCLUDE environment variable not set, NVSHMEM support may not be available")
else()
    set(NVSHMEM_INCLUDE $ENV{NVSHMEM_INCLUDE})
    message(STATUS "Found NVSHMEM at: ${NVSHMEM_INCLUDE}")
endif()

set(OPS_SOURCES "" CACHE STRING "List of CUDA source files")

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Python found: ${Python_EXECUTABLE}")
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_FIND_RESULT
)

if(TORCH_FIND_RESULT EQUAL 0)
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
    message(STATUS "Added Torch to CMAKE_PREFIX_PATH: ${TORCH_CMAKE_PREFIX_PATH}")
else()
    message(WARNING "Could not find Torch via Python, trying default find_package")
endif()

find_package(Torch REQUIRED)
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS ${TORCH_INSTALL_PREFIX}/lib)
if(TORCH_PYTHON_LIBRARY)
    message(STATUS "Found torch_python: ${TORCH_PYTHON_LIBRARY}")
    list(APPEND TORCH_LIBRARIES ${TORCH_PYTHON_LIBRARY})
else()
    message(WARNING "Could not find torch_python library")
endif()

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "
import torch
print(torch.compiled_with_cxx11_abi())
"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "PyTorch CXX11 ABI: ${TORCH_CXX11_ABI}")

include_directories(
    ${CMAKE_CURRENT_DIR}/src
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# 新增：添加 NVSHMEM 头文件路径
if(NVSHMEM_INCLUDE)
    include_directories(${NVSHMEM_INCLUDE})
endif()

set(ALL_SOURCES ${OPS_SOURCES})
message(STATUS "All sources: ${ALL_SOURCES}")

add_library(${PROJECT_NAME} MODULE ${ALL_SOURCES})

# 新增：NVSHMEM 相关的编译选项
if(NVSHMEM_LIB)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        -rdc=true
        >
    )
endif()

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    --extended-lambda
    -gencode=arch=compute_80,code=sm_80
    -gencode=arch=compute_86,code=sm_86
    -gencode=arch=compute_89,code=sm_89
    -gencode=arch=compute_90,code=sm_90
    >
    $<$<COMPILE_LANGUAGE:CXX>:
    -std=c++17
    -D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}
    -fPIC
    >
)

# 新增：NVSHMEM 链接选项
if(NVSHMEM_LIB)
    target_link_directories(${PROJECT_NAME} PRIVATE ${NVSHMEM_LIB})
    
    # 选项1：静态链接（推荐用于生产环境）
    # target_link_libraries(${PROJECT_NAME} PRIVATE
    #     nvshmem
    #     nvidia-ml
    #     cuda
    # )
    
    # 选项2：动态链接（推荐用于开发环境）
    target_link_libraries(${PROJECT_NAME} PRIVATE
        nvshmem_host
        nvshmem_device
    )
endif()

find_library(CUDA_cudadevrt_LIBRARY cudadevrt 
    PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
    NO_DEFAULT_PATH)
if(CUDA_cudadevrt_LIBRARY)
    message(STATUS "Found cudadevrt: ${CUDA_cudadevrt_LIBRARY}")
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_cudadevrt_LIBRARY})
else()
    message(WARNING "cudadevrt library not found, device linking may fail")
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
)

if(Python_LIBRARIES)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${Python_LIBRARIES})
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_DIR}/src
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

# 新增：如果使用 NVSHMEM，添加额外的包含目录
if(NVSHMEM_INCLUDE)
    target_include_directories(${PROJECT_NAME} PRIVATE ${NVSHMEM_INCLUDE})
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES "80;86;89;90"
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET "default"
    CUDA_VISIBILITY_PRESET "default"
    PREFIX ""
    SUFFIX ".so"
    OUTPUT_NAME "rdma_ext"
)

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
)