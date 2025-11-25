cmake_minimum_required(VERSION 3.18)
project(${PROJECT_NAME})

set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUTLASS_ENABLE_SM90 ON CACHE BOOL "Enable SM90 support")

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

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
    ${CUTLASS_PATH}/include
    ${CUTLASS_TOOL_PATH}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(ALL_SOURCES ${OPS_SOURCES})
message(STATUS "All sources: ${ALL_SOURCES}")

add_library(${PROJECT_NAME} MODULE ${ALL_SOURCES})

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
    ${CUTLASS_PATH}/include
    ${CUTLASS_TOOL_PATH}/include
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES "80;86;89;90"
    CUDA_RUNTIME_LIBRARY Shared
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET "default"
    CUDA_VISIBILITY_PRESET "default"
    PREFIX ""
    SUFFIX ".so"
)

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
)