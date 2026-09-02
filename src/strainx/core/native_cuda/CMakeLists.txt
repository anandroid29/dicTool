cmake_minimum_required(VERSION 3.24)

project(strainx_cuda LANGUAGES CUDA)

add_library(strainx_cuda SHARED strainx_cuda.cu)
set_target_properties(strainx_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "strainx_cuda"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

# Build for the GPU on the machine doing the installation by default. A release
# packager can override this with -DCMAKE_CUDA_ARCHITECTURES="75;86;89;120".
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET strainx_cuda PROPERTY CUDA_ARCHITECTURES native)
endif()

target_compile_definitions(strainx_cuda PRIVATE STRAINX_CUDA_EXPORTS=1)
if(MSVC)
    target_compile_options(strainx_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-diag-suppress=177;-Xcompiler=/O2>)
    target_link_options(strainx_cuda PRIVATE /NODEFAULTLIB:LIBCMT)
else()
    target_compile_options(strainx_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-diag-suppress=177>)
endif()
