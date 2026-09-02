cmake_minimum_required(VERSION 3.24)

project(pydic_cuda LANGUAGES CUDA)

add_library(pydic_cuda SHARED pydic_cuda.cu)
set_target_properties(pydic_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "pydic_cuda"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

# Build for the GPU on the machine doing the installation by default. A release
# packager can override this with -DCMAKE_CUDA_ARCHITECTURES="75;86;89;120".
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET pydic_cuda PROPERTY CUDA_ARCHITECTURES native)
endif()

target_compile_definitions(pydic_cuda PRIVATE PYDIC_CUDA_EXPORTS=1)
if(MSVC)
    target_compile_options(pydic_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-diag-suppress=177;-Xcompiler=/O2>)
    target_link_options(pydic_cuda PRIVATE /NODEFAULTLIB:LIBCMT)
else()
    target_compile_options(pydic_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-diag-suppress=177>)
endif()
