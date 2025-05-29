cmake_minimum_required(VERSION 3.10)
project(SimdGaussianBlur)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include Simd headers (adjust path if needed)
include_directories(~/Simd/src/)

# Add executable
add_executable(img_op img_op.cpp)

# Link OpenCV
target_link_libraries(img_op 
            ${OpenCV_LIBS}
            ~/Simd/build/libSimd.a
)
