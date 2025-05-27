cmake_minimum_required(VERSION 3.12)

# Add the project() command
project(mirage LANGUAGES CXX)

set(TARGET_NAME "mirage")

set(INTEL_CXX_COMPILER OFF)
if(INTEL_CXX_COMPILER)
    message(STATUS "Intel compiler found. Using Intel oneAPI DPC++/C++ Compiler.")
    set(CMAKE_C_COMPILER icx)
    set(CMAKE_CXX_COMPILER icpx)
    
    # Intel-specific compiler optimizations
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -ipo -ffast-math -xHost -qopt-zmm-usage=high")
    
    # Enable OpenMP for Intel
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
else()
    message(STATUS "Intel compiler not found. Using GCC.")
    set(CMAKE_C_COMPILER gcc)
    set(CMAKE_CXX_COMPILER g++)
    
    # GCC-specific compiler optimizations
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -flto -ffast-math")
    
    # Enable OpenMP for GCC
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

set (CMAKE_C_STANDARD 11)
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

file (GLOB SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

file (GLOB HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
        )

# find_package(PkgConfig REQUIRED)
# pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
#     libavformat
#     libavcodec
#     libswscale
#     libavutil
# )

# # Ensures both OpenVINO and the Runtime component are required
# find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# Only the Runtime component is required, OpenVINO package is optional
find_package(OpenVINO COMPONENTS Runtime REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(gflags REQUIRED)
find_package(TBB REQUIRED tbb)
# find_package(glog 0.4.0 REQUIRED)

add_executable(${TARGET_NAME} ${SOURCES} ${HEADERS})

target_include_directories(${TARGET_NAME} 
PRIVATE 
    ${OpenCV_INCLUDE_DIRS} 
    ${Boost_INCLUDE_DIRS}
    # ${TBB_INCLUDE_DIRS}
    # ${LIBAV_INCLUDE_DIRS}
)
# target_link_directories(${TARGET_NAME} PRIVATE ${LIBAV_LIBRARY_DIRS})
target_link_libraries(${TARGET_NAME}
PRIVATE
    openvino::runtime
    gflags
    # TBB::tbb 
    Threads::Threads
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    # PkgConfig::LIBAV
    # boost_system
    # boost_thread
)

# Enable interprocedural optimization
if(INTEL_CXX_COMPILER)
    set_property(TARGET ${TARGET_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    set_property(TARGET ${TARGET_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    # For GCC, we need to add the LTO flag explicitly
    target_link_options(${TARGET_NAME} PRIVATE -flto)
endif()