cmake_minimum_required(VERSION 3.10)
project(gradbench)
find_package(benchmark REQUIRED)
set(BENCH_SRCS gradbench.cpp grad.cpp)

set(OpenCV_DIR /Users/garryling/myocv/opencv/build/install/usr/local/lib/cmake/opencv4)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(GRAD_BENCH ${BENCH_SRCS})

target_link_libraries(GRAD_BENCH PRIVATE benchmark::benchmark
                    ${OpenCV_LIBS})
