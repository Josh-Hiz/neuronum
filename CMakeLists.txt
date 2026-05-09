cmake_minimum_required(VERSION 3.13)
project(neuronum)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build the static library from all source files
add_library(neuronum STATIC
    src/Matrix.cpp
    src/Dataloader.cpp
)

# PUBLIC: both the library and anything linking it can find these headers
target_include_directories(neuronum PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external
)

# Build the test executable and link against the library
add_executable(main test/main.cpp)
target_link_libraries(main PRIVATE neuronum)