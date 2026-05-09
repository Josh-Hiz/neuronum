cmake_minimum_required(VERSION 3.13)
project(neuronum)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(neuronum STATIC
    src/Matrix.cpp
    src/Dataloader.cpp
    src/Dense.cpp
    src/Activation.cpp
    src/Network.cpp
)

target_include_directories(neuronum PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external
)

add_executable(main test/main.cpp)
target_link_libraries(main PRIVATE neuronum)