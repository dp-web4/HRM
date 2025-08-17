
cmake_minimum_required(VERSION 3.18)
project(tiling_mailboxes LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
# Default to sm_87 (Orin). Override with -DCMAKE_CUDA_ARCHITECTURES=72 etc.
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

add_library(pbm STATIC src/mailbox_peripheral.cu)
target_include_directories(pbm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(ftm STATIC src/mailbox_focus.cu)
target_include_directories(ftm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
