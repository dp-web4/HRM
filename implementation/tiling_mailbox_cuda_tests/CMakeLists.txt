
cmake_minimum_required(VERSION 3.18)
project(cuda_mailbox_tests LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 14)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

add_library(pbm STATIC ../tiling_mailbox_cuda_pack/src/mailbox_peripheral.cu)
target_include_directories(pbm PUBLIC ../tiling_mailbox_cuda_pack)

add_library(ftm STATIC ../tiling_mailbox_cuda_pack/src/mailbox_focus.cu)
target_include_directories(ftm PUBLIC ../tiling_mailbox_cuda_pack)

add_executable(test_pbm src/test_pbm.cu)
target_include_directories(test_pbm PUBLIC ../tiling_mailbox_cuda_pack)
target_link_libraries(test_pbm PRIVATE pbm cuda cudart)

add_executable(test_ftm src/test_ftm.cu)
target_include_directories(test_ftm PUBLIC ../tiling_mailbox_cuda_pack)
target_link_libraries(test_ftm PRIVATE ftm cuda cudart)
