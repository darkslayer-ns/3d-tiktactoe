cmake_minimum_required(VERSION 3.14)
project(ttt_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ExecuTorch SDK path override: -DEXECUTORCH_ROOT=/path/to/executorch
# (pip wheel does not ship the C++ headers, so a source checkout is needed
#  for a native build; see README.md)
set(EXECUTORCH_ROOT "" CACHE PATH "Path to ExecuTorch source checkout")

if(NOT EXECUTORCH_ROOT)
  message(WARNING "EXECUTORCH_ROOT not set; building stub only.")
  add_library(ttt_engine SHARED stub.cpp)
  target_include_directories(ttt_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
else()
  add_library(ttt_engine SHARED ttt_engine.cpp)
  target_include_directories(ttt_engine PRIVATE
    ${EXECUTORCH_ROOT}
    ${EXECUTORCH_ROOT}/third-party
  )
  target_link_libraries(ttt_engine PRIVATE executorch)
endif()

target_compile_definitions(ttt_engine PRIVATE EXECUTORCH_RUNTIME_CORE)